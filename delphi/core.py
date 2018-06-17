import sys
import json
import pickle
import datetime
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from tqdm import trange, tqdm
from itertools import permutations, cycle, chain
from indra.statements import Influence, Concept
from networkx import DiGraph
from indra.sources import eidos
from scipy.stats import gaussian_kde
from pandas import Series, DataFrame, read_csv, isna
from glob import glob
from fuzzywuzzy import process, fuzz

from functools import partial, lru_cache, singledispatch
from delphi.types import GroupBy, Delta, CausalAnalysisGraph, Node, Indicator
from delphi.paths import data_dir, concept_to_indicator_mapping
from future.utils import lmap, lfilter, lzip
from delphi.utils import (
    flatMap,
    compose,
    iterate,
    ltake,
    exists,
    repeatfunc,
    take,
)

from typing import (
    List,
    Tuple,
    Set,
    Callable,
    Optional,
    Any,
    Dict,
    IO,
    Iterable,
    Union,
    NewType,
)


def construct_default_initial_state(s_index: List[str]) -> Series:
    return Series(ltake(len(s_index), cycle([100.0, 1.0])), s_index)


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    """
    Args:
        s: An INDRA Influence statement.
    Returns:
        A 2-tuple containing the subj_delta and obj_delta attributes of the
        Influence statements.
    """
    return s.subj_delta, s.obj_delta


def top_grounding(c: Concept, ontology="UN") -> str:
    return c.db_refs["UN"][0][0].split("/")[-1] if "UN" in c.db_refs else c.name


def nameTuple(s: Influence) -> Tuple[str, str]:
    return _process_concept_name(top_grounding(s.subj)), _process_concept_name(
        top_grounding(s.obj)
    )


def get_indra_statements_from_directory(directory: str) -> Iterable[Influence]:
    """ Returns a list of INDRA statements from a directory containing JSON-LD
    output from Eidos. """
    return chain.from_iterable(
        map(
            lambda ep: ep.statements,
            map(eidos.process_json_ld_file, tqdm(glob(directory))),
        )
    )


@singledispatch
def is_grounded(arg):
    pass


@is_grounded.register(Concept)
def _(concept: Concept, ontology: str = "UN"):
    return ontology in concept.db_refs


@is_grounded.register(Influence)
def _(s: Influence, ontology: str = "UN"):
    return is_grounded(s.subj) and is_grounded(s.obj)


def top_grounding_score(c: Concept, ontology: str = "UN") -> float:
    return c.db_refs[ontology][0][1]


@singledispatch
def is_well_grounded():
    pass


@is_well_grounded.register(Concept)
def _(c: Concept, ontology: str = "UN", cutoff: float = 0.7) -> bool:

    return is_grounded(c, ontology) and (
        top_grounding_score(c, ontology) >= cutoff
    )


@lru_cache(maxsize=32)
@is_well_grounded.register(Influence)
def _(s: Influence, ontology: str = "UN", cutoff: float = 0.7) -> bool:
    """ Returns true if both subj and obj are grounded to the specified
    ontology"""

    return all(
        map(lambda c: is_well_grounded(c, ontology, cutoff), s.agent_list())
    )


def is_grounded_to_name(c: Concept, name: str, cutoff=0.7) -> bool:
    return (
        (c.db_refs["UN"][0][0].split("/")[-1] == name)
        if is_well_grounded(c, "UN", cutoff)
        else False
    )


def contains_concept(s: Influence, concept_name: str, cutoff=0.7) -> bool:
    return any(
        map(
            lambda c: is_grounded_to_name(c, concept_name, cutoff),
            s.agent_list(),
        )
    )


def contains_relevant_concept(
    s: Influence, relevant_concepts: List[str], cutoff=0.7
) -> bool:
    return any(
        map(lambda c: contains_concept(s, c, cutoff=cutoff), relevant_concepts)
    )


def get_concepts(sts: List[Influence]) -> Set[str]:
    return set(flatMap(nameTuple, sts))


def _process_concept_name(concept: str) -> str:
    return concept.replace("_", " ")


def _make_CAG_nodes(concepts: Set[str]) -> List[str]:
    return [_process_concept_name(concept) for concept in concepts]


def _make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (
        p[0],
        p[1],
        {
            "InfluenceStatements": lfilter(
                lambda s: (p[0], p[1]) == nameTuple(s), sts
            )
        },
    )
    return edge


def construct_CAG_skeleton(sts: List[Influence]) -> CausalAnalysisGraph:
    return CausalAnalysisGraph(
        lfilter(
            lambda e: len(e[2]["InfluenceStatements"]) != 0,
            lmap(
                partial(_make_edge, sts),
                permutations(_make_CAG_nodes(get_concepts(sts)), 2),
            ),
        )
    )


def get_respdevs(gb: GroupBy) -> np.ndarray:
    return gb["respdev"]


def isSimulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x["polarity"], deltas(s))))


def constructConditionalPDF(
    gb: GroupBy, rs: np.ndarray, e: Tuple[str, str, Dict]
) -> gaussian_kde:

    simulableStatements = lfilter(isSimulable, e[2]["InfluenceStatements"])

    if not simulableStatements:
        return None

    else:

        # Make a adjective-response dict.

        def get_adjective(d: Delta) -> Optional[str]:
            """ Get the first adjective from subj_delta or obj_delta """

            if isinstance(d["adjectives"], list):
                if d["adjectives"]:
                    adj = d["adjectives"][0]
                else:
                    adj = None
            else:
                adj = d["adjectives"]

            return adj if adj in gb.groups.keys() else None

        adjectiveResponses = {
            a: get_respdevs(gb.get_group(a))
            for a in set(
                filter(
                    exists,
                    flatMap(
                        lambda s: lmap(get_adjective, deltas(s)),
                        simulableStatements,
                    ),
                )
            )
        }

        def responses(adj: Optional[str]) -> np.ndarray:
            return adjectiveResponses[adj] if exists(adj) else rs

        rs_subj, rs_obj = list(
            *zip(
                lmap(
                    lambda s: map(
                        lambda d: d["polarity"]
                        * np.array(responses(get_adjective(d))),
                        deltas(s),
                    ),
                    simulableStatements,
                )
            )
        )[0]

        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing="xy")

        if (
            len(
                lmap(
                    lambda s: s.subj_delta["polarity"]
                    == s.obj_delta["polarity"],
                    simulableStatements,
                )
            )
            == 1
        ):

            xs2, ys2 = -xs1, -ys1
            thetas = np.append(
                np.arctan2(ys1.flatten(), xs1.flatten()),
                np.arctan2(ys2.flatten(), xs2.flatten()),
            )
        else:
            thetas = np.arctan2(ys1.flatten(), xs1.flatten())

        return gaussian_kde(thetas)


def add_conditional_probabilities(
    CAG: CausalAnalysisGraph, adjectiveData: str
) -> CausalAnalysisGraph:
    # Create a pandas GroupBy object
    gb = read_csv(adjectiveData, delim_whitespace=True).groupby("adjective")
    rs = (
        gaussian_kde(
            flatMap(
                lambda g: gaussian_kde(get_respdevs(g[1]))
                .resample(20)[0]
                .tolist(),
                gb,
            )
        )
        .resample(100)[0]
        .tolist()
    )

    for e in CAG.edges(data=True):
        e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)

    return CAG


def create_dressed_CAG(
    sts: List[Influence], adjectiveData: str
) -> CausalAnalysisGraph:
    return add_conditional_probabilities(
        construct_CAG_skeleton(sts), adjectiveData
    )


def get_latent_state_components(CAG: CausalAnalysisGraph) -> List[str]:
    return flatMap(lambda a: (a, f"∂({a})/∂t"), CAG.nodes())


def initialize_transition_matrix(cs: List[str], Δt: float = 1) -> DataFrame:
    A = DataFrame(np.identity(len(cs)), cs, cs)
    for c in cs[::2]:
        A[f"∂({c})/∂t"][f"{c}"] = Δt
    return A


def sample_transition_matrix(
    CAG: CausalAnalysisGraph, Δt: float = 1.0
) -> DataFrame:
    A = initialize_transition_matrix(get_latent_state_components(CAG))

    for e in CAG.edges(data=True):
        if "ConditionalProbability" in e[2].keys():
            β = np.tan(e[2]["ConditionalProbability"].resample(1)[0][0])
            A[f"∂({e[0]})/∂t"][f"∂({e[1]})/∂t"] = β * Δt

    return A


def sample_sequence(
    CAG: CausalAnalysisGraph, s0: np.ndarray, n_steps: int, Δt: float = 1.0
) -> List[np.ndarray]:

    A = sample_transition_matrix(CAG, Δt).values
    return take(n_steps, iterate(lambda s: A @ s, s0))


def sample_sequences(
    CAG: CausalAnalysisGraph,
    s0: Series,
    steps: int,
    samples: int,
    Δt: float = 1.0,
) -> List[Series]:
    """ Sample a collection of sequences for a CAG """

    s0 = s0.values[np.newaxis].T
    return take(samples, repeatfunc(sample_sequence, CAG, s0, steps, Δt))


def construct_executable_model(sts: List[Influence]) -> CausalAnalysisGraph:
    return add_conditional_probabilities(construct_CAG_skeleton(sts))


def load_model(filename: str) -> CausalAnalysisGraph:
    with open(filename, "rb") as f:
        CAG = pickle.load(f)
    return CAG


def emission_function(x):
    return np.random.normal(x, 0.01 * abs(x))


def write_sequences_to_file(
    CAG: CausalAnalysisGraph, seqs, output_filename: str
) -> None:

    with open(output_filename, "w") as f:
        f.write(
            ",".join(
                ["seq_no", "time_slice", *get_latent_state_components(CAG)[::2]]
            )
            + "\n"
        )
        for n, s in enumerate(seqs):
            for t, l in enumerate(s):
                vs = ",".join([str(x) for x in l.T[0][::2]])
                f.write(",".join([str(n), str(t), vs]) + "\n")


def get_indicators(concept: str, mapping: Dict = None) -> List[str]:
    if mapping is None:
        yaml = YAML()
        with open(concept_to_indicator_mapping, "r") as f:
            mapping = yaml.load(f)

    return (
        [Indicator(x[0], x[1]['source']) for x in
            mapping["concepts"][concept]["indicators"].items()]
        if concept in mapping["concepts"]
        else None
    )


def set_indicators(
    G: CausalAnalysisGraph, mapping: Optional[Dict] = None
) -> CausalAnalysisGraph:
    if mapping is None:
        mapping_yaml_file = concept_to_indicator_mapping
        yaml = YAML()
        with open(mapping_yaml_file, "r") as f:
            mapping = yaml.load(f)

    for n in G.nodes(data=True):
        n[1]["indicators"] = get_indicators(n[0], mapping)

    return G


def get_faostat_wdi_data(filename: str) -> DataFrame:
    return read_csv(filename, sep="|", index_col='Indicator Name')


def get_best_match(indicator_name: str, items: Iterable[str]) -> str:
    return process.extractOne(
        indicator_name, items, scorer=fuzz.token_sort_ratio
    )[0]

def bind(x, f, *args):
    return x if x is None else f(x, *args)


def get_indicator_data(
    indicator: Indicator, df: DataFrame, year: str = None
) -> DataFrame:

    best_match = get_best_match(indicator.name, df.index)
    return df.loc[best_match]


def get_indicator_value_for_year(
    indicator: Indicator, year: str, df
) -> Optional[float]:

    if not year in df.columns:
        return None
    else:
        indicator_value = get_indicator_data(indicator, df)[year]

    return indicator_value if not isna(indicator_value) else None


def set_indicator_initial_values(
    CAG: CausalAnalysisGraph, year: str, df: DataFrame
) -> CausalAnalysisGraph:

    for n in CAG.nodes(data=True):
        if n[1]["indicators"] is not None:
            for indicator in n[1]["indicators"]:
                indicator.initial_value = get_indicator_value_for_year(indicator, year, df)
            n[1]["indicators"] = lfilter(
                lambda ind: ind.initial_value is not None, n[1]["indicators"]
            )

    return CAG
