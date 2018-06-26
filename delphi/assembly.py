from .utils import exists, flatMap
from .types import CausalAnalysisGraph, GroupBy, Delta, Indicator
from typing import *
from indra.statements import Influence, Concept
from fuzzywuzzy import process as fuzzywuzzy_process
from datetime import datetime
from functools import singledispatch, lru_cache
from scipy.stats import gaussian_kde
from itertools import permutations
import pandas as pd
import numpy as np
from future.utils import lfilter, lmap
from .paths import concept_to_indicator_mapping

def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return s.subj_delta, s.obj_delta


def get_respdevs(gb: GroupBy):
    return gb["respdev"]

def make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (p[0], p[1], {
        "InfluenceStatements": [s for s in sts if (p[0], p[1]) == nameTuple(s)]
        },
    )
    return edge


def top_grounding(c: Concept, ontology="UN") -> str:
    return c.db_refs["UN"][0][0].split("/")[-1] if "UN" in c.db_refs else c.name


def top_grounding_score(c: Concept, ontology: str = "UN") -> float:
    return c.db_refs[ontology][0][1]


def nameTuple(s: Influence) -> Tuple[str, str]:
    return top_grounding(s.subj), top_grounding(s.obj)


def get_concepts(sts: List[Influence]) -> Set[str]:
    return set(flatMap(nameTuple, sts))


def process_concept_name(name: str) -> str:
    return name.replace("_", " ")


def make_cag_skeleton(sts: List[Influence]) -> CausalAnalysisGraph:
    node_permutations = permutations(get_concepts(sts), 2)
    edges = [e for e in [make_edge(sts, p) for p in node_permutations]
             if len(e[2]["InfluenceStatements"]) != 0]

    return CausalAnalysisGraph(edges)


def filter_statements(sts: List[Influence]) -> List[Influence]:

    return [s for s in sts if is_well_grounded(s) and is_simulable(s)]


def make_model(sts: List[Influence], adjectiveData: str,
        relevant_concepts = Optional[List[str]]) -> CausalAnalysisGraph:
    """ Construct a Delphi model from INDRA statements """

    filtered_statements = filter_statements(sts)

    if relevant_concepts is not None:
        processed_relevant_concepts = [c.replace(' ', '_') for c in relevant_concepts]
        filtered_statements = [s for s in filtered_statements
                if contains_relevant_concept(s, processed_relevant_concepts)]

    cag_skeleton = make_cag_skeleton(filtered_statements)
    cag_with_pdfs = add_conditional_probabilities(cag_skeleton, adjectiveData)
    return cag_with_pdfs


def add_conditional_probabilities(
    CAG: CausalAnalysisGraph, adjectiveData: str
) -> CausalAnalysisGraph:
    # Create a pandas GroupBy object
    gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby("adjective")
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



def constructConditionalPDF(
    gb: GroupBy, rs: np.ndarray, e: Tuple[str, str, Dict]
) -> gaussian_kde:

    sts = e[2]["InfluenceStatements"]

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
                    sts,
                ),
            )
        )
    }

    def responses(adj: Optional[str]) -> np.ndarray:
        return adjectiveResponses[adj] if exists(adj) else rs

    rs_subj=[]
    rs_obj=[]

    for s in sts:
        rs_subj.append(s.subj_delta['polarity']*np.array(responses(get_adjective(s.subj_delta))))
        rs_obj.append(s.obj_delta['polarity']*np.array(responses(get_adjective(s.obj_delta))))

    rs_subj=np.concatenate(rs_subj)
    rs_obj=np.concatenate(rs_obj)

    xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing="xy")

    if len([s for s in sts
        if s.subj_delta['polarity'] == s.obj_delta['polarity']]) == 1:

        xs2, ys2 = -xs1, -ys1
        thetas = np.append(
            np.arctan2(ys1.flatten(), xs1.flatten()),
            np.arctan2(ys2.flatten(), xs2.flatten()),
        )
    else:
        thetas = np.arctan2(ys1.flatten(), xs1.flatten())

    return gaussian_kde(thetas)

def is_simulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x["polarity"], deltas(s))))

@singledispatch
def is_grounded(arg):
    pass


@is_grounded.register(Concept)
def _(concept: Concept, ontology: str = "UN"):
    return ontology in concept.db_refs


@is_grounded.register(Influence)
def _(s: Influence, ontology: str = "UN"):
    return is_grounded(s.subj) and is_grounded(s.obj)


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
        (top_grounding(c) == name)
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


def get_indicator_value(indicator: Indicator,
        date: datetime, df: pd.DataFrame) -> Optional[float]:

    best_match = get_best_match(indicator, df.index)

    col = str(date.year)

    if not col in df.columns:
        return None
    else:
        indicator_value = df[col][best_match]

    return indicator_value if not pd.isna(indicator_value) else None


def set_indicator_values(
        CAG: CausalAnalysisGraph, time: datetime, df: pd.DataFrame
) -> CausalAnalysisGraph:

    for n in CAG.nodes(data=True):
        if n[1]["indicators"] is not None:
            for indicator in n[1]["indicators"]:
                indicator.value = get_indicator_value(indicator, time, df)
                if not indicator.value is None:
                    indicator.stdev = 0.1*abs(indicator.value)
            n[1]["indicators"] = [ind for ind in n[1]['indicators']
                                  if ind.value is not None]

    return CAG


def get_best_match(indicator: Indicator, items: Iterable[str]) -> str:
    return fuzzywuzzy_process.extractOne(indicator.name, items)[0]


def get_faostat_wdi_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, sep="|", index_col='Indicator Name')

def construct_concept_to_indicator_mapping(
        mapping_file: str = concept_to_indicator_mapping,
        n: int = 2) -> Dict[str, List[str]]:
    """ Create a dictionary mapping high-level concepts to low-level indicators """
    df = read_table(concept_to_indicator_mapping, usecols = [1, 3, 4],
            names=['Concept Grounding', 'Indicator Grounding', 'Score'])
    gb = df.groupby('Concept Grounding')

    construct_variable_name = lambda x: x.split('/')[-1]+' '+x.split('/')[-2]
    return {k.split('/')[-1]:[construct_variable_name(x)
            for x in v['Indicator Grounding'].values[0:n]]
            for k, v in gb}

def get_indicators(concept: str, mapping: Dict = None) -> List[str]:
    if mapping is None:
        yaml = YAML()
        with open(concept_to_indicator_mapping, "r") as f:
            mapping = yaml.load(f)

    return (
        [Indicator(x, None) for x in mapping[concept]]
        if concept in mapping else None
    )


def set_indicators(
        G: CausalAnalysisGraph, mapping: Optional[Dict] = None, n: int = 2
) -> CausalAnalysisGraph:
    if mapping is None:
        mapping = construct_concept_to_indicator_mapping(n = n)

    for n in G.nodes(data=True):
        n[1]["indicators"] = get_indicators(n[0].lower().replace(' ', '_'), mapping)

    return G
