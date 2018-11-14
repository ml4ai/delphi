from datetime import datetime
from delphi.paths import concept_to_indicator_mapping, data_dir
from .utils import exists, flatMap, flatten, get_data_from_url
from .random_variables import Delta, Indicator
from typing import *
from indra.statements import Influence, Concept
from fuzzywuzzy import process
from functools import singledispatch, lru_cache
from itertools import permutations
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def get_valid_statements_for_modeling(sts: List[Influence]) -> List[Influence]:
    """ Select INDRA statements that can be used to construct a Delphi model
    from a given list of statements. """

    return [
        s
        for s in sts
        if is_grounded(s)
        and (s.subj_delta["polarity"] is not None)
        and (s.obj_delta["polarity"] is not None)
    ]


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return s.subj_delta, s.obj_delta


def get_respdevs(gb):
    return gb["respdev"]


def top_grounding(c: Concept) -> str:
    """ Return the top-scoring grounding from the UN ontology. """
    return (
        c.db_refs["UN"][0][0].split("/")[-1] if "UN" in c.db_refs else c.name
    )


def top_grounding_score(c: Concept) -> float:
    return c.db_refs["UN"][0][1]


def nameTuple(s: Influence) -> Tuple[str, str]:
    """ Returns a 2-tuple consisting of the top groundings of the subj and obj
    of an Influence statement. """
    return top_grounding(s.subj), top_grounding(s.obj)


def get_concepts(sts: List[Influence]) -> Set[str]:
    """ Get a set of all unique concepts in the list of INDRA statements. """
    return set(flatMap(nameTuple, sts))


def process_concept_name(name: str) -> str:
    """ Remove underscores from concept name. """
    return name.replace("_", " ")


def filter_statements(sts: List[Influence]) -> List[Influence]:
    return [s for s in sts if is_well_grounded(s) and is_simulable(s)]


def constructConditionalPDF(
    gb, rs: np.ndarray, e: Tuple[str, str, Dict]
) -> gaussian_kde:
    """ Construct a conditional probability density function for a particular
    AnalysisGraph edge. """

    adjective_response_dict = {}
    all_thetas = []
    for stmt in e[2]["InfluenceStatements"]:
        for ev in stmt.evidence:
            for subj_adjective in ev.annotations["subj_adjectives"]:
                if (
                    subj_adjective in gb.groups
                    and subj_adjective not in adjective_response_dict
                ):
                    adjective_response_dict[subj_adjective] = get_respdevs(
                        gb.get_group(subj_adjective)
                    )
                rs_subj = stmt.subj_delta[
                    "polarity"
                ] * adjective_response_dict.get(subj_adjective, rs)

                for obj_adjective in ev.annotations["obj_adjectives"]:
                    if (
                        obj_adjective in gb.groups
                        and obj_adjective not in adjective_response_dict
                    ):
                        adjective_response_dict[obj_adjective] = get_respdevs(
                            gb.get_group(obj_adjective)
                        )

                    rs_obj = stmt.obj_delta[
                        "polarity"
                    ] * adjective_response_dict.get(obj_adjective, rs)

                    xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing="xy")
                    thetas = np.arctan2(ys1.flatten(), xs1.flatten())
                    all_thetas.append(thetas)

            # Prior
            xs1, ys1 = np.meshgrid(
                stmt.subj_delta["polarity"] * rs,
                stmt.obj_delta["polarity"] * rs,
                indexing="xy",
            )
            thetas = np.arctan2(ys1.flatten(), xs1.flatten())
            all_thetas.append(thetas)

    if len(all_thetas) == 1:
        return gaussian_kde(all_thetas)
    else:
        return gaussian_kde(np.concatenate(all_thetas))


def is_simulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x["polarity"], deltas(s))))


@singledispatch
def is_grounded():
    pass


@is_grounded.register(Concept)
def _(c: Concept) -> bool:
    """ Check if a concept is grounded """
    return (
        "UN" in c.db_refs
        and c.db_refs["UN"][0][0].split("/")[1] != "properties"
    )


@is_grounded.register(Influence)
def _(s: Influence) -> bool:
    """ Check if an Influence statement is grounded """
    return is_grounded(s.subj) and is_grounded(s.obj)


@singledispatch
def is_well_grounded():
    pass


@is_well_grounded.register(Concept)
def _(c: Concept, cutoff: float = 0.7) -> bool:
    """Check if a concept has a high grounding score. """

    return is_grounded(c) and (top_grounding_score(c) >= cutoff)


@is_well_grounded.register(Influence)
def _(s: Influence, cutoff: float = 0.7) -> bool:
    """ Returns true if both subj and obj are grounded to the UN ontology. """

    return all(map(lambda c: is_well_grounded(c, cutoff), s.agent_list()))


def is_grounded_to_name(c: Concept, name: str, cutoff=0.7) -> bool:
    """ Check if a concept is grounded to a given name. """
    return (top_grounding(c) == name) if is_well_grounded(c, cutoff) else False


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
    """ Returns true if a given Influence statement has a relevant concept, and
    false otherwise. """

    return any(
        map(lambda c: contains_concept(s, c, cutoff=cutoff), relevant_concepts)
    )


def get_best_match(indicator: Indicator, items: Iterable[str]) -> str:
    """ Get the best match to an indicator name from a list of items. """
    best_match = process.extractOne(indicator.name, items)[0]
    return best_match


def get_data(filename: str) -> pd.DataFrame:
    """ Create a dataframe out of south_sudan_data.csv """
    df = pd.read_csv(filename, sep="|", index_col="Indicator Name")
    return df


def get_mean_precipitation(year: int):
    """ Workaround to get the precipitation from CYCLES. """
    url = "http://vision.cs.arizona.edu/adarsh/export/demos/data/weather.dat"
    df = pd.read_table(get_data_from_url(url))
    df.columns = df.columns.str.strip()
    df.columns = [c + f" ({df.iloc[0][c].strip()})" for c in df.columns]
    df.drop([0], axis=0, inplace=True)
    df["DATE (YYYY-MM-DD)"] = pd.to_datetime(
        df["DATE (YYYY-MM-DD)"], format="%Y-%m-%d"
    )
    return (
        df.loc[
            (datetime(year, 1, 1) < df["DATE (YYYY-MM-DD)"])
            & (df["DATE (YYYY-MM-DD)"] < datetime(year, 12, 31))
        ]["PRECIPITATION (mm)"]
        .values.astype(float)
        .mean()
    )


def get_indicator_value(
    indicator: Indicator, date: datetime, df: pd.DataFrame
) -> Optional[float]:
    """ Get the value of a particular indicator at a particular date and time.
    """

    if indicator.source == "FAO/WDI":
        best_match = get_best_match(indicator, df.index)

        year = str(date.year)
        if not year in df.columns:
            return None
        else:
            indicator_value = df[year][best_match]
            indicator_units = df.loc[best_match]["Unit"]

        return (
            (indicator_value, indicator_units)
            if not pd.isna(indicator_value)
            else (None, indicator_units)
        )

    elif indicator.source == "CYCLES":
        return get_mean_precipitation(date.year), "mm"


def process_variable_name(x: str):
    """ Process the variable name to make it more human-readable. """
    xs = x.replace("\/", "|").split("/")
    xs = [x.replace("|", "/") for x in xs]
    xs.reverse()
    return " ".join(xs[0:2])


def construct_concept_to_indicator_mapping(
    n: int = 2, mapping=concept_to_indicator_mapping
) -> Dict[str, List[str]]:
    """ Create a dictionary mapping high-level concepts to low-level indicators """

    df = pd.read_table(
        mapping,
        usecols=[1, 3, 4],
        names=["Concept Grounding", "Indicator Grounding", "Score"],
    )
    gb = df.groupby("Concept Grounding")

    construct_variable_name = (
        lambda x: x.split("/")[-1] + " " + x.split("/")[-2]
    )
    return {
        k.split("/")[-1]: [
            process_variable_name(x)
            for x in v["Indicator Grounding"].values[0:n]
        ]
        for k, v in gb
    }


def get_indicators(concept: str, mapping: Dict = None) -> Optional[List[str]]:
    return (
        [Indicator(x, "FAO/WDI") for x in mapping[concept]]
        if concept in mapping
        else None
    )
