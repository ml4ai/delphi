from datetime import datetime
from delphi.paths import concept_to_indicator_mapping, data_dir
from .utils import exists, flatMap, flatten, get_data_from_url
from delphi.utils.indra import *
from .random_variables import Delta, Indicator
from typing import *
from indra.statements import Influence, Concept
from fuzzywuzzy import process
from itertools import permutations
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (*p, {"InfluenceStatements": [s for s in sts if nameTuple(s) == p]})
    return edge


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return s.subj_delta, s.obj_delta


def get_respdevs(gb):
    return gb["respdev"]


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
            # To account for discrepancy between Hume and Eidos extractions
            if ev.annotations.get("subj_adjectives") is not None:
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


def get_best_match(indicator: Indicator, items: Iterable[str]) -> str:
    """ Get the best match to an indicator name from a list of items. """
    best_match = process.extractOne(indicator.name, items)[0]
    return best_match


def get_data(filename: str) -> pd.DataFrame:
    """ Create a dataframe out of south_sudan_data.csv """
    df = pd.read_csv(filename, index_col="Variable")
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
    """ Get the value of a particular indicator at a particular date and time. """

    # if indicator.source == "FAO/WDI":
    best_match = get_best_match(indicator, df.index)

    # TODO Fix the above
    df = df.loc[best_match].loc[lambda df: df["Year"] == date.year].loc[lambda df: df["Month"] == date.month]
    if not df["Value"].isna().all():
        indicator_value = float(df["Value"].iloc[0])
        indicator_units = df["Unit"].iloc[0]
    else:
        indicator_value = None
        indicator_units = None

    return (
        (indicator_value, indicator_units)
        if not pd.isna(indicator_value)
        else (None, None)
    )


def get_variable_and_source(x: str):
    """ Process the variable name to make it more human-readable. """
    xs = x.replace("\/", "|").split("/")
    xs = [x.replace("|", "/") for x in xs]
    # return " ".join(xs)
    if xs[0] == "FAO":
        return " ".join(xs[2:]), xs[0]
    else:
        return xs[-1], xs[0]


def construct_concept_to_indicator_mapping(
    n: int = 1, mapping = concept_to_indicator_mapping
) -> Dict[str, List[str]]:
    """ Create a dictionary mapping high-level concepts to low-level indicators """

    df = pd.read_table(
        mapping,
        usecols=[1, 2, 3, 4],
        names=["Concept", "Source", "Indicator", "Score"],
        dtype={"Concept":str, "Source":str, "Indicator":str, "Score":np.float64},
    )
    gb = df.groupby("Concept")

    _dict = {
        k: [
            get_variable_and_source(x)
            for x in v["Indicator"].values[0:n]
        ]
        for k, v in gb
    }
    return _dict


def get_indicators(concept: str, mapping: Dict = None) -> Optional[List[str]]:
    return (
        {x[0]: Indicator(x[0], x[1]) for x in mapping[concept]}
        if concept in mapping
        else None
    )


def make_edges(sts, node_permutations):
    return [
        e
        for e in [make_edge(sts, p) for p in node_permutations]
        if len(e[2]["InfluenceStatements"]) != 0
    ]
