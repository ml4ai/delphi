from datetime import datetime
from .paths import db_path
from .utils import exists, flatMap, flatten, get_data_from_url
from .utils.indra import *
from .random_variables import Delta, Indicator
from typing import *
from indra.statements import Influence, Concept
from fuzzywuzzy import process
from itertools import permutations
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine


def make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (*p, {"InfluenceStatements": [s for s in sts if nameTuple(s) == p]})
    return edge


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return s.subj_delta, s.obj_delta


def get_respdevs(gb):
    return gb["respdev"]


def filter_statements(sts: List[Influence]) -> List[Influence]:
    return [s for s in sts if is_well_grounded(s) and is_simulable(s)]


def constructConditionalPDF(
    gb, rs: np.ndarray, e: Tuple[str, str, Dict]
) -> gaussian_kde:
    """ Construct a conditional probability density function for a particular
    AnalysisGraph edge. """

    adjective_response_dict = {}
    all_thetas = []

    # Setting σ_X and σ_Y that are in Eq. 1.21 of the model document.
    # This assumes that the real-valued variables representing the abstract
    # concepts are on the order of 1.0.
    # TODO Make this more general.

    σ_X = σ_Y = 0.1

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
                            adjective_response_dict[
                                obj_adjective
                            ] = get_respdevs(gb.get_group(obj_adjective))

                        rs_obj = stmt.obj_delta[
                            "polarity"
                        ] * adjective_response_dict.get(obj_adjective, rs)

                        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing="xy")
                        thetas = np.arctan2(σ_Y * ys1.flatten(), xs1.flatten())
                        all_thetas.append(thetas)

            # Prior
            xs1, ys1 = np.meshgrid(
                stmt.subj_delta["polarity"] * rs,
                stmt.obj_delta["polarity"] * rs,
                indexing="xy",
            )
            # TODO - make the setting of σ_X and σ_Y more automated
            thetas = np.arctan2(σ_Y * ys1.flatten(), σ_X * xs1.flatten())

    if len(all_thetas) == 0:
        all_thetas.append(thetas)
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
    df = pd.read_csv(filename)
    return df


def get_indicator_value(
    indicator: Indicator, date: datetime
) -> Optional[float]:
    """ Get the value of a particular indicator at a particular date and time. """

    engine = create_engine("sqlite:///" + str(db_path), echo=False)
    variable_names = [
        x[0]
        for x in engine.execute(
            f"select distinct `Variable` from indicator"
        ).fetchall()
    ]
    best_match = get_best_match(indicator, variable_names)

    # TODO Devise a strategy to get rid of the fetchone() call at the end of the
    # expression below (i.e. add month support instead of taking the first
    # available result.)

    result = engine.execute(
        " ".join(
            [
                f"select * from indicator where `Variable` like '{best_match}'",
                "and `Value` is not null",
                f"and `Year` is {date.year}",
            ]
        )
    ).fetchone()

    # TODO devise a strategy to deal with missing month values

    if not result is None:
        indicator_value = float(result["Value"])
        indicator_units = result["Unit"]
    else:
        indicator_value = None
        indicator_units = None

    return (
        (indicator_value, indicator_units)
        if not indicator_value is None
        else (None, None)
    )


def get_variable_and_source(x: str):
    """ Process the variable name to make it more human-readable. """
    xs = x.replace("\/", "|").split("/")
    xs = [x.replace("|", "/") for x in xs]
    if xs[0] == "FAO":
        return " ".join(xs[2:]), xs[0]
    else:
        return xs[-1], xs[0]


def construct_concept_to_indicator_mapping(n: int = 1) -> Dict[str, List[str]]:
    """ Create a dictionary mapping high-level concepts to low-level indicators """

    engine = create_engine(f"sqlite:///{str(db_path)}", echo=False)
    df = pd.read_sql_table("concept_to_indicator_mapping", con=engine)
    gb = df.groupby("Concept")

    _dict = {
        k: [get_variable_and_source(x) for x in v["Indicator"].values[0:n]]
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
