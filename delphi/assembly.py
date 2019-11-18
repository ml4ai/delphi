from .utils import exists, take
from .utils.indra import is_well_grounded
from .random_variables import Delta, Indicator
from typing import Tuple, List, Dict, Iterable, Optional, Callable
from indra.statements import Influence
import pandas as pd
import numpy as np
import warnings
from .db import engine


def deltas(s: Influence) -> Tuple[Delta, Delta]:
    return s.subj.delta, s.subj.delta


def get_respdevs(gb):
    return gb["respdev"]


def filter_statements(sts: List[Influence]) -> List[Influence]:
    return [s for s in sts if is_well_grounded(s) and is_simulable(s)]


def constructConditionalPDF(
    gb, rs: np.ndarray, e: Tuple[str, str, Dict]
):
    """ Construct a conditional probability density function for a particular
    AnalysisGraph edge. """

    from .cpp.DelphiPython import KDE
    adjective_response_dict = {}
    all_θs = []

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
                    rs_subj = (
                        stmt.subj.delta.polarity
                        * adjective_response_dict.get(subj_adjective, rs)
                    )

                    for obj_adjective in ev.annotations["obj_adjectives"]:
                        if (
                            obj_adjective in gb.groups
                            and obj_adjective not in adjective_response_dict
                        ):
                            adjective_response_dict[
                                obj_adjective
                            ] = get_respdevs(gb.get_group(obj_adjective))

                        rs_obj = (
                            stmt.obj.delta.polarity
                            * adjective_response_dict.get(obj_adjective, rs)
                        )

                        xs1, ys1 = np.meshgrid(rs_subj, rs_obj, indexing="xy")
                        θs = np.arctan2(
                            σ_Y * ys1.flatten(), σ_X * xs1.flatten()
                        )
                        all_θs.append(θs)

            # Prior
            xs1, ys1 = np.meshgrid(
                stmt.subj.delta.polarity * rs,
                stmt.obj.delta.polarity * rs,
                indexing="xy",
            )
            # TODO - make the setting of σ_X and σ_Y more automated
            θs = np.arctan2(σ_Y * ys1.flatten(), σ_X * xs1.flatten())

    if len(all_θs) == 0:
        all_θs = θs.tolist()
    else:
        all_θs = np.concatenate(all_θs).tolist()
    return KDE(all_θs)


def is_simulable(s: Influence) -> bool:
    return all(map(exists, map(lambda x: x["polarity"], deltas(s))))


def get_variable_and_source(x: str):
    """ Process the variable name to make it more human-readable. """
    xs = x.replace("\/", "|").split("/")
    xs = [x.replace("|", "/") for x in xs]
    if xs[0] == "FAO":
        return " ".join(xs[2:]), xs[0]
    else:
        return xs[-1], xs[0]


def construct_concept_to_indicator_mapping(n: int = 1) -> Dict[str, List[str]]:
    """ Create a dictionary mapping high-level concepts to low-level indicators

    Args:
        n: Number of indicators to return

    Returns:
        Dictionary that maps concept names to lists of indicator names.
    """

    df = pd.read_sql_table("concept_to_indicator_mapping", con=engine)
    gb = df.groupby("Concept")

    _dict = {
        k: [get_variable_and_source(x) for x in take(n, v["Indicator"].values)]
        for k, v in gb
    }
    return _dict


def get_indicator_value(
    indicator: Indicator,
    country: Optional[str] = "Ethiopia",
    state: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    unit: Optional[str] = None,
    fallback_aggaxes: List[str] = ["year", "month"],
    aggfunc: Callable = np.mean,
) -> Tuple[float, Optional[str]]:
    query_base = " ".join(
        [
            f"select * from indicator",
            f"where `Variable` like '{indicator.name}'",
        ]
    )

    query_parts = {"base": query_base}

    if aggfunc == np.mean:
        indicator.aggregation_method = "mean"
    else:
        indicator.aggregation_method = "Unknown"
    if country is not None:
        query_parts["country"] = f"and `Country` is '{country}'"
    if state is not None:
        query_parts["state"] = f"and `State` is '{state}'"
    if year is not None:
        query_parts["year"] = f"and `Year` is '{year}'"
    if month is not None:
        query_parts["month"] = f"and `Month` is '{month}'"
    if unit is not None:
        check_q = query_parts["base"] + f" and `Unit` is '{unit}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected units not found for {indicator.name}! Falling back to default units!"
            )
            query_parts["unit"] = ""
        else:
            query_parts["unit"] = f"and `Unit` is '{unit}'"

    indicator.aggaxes = []
    for constraint in ("country", "state", "year", "month"):
        if constraint not in query_parts:
            indicator.aggaxes.append(constraint)

    query = " ".join(query_parts.values())
    results = list(engine.execute(query))
    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        results = [float(r["Value"]) for r in results if r["Unit"] == unit]
        return aggfunc(results), unit
    else:
        for i, aggregation_axis in enumerate(fallback_aggaxes):
            try:
                indicator.aggaxes.append(aggregation_axis)
                query = " ".join(
                    [
                        query_parts[k]
                        for k in query_parts
                        if k not in fallback_aggaxes[: i + 1]
                    ]
                )

                results = list(engine.execute(query))

                if results == []:
                    continue

                # Handling the case where the same indicator name is present
                # with different units - we must not aggregate these values!

                # If there are multiple possible units, use the first in the
                # (alphabetically sorted) set of possible units as a default.

                if not all(map(lambda r: r["Unit"] is None, results)):
                    unit = sorted(
                        list(
                            {
                                r["Unit"]
                                for r in results
                                if r["Unit"] is not None
                            }
                        )
                    )[0]
                agg = aggfunc(
                    [float(r["Value"]) for r in results if r["Unit"] == unit]
                )
                return agg, unit

            except StopIteration:
                raise ValueError(
                    f"No data found for the indicator {indicator.name}!"
                    "Try using additional aggregation axes."
                )
