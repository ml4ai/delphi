import json
import pickle
import datetime
import numpy as np
from typing import List, Dict, Union
from delphi.types import CausalAnalysisGraph
from functools import partial
from future.utils import lmap
from delphi.core import construct_default_initial_state, get_latent_state_components
from delphi.types import Indicator
from typing import Dict

def _process_datetime(indicator_dict: Dict):
    time = indicator_dict.get('time')
    indicator_dict['time'] = str(time)
    return indicator_dict

def _export_node(CAG: CausalAnalysisGraph, n) -> Dict[str, Union[str, List[str]]]:
    """ Return dict suitable for exporting to JSON.

    Args:
        CAG: The causal analysis graph
        n: A dict representing the data in a networkx CausalAnalysisGraph node.

    Returns:
        The node dict with additional fields for name, units, dtype, and
        arguments.

    """
    node_dict = {
        "name" : n[0],
        "units" : _get_units(n[0]),
        "dtype" : _get_dtype(n[0]),
        "arguments" : list(CAG.predecessors(n[0])),
    }
    if not n[1].get('indicators') is None:
        node_dict['indicators'] = [_process_datetime(ind.__dict__) for ind in n[1]["indicators"]]
    else:
        node_dict['indicators'] = None

    return node_dict


def _export_edge(e):
    return { "source": e[0], "target": e[1], "CPT": _construct_CPT(e),
            "polyfit": _get_polynomial_fit(e), }


def export_to_ISI(CAG: CausalAnalysisGraph, args) -> None:

    s0 = construct_default_initial_state(get_latent_state_components(CAG))
    s0.to_csv(args.output_variables_path, index_label="variable")

    model = {
        "name": "Dynamic Bayes Net Model",
        "dateCreated": str(datetime.datetime.now()),
        "variables": lmap(partial(_export_node, CAG), CAG.nodes(data=True)),
    }

    with open(args.output_cag_json, "w") as f:
        json.dump(model, f, indent=2)

    for e in CAG.edges(data=True):
        del e[2]["InfluenceStatements"]

    with open(args.output_dressed_cag, "wb") as f:
        pickle.dump(CAG, f)


def _get_units(n: str) -> str:
    return "units"


def _get_dtype(n: str) -> str:
    return "real"


def _construct_CPT(e, res=100):
    kde = e[2]["ConditionalProbability"]
    arr = np.squeeze(kde.dataset)
    X = np.linspace(min(arr), max(arr), res)
    Y = kde.evaluate(X) * (X[1] - X[0])
    return {"theta": X.tolist(), "P(theta)": Y.tolist()}


def _get_polynomial_fit(e, deg = 7, res = 100):
    kde = e[2]["ConditionalProbability"]
    arr = np.squeeze(kde.dataset)
    X = np.linspace(min(arr), max(arr), res)
    Y = kde.evaluate(X) * (X[1] - X[0])
    coefs = np.polynomial.polynomial.polyfit(X, Y, deg=deg)
    return {"degree": deg, "coefficients" : list(coefs)}


def to_json(CAG: CausalAnalysisGraph, Δt: float = 1.0):
    with open("cag.json", "w") as f:
        json.dump(
            {
                "name": "Dynamic Bayes Net Model",
                "dateCreated": str(datetime.datetime.now()),
                "variables": lmap(
                    partial(_export_node, CAG), CAG.nodes(data=True)
                ),
                "timeStep": str(Δt),
                "edge_data": lmap(_export_edge, CAG.edges(data=True),
                ),
            },
            f,
            indent=2,
        )
