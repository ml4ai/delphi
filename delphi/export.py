import json
import pickle
import datetime
from types import DiGraph
from functools import partial
from future.utils import lmap
from core import construct_default_initial_state, get_latent_state_components


def export_to_ISI(CAG: DiGraph, args) -> None:

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


def _export_node(CAG: DiGraph, n) -> Dict[str, Union[str, List[str]]]:
    """ Return dict suitable for exporting to JSON.

    Args:
        CAG: The causal analysis graph
        n: A dict representing the data in a networkx DiGraph node.

    Returns:
        The node dict with additional fields for name, units, dtype, and
        arguments.

    """
    n[1]["name"] = n[0]
    n[1]["units"] = _get_units(n[0])
    n[1]["dtype"] = _get_dtype(n[0])
    n[1]["arguments"] = list(CAG.predecessors(n[0]))
    return n[1]


def _export_edge(CAG: DiGraph, e):
    return {"source": e[0], "target": e[1], "CPT": e[2]["CPT"]}


def _construct_CPT(e, res=100):
    kde = e[2]["ConditionalProbability"]
    arr = np.squeeze(kde.dataset)
    X = np.linspace(min(arr), max(arr), res)
    Y = kde.evaluate(X) * (X[1] - X[0])
    return {"beta": X.tolist(), "P(beta)": Y.tolist()}


def export_to_CRA(CAG: DiGraph, Δt):
    with open("cra_cag.json", "w") as f:
        json.dump(
            {
                "name": "Dynamic Bayes Net Model",
                "dateCreated": str(datetime.datetime.now()),
                "variables": lmap(
                    partial(_export_node, CAG), CAG.nodes(data=True)
                ),
                "timeStep": Δt,
                "CPTs": lmap(
                    lambda e: {
                        "source": e[0],
                        "target": e[1],
                        "CPT": _construct_CPT(e),
                    },
                    CAG.edges(data=True),
                ),
            },
            f,
            indent=2,
        )
