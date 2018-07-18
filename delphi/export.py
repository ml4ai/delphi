import json
from typing import Dict, Union, List
import numpy as np
import networkx as nx
from .utils import _insert_line_breaks
from .AnalysisGraph import AnalysisGraph
from networkx import DiGraph
from pygraphviz import AGraph
import pickle
from .execution import construct_default_initial_state

# ==========================================================================
# Export
# ==========================================================================

def _process_datetime(indicator_dict: Dict):
    time = indicator_dict.get("time")
    indicator_dict["time"] = str(time)
    return indicator_dict


def _export_edge(e):
    return {
        "source": e[0],
        "target": e[1],
        "CPT": _construct_CPT(e),
        "polyfit": _get_polynomial_fit(e),
        "InfluenceStatements": [s.to_json() for s in e[2]['InfluenceStatements']]
    }


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


def _get_polynomial_fit(e, deg=7, res=100):
    kde = e[2]["ConditionalProbability"]
    arr = np.squeeze(kde.dataset)
    X = np.linspace(min(arr), max(arr), res)
    Y = kde.evaluate(X) * (X[1] - X[0])
    coefs = np.polynomial.polynomial.polyfit(X, Y, deg=deg)
    return {"degree": deg, "coefficients": list(coefs)}

def to_agraph(G: DiGraph, *args, **kwargs) -> AGraph:
    """ Exports AnalysisGraph to pygraphviz AGraph

    Args:
        G
        kwargs

    Returns:
        AGraph
    """

    A = AGraph(directed=True)

    A.graph_attr.update(
        {
            "dpi": 227,
            "fontsize": 20,
            "rankdir": kwargs.get("rankdir", "TB"),
            "fontname": "Gill Sans",
        }
    )

    A.node_attr.update(
        {
            "shape": "rectangle",
            "color": "#650021",
            "style": "rounded",
            "fontname": "Gill Sans",
        }
    )

    nodes_with_indicators = [
        n
        for n in G.nodes(data=True)
        if n[1].get("indicators") is not None
    ]

    n_max = 0
    for e in G.edges(data=True):
        n = len(e[2]['InfluenceStatements'])
        if n > n_max:
            n_max = n

    color_str = "#650021"
    for e in G.edges(data=True):
        opacity = len(e[2]['InfluenceStatements'])/n_max
        h = (opacity*255).hex()
        c_str=color_str+h[4:6]
        A.add_edge(
                e[0].capitalize(),
                e[1].capitalize(),
                color=c_str,
                arrowsize=0.5,
                )

    # Drawing indicator variables

    if kwargs.get("indicators"):
        for n in nodes_with_indicators:
            for ind in n[1]["indicators"]:
                node_label = _insert_line_breaks(ind.name)
                A.add_node(
                    node_label,
                    style="rounded, filled",
                    fillcolor="lightblue",
                )
                A.add_edge(n[0].capitalize(), node_label, color='royalblue4')

    # Drawing indicator values
    if kwargs.get("indicator_values"):
        for n in nodes_with_indicators:
            indicators = [
                i for i in n[1]["indicators"] if i.value is not None
            ]
            for ind in indicators:
                if ind.unit is not None:
                    units = f' {ind.unit}'
                else:
                    units=''
                ind_label = _insert_line_breaks(ind.name)

                node_label = '{:.2f}'.format(ind.value) + units
                A.add_node(
                    node_label,
                    shape='plain',
                    # style="rounded, filled",
                    fillcolor="white",
                    color="royalblue",
                )
                A.add_edge(ind_label, node_label, color='lightblue',
                        arrowhead='none')

    if kwargs.get("nodes_to_highlight") is not None:
        for n in kwargs["nodes_to_highlight"]:
            A.add_node(n.capitalize(), fontcolor="royalblue")

    if kwargs.get('graph_label') is not None:
        A.graph_attr['label'] = kwargs['graph_label']

    return A


def export_node(G: AnalysisGraph, n) -> Dict[str, Union[str, List[str]]]:
    """ Return dict suitable for exporting to JSON.

    Args:
        n: A dict representing the data in a networkx AnalysisGraph node.

    Returns:
        The node dict with additional fields for name, units, dtype, and
        arguments.

    """
    node_dict = {
        "name": n[0],
        "units": _get_units(n[0]),
        "dtype": _get_dtype(n[0]),
        "arguments": list(G.predecessors(n[0])),
    }
    if not n[1].get("indicators") is None:
        node_dict["indicators"] = [
            _process_datetime(ind.__dict__) for ind in n[1]["indicators"]
        ]
    else:
        node_dict["indicators"] = None

    return node_dict

def export(
    G: AnalysisGraph,
    format="full",
    json_file="delphi_cag.json",
    pickle_file="delphi_cag.pkl",
    variables_file="variables.csv",
):
    """ Export the model in various formats.

    Args:
        G
        format
        json_file
        pickle_file
        variables_file
    """

    if format == "full":
        _to_json(G, json_file)
        _pickle(G, pickle_file)
        export_default_initial_values(G, variables_file)

    if format == "agraph":
        return to_agraph(G)

    if format == "json":
        _to_json(G, json_file)

def _pickle(G: AnalysisGraph, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(G, f)

def _to_json(G: AnalysisGraph, filename: str):
    """ Export the CAG to JSON """
    with open(filename, "w") as f:
        json.dump(
            {
                "name": G.name,
                "dateCreated": str(datetime.now()),
                "variables": lmap(partial(export_node, G), G.nodes(data=True)),
                "timeStep": str(G.Δt),
                "edge_data": lmap(_export_edge, G.edges(data=True)),
            },
            f,
            indent=2,
        )

def export_default_initial_values(G: AnalysisGraph, variables_file: str):
    s0 = construct_default_initial_state(G)
    s0.to_csv(variables_file, index_label="variable")
