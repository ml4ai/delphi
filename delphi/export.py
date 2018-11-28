import json
from typing import Dict, Union, List
import numpy as np
import networkx as nx
from .utils.misc import _insert_line_breaks
from .utils.fp import lmap
from functools import partial
from networkx import DiGraph
from pygraphviz import AGraph
import pickle
from datetime import datetime
import platform
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize

operating_system = platform.system()

if operating_system == "Darwin":
    font = "Gill Sans"
elif operating_system == "Windows":
    font = "Candara"
else:
    font = "Ubuntu"

# ==========================================================================
# Export
# ==========================================================================


def to_agraph(G, *args, **kwargs) -> AGraph:
    """ Exports AnalysisGraph to pygraphviz AGraph

    Args:
        G
        args
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
            "fontname": font,
            "overlap": "scale",
            "splines": True,
        }
    )

    A.node_attr.update(
        {
            "shape": "rectangle",
            "color": "#650021",
            "style": "rounded",
            "fontname": font,
        }
    )

    nodes_with_indicators = [
        n for n in G.nodes(data=True) if n[1].get("indicators") is not None
    ]

    n_max = max(
        [
            sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
            for e in G.edges(data=True)
        ]
    )

    color_str = "#650021"
    for n in G.nodes():
        A.add_node(n, label=n.capitalize().replace("_", " "))

    for e in G.edges(data=True):
        reinforcement = np.mean(
            [
                stmt.subj_delta["polarity"] * stmt.obj_delta["polarity"]
                for stmt in e[2]["InfluenceStatements"]
            ]
        )
        opacity = (
            sum([len(s.evidence) for s in e[2]["InfluenceStatements"]]) / n_max
        )
        h = (opacity * 255).hex()
        cmap = cm.Greens if reinforcement > 0 else cm.Reds
        c_str = matplotlib.colors.rgb2hex(cmap(abs(reinforcement))) + h[4:6]
        A.add_edge(e[0], e[1], color=c_str, arrowsize=0.5)

    # Drawing indicator variables

    if kwargs.get("indicators"):
        for n in nodes_with_indicators:
            for indicator_name, ind in n[1]["indicators"].items():
                node_label = _insert_line_breaks(ind.name.replace("_", " "))
                if kwargs.get("indicator_values"):
                    if ind.unit is not None:
                        units = f" {ind.unit}"
                    else:
                        units = ""

                    if ind.mean is not None:
                        ind_value = "{:.2f}".format(ind.mean) + f"{units}"
                        node_label = f"{node_label}\n[{ind_value}]"

                A.add_node(
                    node_label, style="rounded, filled", fillcolor="lightblue"
                )
                A.add_edge(n[0], node_label, color="royalblue4")

    if kwargs.get("nodes_to_highlight") is not None:
        nodes = kwargs.pop("nodes_to_highlight")
        if isinstance(nodes, list):
            for n in nodes:
                if n in A.nodes():
                    A.add_node(n, fontcolor="royalblue")
        elif isinstance(nodes, str):
            if n in A.nodes():
                A.add_node(nodes, fontcolor="royalblue")

    if kwargs.get("graph_label") is not None:
        A.graph_attr["label"] = kwargs["graph_label"]

    return A


def _process_datetime(indicator_dict: Dict):
    time = indicator_dict.get("time")
    indicator_dict["time"] = str(time)
    return indicator_dict


def export_edge(e):
    return {
        "source": e[0],
        "target": e[1],
        "CPT": _construct_CPT(e),
        "polyfit": _get_polynomial_fit(e),
        "InfluenceStatements": [
            s.to_json() for s in e[2]["InfluenceStatements"]
        ],
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
