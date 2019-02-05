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
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from delphi.utils.misc import choose_font


FONT = choose_font()

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
            "fontname": FONT,
            "overlap": "scale",
            "splines": True,
        }
    )

    A.node_attr.update(
        {
            "shape": "rectangle",
            "color": "black",
            # "color": "#650021",
            "style": "rounded",
            "fontname": FONT,
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

    nodeset = {n.split("/")[-1] for n in G.nodes}

    simplified_labels = len(nodeset) == len(G)
    color_str = "#650021"
    for n in G.nodes(data=True):
        if kwargs.get("values"):
            node_label = (
                n[0].capitalize().replace("_", " ")
                + " ("
                + str(np.mean(n[1]["rv"].dataset))
                + ")"
            )
        else:
            node_label = (
                n[0].split("/")[-1].replace("_", " ").capitalize()
                if simplified_labels
                else n[0]
            )
        A.add_node(n[0], label=node_label)

    max_median_betas = max(
        [abs(np.median(e[2]["βs"])) for e in G.edges(data=True)]
    )

    for e in G.edges(data=True):
        # Calculate reinforcement (ad-hoc!)

        sts = e[2]["InfluenceStatements"]
        total_evidence_pieces = sum([len(s.evidence) for s in sts])
        reinforcement = (
            sum([stmt.overall_polarity() * len(stmt.evidence) for stmt in sts])
            / total_evidence_pieces
        )
        opacity = total_evidence_pieces / n_max
        h = (opacity * 255).hex()
        cmap = cm.Greens if reinforcement > 0 else cm.Reds
        c_str = matplotlib.colors.rgb2hex(cmap(abs(reinforcement))) + h[4:6]

        A.add_edge(
            e[0],
            e[1],
            color=c_str,
            penwidth=3 * abs(np.median(e[2]["βs"]) / max_median_betas),
        )

    # Drawing indicator variables

    if kwargs.get("indicators"):
        for n in nodes_with_indicators:
            for indicator_name, ind in n[1]["indicators"].items():
                node_label = _insert_line_breaks(
                    ind.name.replace("_", " "), 30
                )
                if kwargs.get("indicator_values"):
                    if ind.unit is not None:
                        units = f" {ind.unit}"
                    else:
                        units = ""

                    if ind.mean is not None:
                        ind_value = "{:.2f}".format(ind.mean)
                        node_label = (
                            f"{node_label}\n{ind_value} {ind.unit}"
                            f"\nSource: {ind.source}"
                            f"\nAggregation axes: {ind.aggaxes}"
                            f"\nAggregation method: {ind.aggregation_method}"
                        )

                A.add_node(
                    indicator_name,
                    style="rounded, filled",
                    fillcolor="lightblue",
                    label=node_label,
                )
                A.add_edge(n[0], indicator_name, color="royalblue4")

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
