import numpy as np
from typing import Dict
import networkx as nx
from .utils import _insert_line_breaks
from pygraphviz import AGraph

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

def to_agraph(G, *args, **kwargs):

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
