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
    arr = kde.dataset
    X = np.linspace(min(arr), max(arr), res)
    Y = np.array(kde.pdf(X)) * (X[1] - X[0])
    return {"theta": X, "P(theta)": Y.tolist()}


def _get_polynomial_fit(e, deg=7, res=100):
    kde = e[2]["ConditionalProbability"]
    arr = kde.dataset
    X = np.linspace(min(arr), max(arr), res)
    Y = np.array(kde.pdf(X)) * (X[1] - X[0])
    coefs = np.polynomial.polynomial.polyfit(X, Y, deg=deg)
    return {"degree": deg, "coefficients": list(coefs)}
