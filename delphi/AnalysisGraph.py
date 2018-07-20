from typing import Dict, List, Optional, Union, Callable, Tuple
import random
import json
import pickle
from dataclasses import dataclass
import networkx as nx
import pandas as pd
import numpy as np

from .assembly import (
    get_data,
    get_respdevs,
    construct_concept_to_indicator_mapping,
    get_indicators,
    get_indicator_value,
)

from future.utils import lmap, lzip
from delphi.assembly import (
    constructConditionalPDF,
    get_concepts,
    get_valid_statements_for_modeling,
    nameTuple,
)

from .jupyter_tools import (
    print_full_edge_provenance,
    create_statement_inspection_table,
)
from .utils import flatMap, iterate, take, ltake, _insert_line_breaks, compose
from .types import RV, LatentVar, Indicator
from .paths import adjectiveData, south_sudan_data
from datetime import datetime
from scipy.stats import gaussian_kde
from itertools import chain, permutations, cycle
from indra.statements import Influence
from tqdm import tqdm, trange
from IPython.display import set_matplotlib_formats
from functools import partial
from pygraphviz import AGraph


def make_edge(
    sts: List[Influence], p: Tuple[str, str]
) -> Tuple[str, str, Dict[str, List[Influence]]]:
    edge = (*p, {"InfluenceStatements": [s for s in sts if nameTuple(s) == p]})
    return edge


class AnalysisGraph(nx.DiGraph):
    """ The primary data structure for Delphi """

    def __init__(self, *args, **kwargs):
        """ Default constructor, accepts a list of edge tuples. """
        super().__init__(*args, **kwargs)
        self.t: float = 0.0
        self.Δt: float = 1.0
        self.time_unit: str = "Placeholder time unit"
        self.data = None
        self.name: str = "Linear Dynamical System with Stochastic Transition Model"

    # ==========================================================================
    # Constructors
    # ==========================================================================

    @classmethod
    def from_statements(cls, sts: List[Influence]):
        """ Construct an AnalysisGraph object from a list of INDRA statements. """
        sts = get_valid_statements_for_modeling(sts)
        node_permutations = permutations(get_concepts(sts), 2)
        edges = [
            e
            for e in [make_edge(sts, p) for p in node_permutations]
            if len(e[2]["InfluenceStatements"]) != 0
        ]

        return cls(edges)

    @staticmethod
    def from_pickle(file: str):
        """ Load an AnalysisGraph object from a pickle file. """
        with open(file, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, cls):
            raise TypeError(
                f"The pickled object in {file} is not an instance of AnalysisGraph"
            )
        else:
            return G

    def infer_transition_model(
        self, adjective_data: str = None, res: int = 100
    ):
        """ Add probability distribution functions constructed from gradable
        adjective data to the edges of the analysis graph data structure.

        Args:
            adjective_data
            res
        """

        self.res = res
        if adjective_data is None:
            adjective_data = adjectiveData

        gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby(
            "adjective"
        )

        rs = (
            gaussian_kde(
                flatMap(
                    lambda g: gaussian_kde(get_respdevs(g[1]))
                    .resample(20)[0]
                    .tolist(),
                    gb,
                )
            )
            .resample(100)[0]
            .tolist()
        )

        for e in self.edges(data=True):
            e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)
            e[2]["betas"] = np.tan(
                e[2]["ConditionalProbability"].resample(self.res)[0]
            )

    def _repr_png_(self, *args, **kwargs):
        return self.to_agraph(self, *args, **kwargs).draw(
            format="png", prog=kwargs.get("prog", "dot")
        )

    def to_agraph(self, *args, **kwargs) -> AGraph:
        """ Exports AnalysisGraph to pygraphviz AGraph

        Args:
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
            n for n in G.nodes(data=True) if n[1].get("indicators") is not None
        ]

        n_max= max([len(e[2]["InfluenceStatements"]) for e in G.edges(data=True)])

        color_str = "#650021"
        for e in G.edges(data=True):
            opacity = len(e[2]["InfluenceStatements"]) / n_max
            h = (opacity * 255).hex()
            c_str = color_str + h[4:6]
            A.add_edge(
                e[0].capitalize(), e[1].capitalize(), color=c_str, arrowsize=0.5
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
                    A.add_edge(
                        n[0].capitalize(), node_label, color="royalblue4"
                    )

        # Drawing indicator values
        if kwargs.get("indicator_values"):
            for n in nodes_with_indicators:
                indicators = [
                    i for i in n[1]["indicators"] if i.value is not None
                ]
                for ind in indicators:
                    if ind.unit is not None:
                        units = f" {ind.unit}"
                    else:
                        units = ""
                    ind_label = _insert_line_breaks(ind.name)

                    node_label = "{:.2f}".format(ind.value) + units
                    A.add_node(
                        node_label,
                        shape="plain",
                        # style="rounded, filled",
                        fillcolor="white",
                        color="royalblue",
                    )
                    A.add_edge(
                        ind_label,
                        node_label,
                        color="lightblue",
                        arrowhead="none",
                    )

        if kwargs.get("nodes_to_highlight") is not None:
            for n in kwargs["nodes_to_highlight"]:
                A.add_node(n.capitalize(), fontcolor="royalblue")

        if kwargs.get("graph_label") is not None:
            A.graph_attr["label"] = kwargs["graph_label"]

        return A
