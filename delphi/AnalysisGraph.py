from typing import Dict, List, Optional, Union, Callable, Tuple
import random
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import pandas as pd
import numpy as np
from uuid import uuid4

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
from .random_variables import RV, LatentVar, Indicator
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
        self.dateCreated = datetime.now()
        self.data = None
        self.name: str = "Linear Dynamical System with Stochastic Transition Model"
        self.id = uuid4()

    # ==========================================================================
    # Constructors
    # ==========================================================================

    @classmethod
    def from_statements_file(cls, file: str):
        """ Construct an AnalysisGraph object from a pickle file containing a
        list of INDRA statements. """

        with open(file, "rb") as f:
            sts = pickle.load(f)

        return cls.from_statements(sts)

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

    @classmethod
    def from_pickle(cls, file: str):
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

        gb = pd.read_csv(adjective_data, delim_whitespace=True).groupby(
            "adjective"
        )

        rs = (
            gaussian_kde(
                flatMap(
                    lambda g: gaussian_kde(get_respdevs(g[1]))
                    .resample(100)[0]
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
