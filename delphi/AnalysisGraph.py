from typing import Dict, List, Optional, Union, Callable, Tuple
import random
import json
from uuid import uuid4
import pickle
from dataclasses import dataclass
import networkx as nx
import pandas as pd
import numpy as np
from .assembly import get_respdevs
from future.utils import lmap, lzip
from delphi.assembly import (
    constructConditionalPDF,
    get_concepts,
    get_valid_statements_for_modeling,
    nameTuple,
)
from delphi.utils.fp import flatMap, iterate, take, ltake,  compose
from delphi.utils.misc import _insert_line_breaks
from .paths import adjectiveData
from datetime import datetime
from scipy.stats import gaussian_kde
from itertools import chain, permutations, cycle
from indra.statements import Influence


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
        self.id = str(uuid4())
        self.t: float = 0.0
        self.Δt: float = 1.0
        self.time_unit: str = "Placeholder time unit"
        self.dateCreated = datetime.now()
        self.data = None
        self.name: str = "Linear Dynamical System with Stochastic Transition Model"

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
        G = cls(edges)

        for n in G.nodes(data=True):
            n[1]["id"] = str(uuid4())

        return G

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


    @classmethod
    def from_json_serialized_statements_list(cls, json_serialized_list):
        from delphi.utils.indra import get_statements_from_json
        return cls.from_statements(get_statements_from_json(json_serialized_list))


    @classmethod
    def from_json_serialized_statements_file(cls, file):
        with open(file, "r") as f:
            return cls.from_json_serialized_statements_list(f.read())


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
                    .resample(res)[0]
                    .tolist(),
                    gb,
                )
            )
            .resample(res)[0]
        )

        for e in self.edges(data=True):
            e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)
            e[2]["betas"] = np.tan(
                e[2]["ConditionalProbability"].resample(self.res)[0]
            )
