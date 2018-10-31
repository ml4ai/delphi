from typing import Dict, List, Optional, Union, Callable, Tuple
import json
import pickle
from uuid import uuid4
import networkx as nx
from delphi.assembly import (
    get_concepts,
    get_valid_statements_for_modeling,
    nameTuple,
)
from datetime import datetime
from itertools import permutations
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
