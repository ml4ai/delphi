from typing import Dict, List, Optional, Union, Callable, Tuple
import json
import pickle
from uuid import uuid4
import networkx as nx
import pandas as pd
from scipy.stats import gaussian_kde
from delphi.paths import adjectiveData
from delphi.utils.indra import get_valid_statements_for_modeling, get_concepts
from .utils.web import get_data_from_url
from .assembly import (
    constructConditionalPDF,
    get_respdevs,
    make_edges,
    construct_concept_to_indicator_mapping,
    get_indicators,
)
from delphi.utils.fp import flatMap
from datetime import datetime
from itertools import permutations
from indra.statements import Influence
import numpy as np


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
        edges = make_edges(sts, node_permutations)
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

        return cls.from_statements(
            get_statements_from_json(json_serialized_list)
        )

    @classmethod
    def from_json_serialized_statements_file(cls, file):
        with open(file, "r") as f:
            return cls.from_json_serialized_statements_list(f.read())

    def assemble_transition_model_from_gradable_adjectives(
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

        rs = gaussian_kde(
            flatMap(
                lambda g: gaussian_kde(get_respdevs(g[1]))
                .resample(res)[0]
                .tolist(),
                gb,
            )
        ).resample(res)[0]

        for e in self.edges(data=True):
            e[2]["ConditionalProbability"] = constructConditionalPDF(gb, rs, e)
            e[2]["betas"] = np.tan(
                e[2]["ConditionalProbability"].resample(self.res)[0]
            )

    def map_concepts_to_indicators(
        self, n: int = 1, mapping_file: Optional[str] = None
    ):
        """ Add indicators to the analysis graph.

        Args:
            n
            mapping_file
        """
        if mapping_file is None:
            url = "http://vision.cs.arizona.edu/adarsh/export/demos/data/concept_to_indicator_mapping.txt"
            mapping_file = get_data_from_url(url)

        mapping = construct_concept_to_indicator_mapping(n, mapping_file)

        for n in self.nodes(data=True):
            n[1]["indicators"] = get_indicators(
                n[0].lower().replace(" ", "_"), mapping
            )

        return self
