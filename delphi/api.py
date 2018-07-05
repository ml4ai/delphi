# -*- coding: utf-8 -*-

""" This module defines the public API for Delphi. """

from indra.statements import Influence
from typing import *
from datetime import datetime
import pandas as pd
from .types import AnalysisGraph
import delphi
from .assembly import (
    is_grounded,
    add_conditional_probabilities,
    make_cag_skeleton,
    set_indicators,
    set_indicator_values,
    top_grounding_score
)
from .paths import adjectiveData
from .viz import to_agraph
from .export import to_json
import networkx as nx
from itertools import chain, permutations
import pickle


def create_qualitative_analysis_graph(sts: List[Influence]) -> AnalysisGraph:
    """ Create a qualitative analysis graph from a list of INDRA Influence
    statements.

    Args:
        sts
    """
    return make_cag_skeleton(sts)


def get_subgraph_for_concept(concept: str, cag: AnalysisGraph,
                             depth_limit = None) -> AnalysisGraph:
    """ Get a subgraph of the analysis graph for a single concept.

    Args:
        concept
        cag
        depth_limit
    """
    rev = cag.reverse()
    dfs_edges = nx.dfs_edges(rev, concept, depth_limit = depth_limit)
    return cag.subgraph(chain.from_iterable(dfs_edges))


def get_subgraph_for_concept_pair(source: str, target: str,
                                  cag: AnalysisGraph,
                                  cutoff: Optional[int] = None) -> AnalysisGraph:
    """ Get subgraph comprised of simple paths between the source and the
    target.

    Args:
        source
        target
        cag
        cutoff

    """
    paths = nx.all_simple_paths(cag, source, target, cutoff = cutoff)
    return cag.subgraph(set(chain.from_iterable(paths)))


def get_subgraph_for_concept_pairs(concepts: List[str],
                                   cag: AnalysisGraph,
                                   cutoff: Optional[int] = None) -> AnalysisGraph:
    """ Get subgraph comprised of simple paths between the source and the
    target.

    Args:
        concepts
        cag
        cutoff
    """
    path_generator = (nx.all_simple_paths(cag, source, target, cutoff = cutoff)
                      for source, target in permutations(concepts, 2))
    paths = chain.from_iterable(path_generator)
    return cag.subgraph(set(chain.from_iterable(paths)))


def add_transition_model(adjective_data, cag: AnalysisGraph,) -> AnalysisGraph:
    """ Add probability distribution functions constructed from gradable
    adjective data to the edges of the analysis graph data structure.

    Args:
        adjective_data
        cag
    """

    return add_conditional_probabilities(cag, adjectiveData)


def add_indicators(n, cag: AnalysisGraph) -> AnalysisGraph:
    """ Add indicators to the analysis graph.

    Args:
        n
        cag
    """
    return set_indicators(cag, n = n)


def parameterize(
    time: datetime, df: pd.DataFrame, cag: AnalysisGraph,
) -> AnalysisGraph:
    """ Parameterize the analysis graph.

    Args:
        time
        df
        cag
    """
    return set_indicator_values(cag, time, df)



def load(filename: str) -> AnalysisGraph:
    """ Load a pickled AnalysisGraph object from a given file. """

    with open(filename, "rb") as f:
        cag = pickle.load(f)

    return cag


def get_valid_statements_for_modeling(sts):
    """ Select INDRA statements that can be used to construct a Delphi model
    from a given list of statements. """

    return [s for s in sts if is_grounded(s) and (s.subj_delta['polarity'] is
        not None) and (s.obj_delta['polarity'] is not None)]
