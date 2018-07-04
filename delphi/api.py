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


def create_qualitative_analysis_graph(sts: List[Influence]) -> AnalysisGraph:
    """ Create a qualitative analysis graph from a list of INDRA Influence
    statements."""
    return make_cag_skeleton(sts)


def get_subgraph_for_concept(concept: str, cag: AnalysisGraph, depth_limit = 2) -> AnalysisGraph:
    """ Get a subgraph of the analysis graph for a single concept """
    pred = nx.dfs_predecessors(cag, concept, depth_limit = depth_limit)
    return cag.subgraph(list(pred.keys())+[concept])


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


def add_transition_model(cag: AnalysisGraph,
                         adjective_data: str = None) -> AnalysisGraph:
    """ Add probability distribution functions constructed from gradable
    adjective data to the edges of the analysis graph data structure.

    Args:
        cag
        adjective_data
    """

    if adjective_data is None:
        adjective_data = adjectiveData

    return add_conditional_probabilities(cag, adjectiveData)


def add_indicators(cag: AnalysisGraph, n: int = 1) -> AnalysisGraph:
    return set_indicators(cag, n = n)


def parameterize(
    cag: AnalysisGraph, time: datetime, df: pd.DataFrame
) -> AnalysisGraph:
    """ Parameterize the model

    Args:
        cag
        time
        df
    """
    return set_indicator_values(set_indicators(cag), time, df)



def load():
    pass


def export(cag: AnalysisGraph, format="pkl", pkl_filename="delphi_model.pkl"):
    if format == "pkl":
        with open(pkl_filename, "wb") as f:
            pickle.dump(cag, f)
    elif format == "cra":
        to_json(cag)

def get_valid_statements(sts):
    return [s for s in sts if is_grounded(s) and (s.subj_delta['polarity'] is
        not None) and (s.obj_delta['polarity'] is not None)]
