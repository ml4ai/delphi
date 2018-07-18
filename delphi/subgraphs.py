import networkx as nx
from itertools import chain
from .AnalysisGraph import AnalysisGraph

# ==========================================================================
# Subgraphs
# ==========================================================================


def get_subgraph_for_concept(
    G: AnalysisGraph, concept: str, depth_limit: Optional[int] = None
) -> AnalysisGraph:
    """ Returns a subgraph of the analysis graph for a single concept.

    Args:
        G
        concept
        depth_limit
    """
    rev = G.reverse()
    dfs_edges = nx.dfs_edges(rev, concept, depth_limit)

    return AnalysisGraph(G.subgraph(chain.from_iterable(dfs_edges)).copy())


def get_subgraph_for_concept_pair(
    G: AnalysisGraph, source: str, target: str, cutoff: Optional[int] = None
) -> AnalysisGraph:
    """ Get subgraph comprised of simple paths between the source and the
    target.

    Args:
        G
        source
        target
        cutoff
    """
    paths = nx.all_simple_paths(G, source, target, cutoff=cutoff)
    return AnalysisGraph(G.subgraph(set(chain.from_iterable(paths))))


def get_subgraph_for_concept_pairs(
    G: AnalysisGraph, concepts: List[str], cutoff: Optional[int] = None
) -> AnalysisGraph:
    """ Get subgraph comprised of simple paths between the source and the
    target.

    Args:
        G
        concepts
        cutoff
    """
    path_generator = (
        nx.all_simple_paths(G, source, target, cutoff=cutoff)
        for source, target in permutations(concepts, 2)
    )
    paths = chain.from_iterable(path_generator)
    return AnalysisGraph(G.subgraph(set(chain.from_iterable(paths))))
