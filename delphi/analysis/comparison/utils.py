from networkx.drawing.nx_agraph import read_dot, to_agraph
from typing import List
import networkx as nx


def draw_graph(G: nx.DiGraph, filename: str):
    """ Draw a networkx graph with Pygraphviz. """
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "LR"
    A.draw(filename, prog="dot")


def nx_graph_from_dotfile(filename: str) -> nx.DiGraph:
    """ Get a networkx graph from a DOT file, and reverse the edges. """
    return nx.DiGraph(read_dot(filename).reverse())


def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)


def get_shared_nodes(G1: nx.DiGraph, G2: nx.DiGraph) -> List[str]:
    """Get all the nodes that are common to both networks."""
    return list(set(G1.nodes()).intersection(set(G2.nodes())))
