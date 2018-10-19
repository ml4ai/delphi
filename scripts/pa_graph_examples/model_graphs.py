import networkx as nx
from networkx.drawing.nx_agraph import read_dot, to_agraph


def nx_graph_from_dotfile(filename: str) -> nx.DiGraph:
    """ Get a networkx graph from a DOT file, and reverse the edges. """
    return read_dot(filename).reverse()


def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)


def to_png(G: nx.DiGraph, filename: str):
    """ Draw a networkx graph with Pygraphviz. """
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "LR"
    A.draw(filename, prog="dot")


if __name__ == "__main__":
    asce = nx_graph_from_dotfile("asce-graph.dot")
    priestley_taylor = nx_graph_from_dotfile("priestley-taylor-graph.dot")
    to_png(asce, "asce-graph.png")
    to_png(priestley_taylor, "priestley-taylor.png")
