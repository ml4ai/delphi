import networkx as nx
from networkx.drawing.nx_agraph import read_dot, to_agraph
from delphi.paths import data_dir


def nx_graph_from_dotfile(filename: str) -> nx.DiGraph:
    """ Get a networkx graph from a DOT file, and reverse the edges. """
    return nx.DiGraph(read_dot(filename).reverse())


def to_dotfile(G: nx.DiGraph, filename: str):
    """ Output a networkx graph to a DOT file. """
    A = to_agraph(G)
    A.write(filename)


def draw_graph(G: nx.DiGraph, filename: str):
    """ Draw a networkx graph with Pygraphviz. """
    A = to_agraph(G)
    A.graph_attr["rankdir"] = "LR"
    A.draw(filename, prog="dot")


def get_input_nodes(network):
    return [node for node, degree in network.in_degree() if degree == 0]


def get_output_nodes(network):
    return [node for node, degree in network.out_degree() if degree == 0]


def get_io_paths(network, input_nodes, output_nodes):
    results = dict()
    for in_node in input_nodes:
        for out_node in output_nodes:
            short_simple_path = nx.algorithms.simple_paths.shortest_simple_paths(network, in_node, out_node)
            results[(in_node, out_node)] = list(short_simple_path)
    return results


if __name__ == "__main__":
    asce = nx_graph_from_dotfile(str(data_dir/"program_analysis"/"pa_graph_examples"/"asce-graph.dot"))
    priestley_taylor = nx_graph_from_dotfile(str(data_dir/"program_analysis"/"pa_graph_examples"/"priestley-taylor-graph.dot"))
    draw_graph(asce, "asce-graph.pdf")
    draw_graph(priestley_taylor, "priestley-taylor.pdf")

    asce_inputs = get_input_nodes(asce)
    asce_outputs = get_output_nodes(asce)
    asce_paths = get_io_paths(asce, asce_inputs, asce_outputs)
    print(asce_paths)
