from typing import List, Dict
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


def get_input_nodes(network: nx.DiGraph) -> List[str]:
    return [node for node, degree in network.in_degree() if degree == 0]


def get_output_nodes(network: nx.DiGraph) -> List[str]:
    return [node for node, degree in network.out_degree() if degree == 0]


def get_io_paths(network: nx.DiGraph, input_nodes: List[str], output_nodes: List[str]) -> Dict:
    results = dict()
    for in_node in input_nodes:
        for out_node in output_nodes:
            short_simple_path = nx.algorithms.simple_paths.all_simple_paths(network, in_node, out_node)
            results[(in_node, out_node)] = list(short_simple_path)
    return results


if __name__ == "__main__":
    pa_graph_example_dir = data_dir/"program_analysis"/"pa_graph_examples"
    asce = nx_graph_from_dotfile(str(pa_graph_example_dir/"asce-graph.dot"))
    priestley_taylor = nx_graph_from_dotfile(str(pa_graph_example_dir/"priestley-taylor-graph.dot"))

    draw_graph(asce, "asce-graph.pdf")
    draw_graph(priestley_taylor, "priestley-taylor.pdf")

    asce_inputs = get_input_nodes(asce)
    asce_outputs = get_output_nodes(asce)
    asce_paths = get_io_paths(asce, asce_inputs, asce_outputs)
    print(asce_paths)

    pt = priestley_taylor
    pt_inputs = get_input_nodes(pt)
    pt_outputs = get_output_nodes(pt)
    pt_paths = get_io_paths(pt, pt_inputs, pt_outputs)
    print(pt_paths)

    found_in_asce = [key in asce_paths for key in pt_paths.keys()]
    print(found_in_asce)

    found_in_pt = [key in pt_paths for key in asce_paths.keys()]
    print(found_in_pt)
