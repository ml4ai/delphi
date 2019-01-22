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
    """ Get all input nodes from a network. """
    return [node for node, degree in network.in_degree() if degree == 0]


def get_output_nodes(network: nx.DiGraph) -> List[str]:
    """ Get all output nodes from a network. """
    return [node for node, degree in network.out_degree() if degree == 0]


def get_io_paths(network: nx.DiGraph, input_nodes: List[str], output_nodes: List[str]) -> dict:
    """ Returns a dict of all paths for each input/output node pair. """
    results = dict()
    for in_node in input_nodes:
        for out_node in output_nodes:
            short_simple_path = nx.algorithms.simple_paths.all_simple_paths(network, in_node, out_node)
            results[(in_node, out_node)] = list(short_simple_path)
    return results


def io_paths_in_both(p_dict1: dict, p_dict2: dict) -> List[str]:
    return list(set(asce_paths.keys()).intersection(set(pt_paths.keys())))


def get_simple_networks(network1: nx.DiGraph, network2: nx.DiGraph,
                        n1_paths: dict, n2_paths: dict, path_set: list):
    """
    Returns simplified versions of networks 1 and 2 that have equivalent node sets
    """
    def find_same_node_paths(path_list, other_net):
        return [[n for n in path if n in other_net] for path in path_list]

    def digraph_from_paths(path_list):
        G = nx.DiGraph()
        for path in path_list:
            G.add_edges_from(list(zip(path, path[1:])))
        return G

    n1_same_nodes_paths = list()
    n2_same_nodes_paths = list()
    for key in path_set:
        (in_node, out_node) = key
        n1_paths_list = n1_paths[key]
        n2_paths_list = n2_paths[key]

        n1_same_nodes_paths.extend(find_same_node_paths(n1_paths_list, network2))
        n2_same_nodes_paths.extend(find_same_node_paths(n2_paths_list, network1))

    simple_n1 = digraph_from_paths(n1_same_nodes_paths)
    simple_n2 = digraph_from_paths(n2_same_nodes_paths)
    return simple_n1, simple_n2


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

    found_in_both = io_paths_in_both(asce_paths, pt_paths)
    simple_asce, simple_pt = get_simple_networks(asce, pt, asce_paths, pt_paths, found_in_both)
    draw_graph(simple_asce, "simple-asce.pdf")
    draw_graph(simple_pt, "simple-pt.pdf")
