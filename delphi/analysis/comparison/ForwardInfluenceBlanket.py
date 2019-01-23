from typing import List
import networkx as nx
from itertools import product

from networkx.algorithms.simple_paths import all_simple_paths

import delphi.analysis.comparison.utils as utils


class ForwardInfluenceBlanket(nx.DiGraph):
    """
    This class takes a network and a list of a shared nodes between the input
    network and a secondary network. From this list a shared nodes and blanket
    network is created including all of the nodes between any input/output pair
    in the shared nodes, as well as all nodes required to blanket the network
    for forward influence. This class itself becomes the blanket and inherits
    from the NetworkX DiGraph class.
    """
    def __init__(self, G: nx.DiGraph, shared: List[str]):
        super().__init__()
        self.orig_graph = G
        self.shared_nodes = shared
        self.outputs = utils.get_output_nodes(self.orig_graph)

        # Get all paths from shared inputs to shared outputs
        new_inputs = list(set(self.shared_nodes) - set(self.outputs))
        self.paths = [pth for (inp, out) in product(new_inputs, self.outputs)
                      for pth in all_simple_paths(self.orig_graph, inp, out)]

        # Get all edges needed to blanket the included nodes
        main_nodes = list(set([node for path in self.paths for node in path]))
        self.cover_edges = [[pred, node] for node in main_nodes
                            for pred in self.orig_graph.predecessors(node)
                            if pred not in main_nodes]

        # Need to include children and parents of children for markov blanket
        # successors = [[node, succ] for node in main_nodes
        #               for succ in self.orig_graph.successors(node)
        #               if succ not in main_nodes]
        # succ_preds = [[pred, node] for node in successors
        #               for pred in self.orig_graph.predecessors(node)
        #               if pred not in main_nodes]
        # self.cover_edges.extend(successors)
        # self.cover_edges.extend(succ_preds)

        self.cover_nodes = [node for node, _ in self.cover_edges]

        for path in self.paths:
            self.add_edges_from(list(zip(path, path[1:])))
        self.add_edges_from(self.cover_edges)

        for node_name in self.cover_nodes:
            self.node[node_name]["color"] = "green"

        for node_name in self.shared_nodes:
            self.node[node_name]["color"] = "blue"
            for dest in self.successors(node_name):
                self[node_name][dest]["color"] = "blue"

        for source, dest in self.cover_edges:
            self[source][dest]["color"] = "green"
