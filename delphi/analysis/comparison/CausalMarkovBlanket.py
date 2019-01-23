from typing import List
import networkx as nx
from itertools import product

from networkx.algorithms.simple_paths import all_simple_paths


class CausalMarkovBlanket(nx.DiGraph):
    """
    Stub for class that will create a markov blanket around components of two
    congruent subnetworks that are not shared between the two networks.
    """
    def __init__(self, G: nx.DiGraph, shared: List[str]):
        super().__init__()
        self.orig_graph = G
        self.shared_nodes = shared
        self.outputs = self.get_output_nodes()
        self.paths = self.get_shared_net_paths()
        self.cover_edges = self.get_cover_edges()
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

    def get_output_nodes(self) -> List[str]:
        """ Get all output nodes from a network. """
        return [n for n, d in self.orig_graph.out_degree() if d == 0]

    def get_shared_net_paths(self) -> List:
        """Get all paths from shared inputs to shared outputs."""
        new_inputs = list(set(self.shared_nodes) - set(self.outputs))
        return [pth for (inp, out) in product(new_inputs, self.outputs)
                for pth in all_simple_paths(self.orig_graph, inp, out)]

    def get_cover_edges(self) -> List[List[str]]:
        """Get all edges needed to blanket the included nodes."""
        main_nodes = list(set([node for path in self.paths for node in path]))
        # Need to include children and parents of children in the markov blanket
        predecessors = [[pred, node] for node in main_nodes
                        for pred in self.orig_graph.predecessors(node)
                        if pred not in main_nodes]

        successors = [[node, succ] for node in main_nodes
                      for succ in self.orig_graph.successors(node)
                      if succ not in main_nodes]

        succ_preds = [[pred, node] for node in successors
                      for pred in self.orig_graph.predecessors(node)
                      if pred not in main_nodes]

        return predecessors + successors + succ_preds
