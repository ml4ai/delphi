import networkx as nx
from typing import Dict
import json
from pprint import pprint
from pygraphviz import AGraph
from IPython.core.display import Image
from functools import partial
from inspect import signature


def add_variable_node(G, n):
    """ Add a variable node to the CAG. """
    name = n.attr["label"]
    G.add_node(name, value=None, pred_fns=[], agraph_name=n)

    # If the node is a loop index, set special initialization
    # and update functions.
    if n.attr["is_index"] == "True":
        G.nodes[name]["init_fn"] = lambda: 1
        G.nodes[name]["update_fn"] = (
            lambda **kwargs: int(kwargs.pop(list(kwargs.keys())[0])) + 1
        )
        G.add_edge(name, name)


def add_action_node(A: Agraph, G: nx.DiGraph, lambdas, n):
    """ Add an action node to the CAG. """
    output, = A.successors(n)
    oname = output.attr["label"]

    # Check if it is an initialization function
    if len(A.predecessors(n)) == 0:
        G.nodes[oname]["init_fn"] = getattr(lambdas, n.attr["lambda_fn"])

    # Otherwise append the predecessor function list
    elif n.attr["label"] == "__decision__":
        preds = A.predecessors(n)
        if_var, = [
            n
            for n in preds
            if list(A.predecessors(n))[0].attr["label"] == "__condition__"
        ]
        condition_fn, = A.predecessors(if_var)
        condition_fn = condition_fn[: condition_fn.rfind("__")]
        condition_lambda = condition_fn.replace("condition", "lambda")
        G.nodes[oname]["condition_fn"] = getattr(lambdas, condition_lambda)
    else:
        G.nodes[oname]["pred_fns"].append(
            getattr(lambdas, n.attr["lambda_fn"])
        )

    # If the type of the function is assign, then add an edge in the CAG
    if n.attr["label"] == "__assign__":
        for i in A.predecessors(n):
            iname = i.attr["label"]
            G.add_edge(iname, oname)


class ProgramAnalysisGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_agraph(cls, A: AGraph, lambdas):
        """ Construct a ProgramAnalysisGraph from an AGraph """
        G = nx.DiGraph()

        for n in A.nodes():
            if n.attr["node_type"] != "ActionNode":
                add_variable_node(G, n)

        for n in A.nodes():
            if n.attr["node_type"] == "ActionNode":
                add_action_node(A, G, lambdas, n)

        for n in G.nodes(data=True):
            n_preds = len(n[1]["pred_fns"])
            if n_preds == 0:
                del n[1]["pred_fns"]
            elif n_preds == 1:
                n[1]["update_fn"], = n[1].pop("pred_fns")
            else:
                n[1]["choice_fns"] = n[1].pop("pred_fns")

                def update_fn(n, **kwargs):
                    cond_fn = n[1]["condition_fn"]
                    sig = signature(cond_fn)
                    if cond_fn(**kwargs):
                        return n[1]["choice_fns"][0](**kwargs)
                    else:
                        return n[1]["choice_fns"][1](**kwargs)

                n[1]["update_fn"] = partial(update_fn, n)

        isolated_nodes = [
            n
            for n in G.nodes()
            if len(list(G.predecessors(n))) == len(list(G.successors(n))) == 0
        ]

        for n in isolated_nodes:
            G.remove_node(n)

        return cls(G)

    def visualize(self, show_values=False):
        """ Exports AnalysisGraph to pygraphviz AGraph

        Args:
            args
            kwargs

        Returns:
            AGraph
        """

        A = AGraph(directed=True)
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
        A.node_attr.update(
            {
                "shape": "rectangle",
                "color": "#650021",
                "style": "rounded",
                "fontname": "Gill Sans",
            }
        )

        color_str = "#650021"

        for n in self.nodes():
            A.add_node(n, label=n)

        for e in self.edges(data=True):
            A.add_edge(e[0], e[1], color=color_str, arrowsize=0.5)

        if show_values:
            for n in A.nodes():
                value = str(self.nodes[n]["value"])
                n.attr["label"] = n.attr["label"] + f": {value:.4}"

        # Drawing indicator variables

        return Image(A.draw(format="png", prog="dot"), retina=True)
