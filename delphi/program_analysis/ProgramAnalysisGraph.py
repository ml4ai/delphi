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
    name = n.attr["cag_label"]
    G.add_node(
        name,
        value=None,
        pred_fns=[],
        agraph_name=n,
        index=n.attr["index"],
        node_type=n.attr["node_type"],
        start=n.attr["start"],
        end=n.attr["end"],
        index_var=n.attr["index_var"],
        visited=False,
    )

    # If the node is a loop index, set special initialization
    # and update functions.
    if n.attr["is_index"] == "True":
        G.nodes[name]["is_index"] = True
        G.nodes[name]["value"] = int(n.attr["start"])
        G.nodes[name]["visited"] = True
        G.nodes[name]["update_fn"] = (
            lambda **kwargs: int(kwargs.pop(list(kwargs.keys())[0])) + 1
        )
        G.add_edge(name, name)


def add_action_node(A: AGraph, G: nx.DiGraph, λs, n):
    """ Add an action node to the CAG. """
    output, = A.successors(n)

    # Only allow LoopVariableNodes in the DBN
    if output.attr["node_type"] == "LoopVariableNode":
        oname = output.attr["cag_label"]
        onode = G.nodes[oname]

        # Check if it is an initialization function
        if len(A.predecessors(n)) == 0:
            onode["init_fn"] = getattr(λs, n.attr["lambda_fn"])

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
            condition_λ = condition_fn.replace("condition", "lambda")
            onode["condition_fn"] = getattr(λs, condition_λ)
        else:
            onode["pred_fns"].append(getattr(λs, n.attr["lambda_fn"]))

        # If the type of the function is assign, then add an edge in the CAG
        if n.attr["label"] == "__assign__":
            for i in A.predecessors(n):
                iname = i.attr["cag_label"]
                G.add_edge(iname, oname)


class ProgramAnalysisGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop_index, = [
            n[0] for n in self.nodes(data=True) if n[1].get("is_index")
        ]

    @classmethod
    def from_agraph(cls, A: AGraph, λs):
        """ Construct a ProgramAnalysisGraph from an AGraph """
        G = nx.DiGraph()

        for n in A.nodes():
            if n.attr["node_type"] == "LoopVariableNode":
                add_variable_node(G, n)

        for n in A.nodes():
            if n.attr["node_type"] == "ActionNode":
                add_action_node(A, G, λs, n)

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
                    ind = 0 if cond_fn(**kwargs) else 1
                    return n[1]["choice_fns"][ind](**kwargs)

                n[1]["update_fn"] = partial(update_fn, n)

        isolated_nodes = [
            n
            for n in G.nodes()
            if len(list(G.predecessors(n))) == len(list(G.successors(n))) == 0
        ]

        for n in isolated_nodes:
            G.remove_node(n)

        return cls(G)
