import json
import importlib
import networkx as nx
from typing import Dict
from pathlib import Path
from inspect import signature
from pprint import pprint
from functools import partial
from pygraphviz import AGraph
from IPython.core.display import Image
from delphi.program_analysis.scopes import Scope


class ProgramAnalysisGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_action_node(self, A: AGraph, lambdas, n):
        """ Add an action node to the CAG. """
        output, = A.successors(n)

        # Only allow LoopVariable and FuncVariable nodes in the DBN
        if output.attr["node_type"] in (
            "LoopVariableNode",
            "FuncVariableNode",
        ):
            oname = output.attr["cag_label"]
            onode = self.nodes[oname]

            # Check if it is an initialization function
            if len(A.predecessors(n)) == 0:
                onode["init_fn"] = getattr(lambdas, n.attr["lambda_fn"])

            # Otherwise append the predecessor function list
            elif n.attr["label"] == "__decision__":
                preds = A.predecessors(n)
                if_var, = [
                    n
                    for n in preds
                    if list(A.predecessors(n))[0].attr["label"]
                    == "__condition__"
                ]
                condition_fn, = A.predecessors(if_var)
                condition_fn = condition_fn[: condition_fn.rfind("__")]
                condition_lambda = condition_fn.replace("condition", "lambda")
                onode["condition_fn"] = getattr(lambdas, condition_lambda)
            else:
                onode["pred_fns"].append(getattr(lambdas, n.attr["lambda_fn"]))

            # If the type of the function is assign, then add an edge in the CAG
            if n.attr["label"] == "__assign__":
                for i in A.predecessors(n):
                    iname = i.attr["cag_label"]
                    self.add_edge(iname, oname)

    def add_variable_node(self, n):
        """ Add a variable node to the CAG. """
        name = n.attr["cag_label"]
        self.add_node(
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
            self.nodes[name]["is_index"] = True
            self.nodes[name]["value"] = int(n.attr["start"])
            self.nodes[name]["visited"] = True
            self.nodes[name]["update_fn"] = (
                lambda **kwargs: int(kwargs.pop(list(kwargs.keys())[0])) + 1
            )
            self.add_edge(name, name)

    @classmethod
    def from_agraph(cls, A: AGraph, lambdas):
        """ Construct a ProgramAnalysisGraph from an AGraph """
        self = cls(nx.DiGraph())

        for n in A.nodes():
            if n.attr["node_type"] in ("LoopVariableNode", "FuncVariableNode"):
                self.add_variable_node(n)

        for n in A.nodes():
            if n.attr["node_type"] == "ActionNode":
                self.add_action_node(A, lambdas, n)

        for n in self.nodes(data=True):
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
            for n in self.nodes()
            if len(list(self.predecessors(n)))
            == len(list(self.successors(n)))
            == 0
        ]

        for n in isolated_nodes:
            self.remove_node(n)

        # Create lists of all input function and all input variables
        self.input_variables = list()
        self.input_functions = list()
        for n in self.nodes():
            if self.nodes[n].get("init_fn") is not None:
                self.input_functions.append(n)

            if (
                self.nodes[n]["node_type"]
                in ("LoopVariableNode", "FuncVariableNode")
                and len(list(self.predecessors(n))) == 0
            ):
                self.input_variables.append(n)

        return self

    @classmethod
    def from_fortran_file(cls, fortran_file):
        A = Scope.from_fortran_file(fortran_file).to_agraph()
        lambdas = importlib.__import__(stem + "_lambdas")
        return cls.from_agraph(A, lambdas)

    def _update_node(self, n: str):
        """ Update the value of node n, recursively visiting its ancestors. """
        node = self.nodes[n]
        if node.get("update_fn") is not None and not node["visited"]:
            node["visited"] = True
            for p in self.predecessors(n):
                self._update_node(p)
            ivals = {i: self.nodes[i]["value"] for i in self.predecessors(n)}
            node["value"] = node["update_fn"](**ivals)

    # ==========================================================================
    # Basic Modeling Interface (BMI)
    # ==========================================================================

    def initialize(self):
        """ Initialize the value of input function nodes in the CAG."""

        for n in self.nodes():
            # if self.nodes[n].get("init_fn") is not None:
            if n in self.input_functions:
                self.nodes[n]["value"] = self.nodes[n]["init_fn"]()
        self.update()

    def update(self):
        for n in self.nodes():
            self._update_node(n)

        for n in self.nodes(data=True):
            n[1]["visited"] = False

    def call(self, inputs):
        pass
