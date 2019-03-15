import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Dict
from enum import Enum
import importlib
import inspect
import time
import json
import os
import ast

import networkx as nx

from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)

import subprocess as sp


def timeit(method):
    """Timing wrapper for exectuion comparison."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args)
        te = time.time()
        print(f"{method.__name__}:\t{((te - ts) * 1000):2.4f}ms")
        return result

    return timed


class GroundedFunctionNetwork(nx.DiGraph):
    """
    Stub for the Grounded Function Network class
    """

    def __init__(self, nodes, edges, subgraphs):
        super(GroundedFunctionNetwork, self).__init__()
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        self.scopes = subgraphs

        self.inputs = [n for n, d in self.in_degree() if d == 0]
        self.outputs = [n for n, d in self.out_degree() if d == 0]
        self.output_node = self.outputs[-1]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.traverse_nodes(self.inputs)

    def traverse_nodes(self, node_set, depth=0):
        """BFS traversal of nodes that returns name traversal as large string."""
        tab = "  "
        result = ""
        for n in node_set:
            repr = self.nodes[n]["name"] \
                if self.nodes[n]["type"] == NodeType.VARIABLE else \
                f"{self.nodes[n]['name']}{inspect.signature(self.nodes[n]['lambda'])}"

            result += f"{tab * depth}{repr}\n"
            result += self.traverse_nodes(self.successors(n), depth=depth+1)
        return result

    @classmethod
    def from_json(cls, file: str):
        """Builds a GrFN from a JSON object."""
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict, lambdas):
        """
        Builds a GrFN object from a set of extracted function data objects and
        an associated file of lambda functions.

        :param data: [dict] A set of function data object that specify the
                            wiring of a GrFN object
        :param lambdas: [Module] A python module containing actual python
                                 functions to be computed during GrFN execution
        :returns: [GroundedFunctionNetwork] A new GrFN object
        """
        nodes, edges, subgraphs = list(), list(), dict()

        # Get a list of all container/loop plates contained in the data object
        containers = {obj["name"]: obj for obj in data["functions"]
                      if obj["type"] in ["container", "loop_plate"]}

        def process_container(container, inputs):
            """
            Wires the body statements found in a given container/loop plate.

            :param container: [dict] The container object containing the body
                                     statements that specify GrFN wiring.
            :param inputs: [dict: str->Var] A dict of input variables from the
                                            outer container
            """
            con_name = container["name"]
            subgraphs[con_name] = list()
            for stmt in container["body"]:
                is_container = False
                if "name" in stmt:              # Found something other than a container
                    stmt_name = stmt["name"]

                    # Get the type information for identification of stmt type
                    # TODO: replace this with simple lookup from functions
                    short_type = stmt_name[stmt_name.find("__") + 2: stmt_name.rfind("__")]
                    stmt_type = get_node_type(short_type)
                else:                           # Found a container (non loop plate)
                    stmt_name = stmt["function"]
                    is_container = True
                if is_container or stmt_type == NodeType.LOOP:  # Handle container or loop plate
                    container_name = stmt_name

                    # Skip over unmentioned containers
                    if container_name not in containers:
                        continue

                    # Get input set to send into new container
                    new_inputs = {
                        var["variable"]: get_variable_name(var, con_name)
                        if var["index"] != -1 else inputs[var["variable"]]
                        for var in stmt["input"]
                    }

                    # Do wiring of the call to this container
                    process_container(containers[container_name], new_inputs)
                else:                                           # Handle regular statement
                    # Need to wire all inputs to their lambda function and
                    # preserve the input argument order for execution
                    ordered_inputs = list()
                    for var in stmt["input"]:
                        # Check if the node is an input node from an outer container
                        if var["index"] == -1:
                            input_node_name = inputs[var["variable"]]
                        else:
                            input_node_name = get_variable_name(var, con_name)

                        # Add input node and node unique name to edges, subgraph set, and arg set
                        ordered_inputs.append(input_node_name)
                        subgraphs[con_name].append(input_node_name)
                        edges.append((input_node_name, stmt_name))
                        nodes.append((input_node_name, {
                            "name": input_node_name,
                            "type": NodeType.VARIABLE,
                            "value": None,
                            "scope": con_name
                        }))

                    # Add function node name to subgraph set and create function node
                    subgraphs[con_name].append(stmt_name)
                    nodes.append((stmt_name, {
                        "name": stmt_name,
                        "type": stmt_type,
                        "lambda": getattr(lambdas, stmt_name),  # Gets the lambda function
                        "func_inputs": ordered_inputs,          # saves indexed arg ordering
                        "scope": con_name
                    }))

                    # Add output node and node unique name to edges, subgraph set, and arg set
                    out_node_name = get_variable_name(stmt["output"], con_name)
                    subgraphs[con_name].append(out_node_name)
                    edges.append((stmt_name, out_node_name))
                    nodes.append((out_node_name, {
                        "name": out_node_name,
                        "type": NodeType.VARIABLE,
                        "value": None,
                        "scope": con_name
                    }))

        # Use the start field to find the starting container and begin building
        # the GrFN. Building in containers will occur recursively from this call
        process_container(containers[data["start"]], [])
        return cls(nodes, edges, subgraphs)

    @classmethod
    def from_python_file(cls, python_file, lambdas_path, json_filename, stem):
        """Builds GrFN object from Python file."""
        with open(python_file, "r") as f:
            pySrc = f.read()
        return cls.from_python_src(pySrc, lambdas_path, json_filename, stem)

    @classmethod
    def from_python_src(cls, pySrc, lambdas_path, json_filename, stem):
        """Builds GrFN object from Python source code."""
        asts = [ast.parse(pySrc)]
        pgm_dict = genPGM.create_pgm_dict(
            lambdas_path, asts, json_filename, save_file=True,
        )
        lambdas = importlib.__import__(stem + "_lambdas")
        return cls.from_dict(pgm_dict, lambdas)

    @classmethod
    def from_fortran_file(cls, fortran_file, tmpdir="."):
        """Builds GrFN object from a Fortran program."""
        stem = Path(fortran_file).stem
        preprocessed_fortran_file = f"{tmpdir}/{stem}_preprocessed.f"
        lambdas_path = f"{tmpdir}/{stem}_lambdas.py"
        json_filename = stem + ".json"

        with open(fortran_file, "r") as f:
            inputLines = f.readlines()

        with open(preprocessed_fortran_file, "w") as f:
            f.write(preprocessor.process(inputLines))

        xml_string = sp.run(
            [
                "java",
                "fortran.ofp.FrontEnd",
                "--class",
                "fortran.ofp.XMLPrinter",
                "--verbosity",
                "0",
                preprocessed_fortran_file,
            ],
            stdout=sp.PIPE,
        ).stdout
        trees = [ET.fromstring(xml_string)]
        comments = get_comments.get_comments(preprocessed_fortran_file)
        os.remove(preprocessed_fortran_file)
        xml_to_json_translator = translate.XMLToJSONTranslator()
        outputDict = xml_to_json_translator.analyze(trees, comments)
        pySrc = pyTranslate.create_python_string(outputDict)[0][0]

        return cls.from_python_src(pySrc, lambdas_path, json_filename, stem)

    def clear(self):
        """Clear variable node for next computation."""
        for n in self.nodes():
            if self.nodes[n]["type"] == NodeType.VARIABLE:
                self.nodes[n]["value"] = None

    @timeit
    def run(self, inputs):
        """
        Executes the GrFN over a particular set of inputs and returns the result.

        :param inputs: [dict: str->{float, iterable}] Input set where keys are
                       the names of input nodes in the GrFN and each key points
                       to a set of input values (or just one)
        :returns: [{float, iterable}] A set of outputs from executing the GrFN,
                                      one for every set of inputs.
        """
        # Abort run if inputs does not match our expected input set
        if len(inputs) != len(self.inputs):
            raise ValueError("Incorrect number of inputs.")

        def update_function_stack(stack, successors):
            """
            Adds all new functions from the successors of the output variables
            to their proper location in the function stack. Position is
            determined by the maximum distance needed to be traveled to get to
            the output from a given function node.

            :param stack: [dict: int->str] The function stack to be updated
            :param successors: [list: str] The list of successor node names
            """
            for n in successors:
                # Get all paths from the function node to an output
                paths = list(nx.all_simple_paths(self, n, self.output_node))

                # No paths found, this node does not lead to the output
                if len(paths) == 0:
                    continue

                # Use the maximum distance as hash value to place function in stack
                dist = max([len(p) for p in paths])

                # Add function to the stack
                if dist in stack:
                    if n not in stack[dist]:        # Do not add if already in the stack
                        stack[dist].append(n)
                else:
                    stack[dist] = [n]

        # Use deepcopy to avoid ever exanding input set on second run
        defined_variables = deepcopy(inputs)
        function_stack = dict()

        # Build initial function stack from inputs
        for node_name, val in inputs.items():
            self.nodes[node_name]["value"] = val
            update_function_stack(function_stack, self.successors(node_name))

        while len(function_stack) > 0:
            # We will calculate the functions that are farthest away first to
            # ensure all values are populated before computing with them
            max_dist = max(function_stack.keys())
            outputs = list()
            for func_name in function_stack[max_dist]:
                # Get function arguments via signature derived from JSON
                signature = self.nodes[func_name]["func_inputs"]
                lambda_fn = self.nodes[func_name]["lambda"]

                # Run the lambda function
                res = lambda_fn(*tuple(defined_variables[n] for n in signature))

                # Get the output node for this function
                # TODO: extend this to handle multiple outputs in the future
                output_node = list(self.successors(func_name))[0]

                # Save the output
                self.nodes[output_node]["value"] = res
                defined_variables[output_node] = res
                outputs.append(output_node)             # Use this to build successors
            del function_stack[max_dist]                # Done processing this layer

            # Add new successors from computed outputs
            all_successors = list(set(n for node in outputs for n in self.successors(node)))
            update_function_stack(function_stack, all_successors)

        # return the output
        return self.nodes[self.output_node]["value"]

    def to_ProgramAnalysisGraph(self):
        def gather_nodes_and_edges(prev_name, inputs, nodes, edges):
            for name in inputs:
                uniq_name = name[name.find("::") + 2: name.rfind("_")]
                nodes.append((uniq_name, {
                    "name": name,
                    "cag_label": uniq_name,
                    "value": self.nodes[name]["value"]
                }))

                if prev_name is not None:
                    edges.append((prev_name, uniq_name))

                next_inputs = list(set([v for f in self.successors(name)
                                        for v in self.successors(f)]))
                gather_nodes_and_edges(uniq_name, next_inputs, nodes, edges)
            return nodes, edges

        nodes, edges = gather_nodes_and_edges(None, self.inputs, [], [])
        PAG = ProgramAnalysisGraph()
        PAG.add_nodes_from(nodes)
        PAG.add_edges_from(edges)
        return PAG


class ProgramAnalysisGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NodeType(Enum):
    """Enum for the identification/tracking of a JSON function objects type."""
    CONTAINER = 0
    LOOP = 1
    ASSIGN = 2
    CONDITION = 3
    DECISION = 4
    VARIABLE = 5


def get_variable_name(var_dict, container_name):
    """Returnns the unique node name of a variable."""
    return f"{container_name}::{var_dict['variable']}_{var_dict['index']}"


def get_node_type(type_str):
    """Returns the NodeType given a name of a JSON function object."""
    if type_str == "container":
        return NodeType.CONTAINER
    elif type_str == "loop_plate":
        return NodeType.LOOP
    elif type_str == "assign":
        return NodeType.ASSIGN
    elif type_str == "condition":
        return NodeType.CONDITION
    elif type_str == "decision":
        return NodeType.DECISION
    else:
        raise ValueError("Unrecognized type string: ", type_str)
