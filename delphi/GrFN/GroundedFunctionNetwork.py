import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from enum import Enum
import importlib
import inspect
import json
import re
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
        self.output_node = self.outputs[0]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.traverse_nodes(self.inputs)

    def traverse_nodes(self, node_set, depth=0):
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
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict, lambdas):
        nodes, edges, subgraphs = list(), list(), dict()
        containers_and_plates = [obj for obj in data["functions"]
                                 if obj["type"] in ["container", "loop_plate"]]

        for obj in containers_and_plates:
            # func_type = get_node_type(obj["type"])
            func_name = obj["name"]
            contained_graphs = list()
            # TODO: check how to do subgraphs with scopes
            for stmt in obj["body"]:
                lambda_name = stmt["name"]
                short_type = stmt["name"][stmt["name"].find("__") + 2: stmt["name"].rfind("__")]
                stmt_type = get_node_type(short_type)
                stmt_name = stmt["name"]
                if stmt_type != NodeType.LOOP and stmt_type != NodeType.CONTAINER:
                    ordered_inputs = list()
                    for var in stmt["input"]:
                        input_node_name = get_variable_name(var)
                        ordered_inputs.append(input_node_name)
                        edges.append((input_node_name, stmt_name))
                        nodes.append((input_node_name, {
                            "name": input_node_name,
                            "type": NodeType.VARIABLE,
                            "value": None,
                            "scope": func_name
                        }))

                    nodes.append((stmt_name, {
                        "name": stmt_name,
                        "type": stmt_type,
                        "lambda": getattr(lambdas, lambda_name),
                        "func_inputs": ordered_inputs,
                        "scope": func_name
                    }))

                    out_node_name = get_variable_name(stmt["output"])
                    edges.append((stmt_name, out_node_name))
                    nodes.append((out_node_name, {
                        "name": out_node_name,
                        "type": NodeType.VARIABLE,
                        "value": None,
                        "scope": func_name
                    }))
                else:
                    contained_graphs.append({
                        "name": stmt_name,
                        "type": stmt_type
                    })

        return cls(nodes, edges, subgraphs)

    @classmethod
    def from_python_src(cls, pySrc, lambdas_path, json_filename, stem):
        asts = [ast.parse(pySrc)]
        pgm_dict = genPGM.create_pgm_dict(
            lambdas_path, asts, json_filename, save_file=True,
        )
        lambdas = importlib.__import__(stem + "_lambdas")
        return cls.from_dict(pgm_dict, lambdas)

    @classmethod
    def from_fortran_file(cls, fortran_file, tmpdir="."):
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
        for n in self.nodes():
            if self.nodes[n]["type"] == NodeType.VARIABLE:
                self.nodes[n]["value"] = None

    def run(self, inputs):
        if len(inputs) != len(self.inputs):
            raise ValueError("Incorrect number of inputs.")

        def update_function_stack(stack, successors):
            for n in successors:
                paths = nx.all_simple_paths(self, n, self.output_node)
                dist = max([len(p) for p in paths])
                if dist in stack:
                    if n not in stack[dist]:
                        stack[dist].append(n)
                else:
                    stack[dist] = [n]

        defined_variables = inputs
        function_stack = dict()
        for node_name, val in inputs.items():
            self.nodes[node_name]["value"] = val
            update_function_stack(function_stack, self.successors(node_name))

        while len(function_stack) > 0:
            max_dist = max(function_stack.keys())
            outputs = list()
            for func_name in function_stack[max_dist]:
                signature = self.nodes[func_name]["func_inputs"]
                lambda_fn = self.nodes[func_name]["lambda"]
                res = lambda_fn(*tuple(defined_variables[n] for n in signature))
                output_node = list(self.successors(func_name))[0]
                self.nodes[output_node]["value"] = res
                defined_variables[output_node] = res
                outputs.append(output_node)
            del function_stack[max_dist]
            all_successors = list(set(n for node in outputs for n in self.successors(node)))
            update_function_stack(function_stack, all_successors)

        return self.nodes[self.output_node]["value"]


class NodeType(Enum):
    CONTAINER = 0
    LOOP = 1
    ASSIGN = 2
    CONDITION = 3
    DECISION = 4
    VARIABLE = 5


def get_variable_name(var_dict):
    return "{}_{}".format(var_dict["variable"], var_dict["index"])


def get_node_type(type_str):
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
