import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from enum import Enum
import importlib
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

    @classmethod
    def from_json(cls, file: str):
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict, lambdas):
        nodes, edges, subgraphs = list(), list(), dict()
        containers_and_plates = [obj for obj in data["functions"] if obj["type"] in ["container", "loop_plate"]]
        reg_patt = r"(__assign__|__decision__|__condition__)"
        for obj in containers_and_plates:
            func_type = get_node_type(obj["type"])
            func_name = obj["name"]
            contained_graphs = list()
            # TODO: check how to do subgraphs with scopes
            for stmt in obj["body"]:
                lambda_name = stmt["name"]      # re.sub(reg_patt, "__lambda__", stmt["name"])
                short_type = stmt["name"][stmt["name"].find("__") + 2: stmt["name"].rfind("__")]
                stmt_type = get_node_type(short_type)
                stmt_name = stmt["name"]
                if stmt_type != NodeType.LOOP and stmt_type != NodeType.CONTAINER:
                    nodes.append((stmt_name, {
                        "name": stmt_name,
                        # TODO: check to see if the assignment below is right
                        "lambda": getattr(lambdas, lambda_name),
                        "type": stmt_type,
                        "scope": func_name
                    }))

                    for var in stmt["input"]:
                        input_node_name = get_variable_name(var)
                        edges.append((input_node_name, stmt_name))
                        nodes.append((input_node_name, {
                            "name": input_node_name,
                            "type": NodeType.VARIABLE,
                            "value": None,
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
