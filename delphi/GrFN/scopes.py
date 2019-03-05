import os
import ast
from abc import ABCMeta, abstractmethod
from typing import Dict
import json
from pygraphviz import AGraph
import platform
from typing import Dict
from pathlib import Path
import xml.etree.ElementTree as ET
from delphi.translators.for2py.scripts import (
    f2py_pp,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)
import subprocess as sp


rv_maroon = "#650021"


class Scope(metaclass=ABCMeta):
    def __init__(self, name, data):
        self.name = name
        self.parent_scope = None
        self.child_names = list()
        self.child_scopes = list()
        self.child_vars = list()

        self.inputs = list()
        self.calls = dict()

        self.nodes = list()
        self.edges = list()

        self.json = data

        # NOTE: default edge color for debugging
        self.border_color = "red"

        self.build_child_names()

    @classmethod
    def from_json(cls, file: str):
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict):
        scope_types_dict = {
            "container": ContainerScope,
            "loop_plate": LoopScope,
        }

        scopes = {
            f["name"]: scope_types_dict[f["type"]](f["name"], f)
            for f in data["functions"]
            if f["type"] in scope_types_dict
        }

        # Make a list of all scopes by scope names
        scope_names = list(scopes.keys())

        # Remove pseudo-scopes we wish to not display (such as print)
        for scope in scopes.values():
            scope.remove_non_scope_children(scope_names)

        # Build the nested tree of scopes using recursion

        if scopes.get(data["start"]) is not None:
            root = scopes[data["start"]]
        else:
            non_lambdas = [f["name"] for f in data["functions"] if "__" not in f["name"]]
            # TODO Right now, only the first subroutine is taken as the root -
            # in the future, we will need to merge scope trees from multiple
            # subroutines.
            root_func_name = non_lambdas[0]     # There should only ever be one, otherwise we need multiple roots
            root = scopes[root_func_name]

        root.build_scope_tree(scopes)
        root.setup_from_json()
        return root

    @classmethod
    def from_python_src(cls, pySrc, lambdas_path, json_filename):
        asts = [ast.parse(pySrc)]
        pgm_dict = genPGM.create_pgm_dict(
            lambdas_path, asts, json_filename
        )
        return cls.from_dict(pgm_dict)

    @classmethod
    def from_fortran_file(cls, fortran_file, tmpdir="."):
        stem = Path(fortran_file).stem
        preprocessed_fortran_file = f"{tmpdir}/{stem}_preprocessed.f"
        lambdas_path = f"{tmpdir}/{stem}_lambdas.py"
        json_filename = stem + ".json"

        with open(fortran_file, "r") as f:
            inputLines = f.readlines()

        with open(preprocessed_fortran_file, "w") as f:
            f.write(f2py_pp.process(inputLines))

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
        return cls.from_python_src(pySrc, lambdas_path, json_filename)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        root = "(R) " if self.parent_scope is None else ""
        return f"{root}{self.name}: {self.child_names}"

    def to_agraph(self):
        A = AGraph(directed=True)
        A.node_attr["shape"] = "rectangle"
        A.graph_attr["rankdir"] = "TB"

        operating_system = platform.system()

        if operating_system == "Darwin":
            font = "Menlo"
        elif operating_system == "Windows":
            font = "Consolas"
        else:
            font = "Ubuntu Mono"

        A.node_attr["fontname"] = font
        A.graph_attr["fontname"] = font

        self.build_containment_graph(A)
        return A

    def build_child_names(self):
        for expr in self.json["body"]:
            if expr.get("name") is not None and "loop_plate" in expr["name"]:
                self.child_names.append(expr["name"])
            if expr.get("function") is not None:
                self.child_names.append(expr["function"])

    def build_scope_tree(self, all_scopes):
        for name in self.child_names:
            new_scope = all_scopes[name]
            new_scope.parent_scope = self
            new_scope.build_scope_tree(all_scopes)
            self.child_scopes.append(new_scope)

    def is_in_loop(self):
        if isinstance(self, LoopScope):
            return self
        elif self.parent_scope is not None:
            return self.parent_scope.is_in_loop()
        else:
            return None

    def make_var_node(self, name, idx, scp, inp_node=False, child_loop=None):
        if child_loop is not None:
            loop_scope = child_loop
        else:
            loop_scope = self.is_in_loop()

        if loop_scope is not None:
            if name == loop_scope.index_var.name:
                return LoopVariableNode(
                    name=name,
                    idx=idx,
                    scp=scp,
                    is_index=True,
                    start=loop_scope.index_var.start,
                    end=loop_scope.index_var.end,
                )

            if inp_node:
                return LoopVariableNode(
                    name=name,
                    idx=idx,
                    scp=scp,
                    loop_index=-1,
                    loop_var=loop_scope.index_var.name,
                )
            return LoopVariableNode(
                name=name, idx=idx, scp=scp, loop_var=loop_scope.index_var.name
            )
        else:
            return FuncVariableNode(name=name, idx=idx, scp=scp)

    def make_action_node(self, name):
        cut = name.rfind("_")
        inst_name = name[:cut]
        index = name[cut + 1 :]
        return ActionNode(name=inst_name, idx=index, scp=self.name)

    def remove_non_scope_children(self, scopes):
        self.child_names = [c for c in self.child_names if c in scopes]

    @abstractmethod
    def setup_from_json(self, input_vars=[]):
        for expr in self.json["body"]:
            if expr.get("name") is not None:
                # Do this for regular assignment/decision/condition operations
                # and loop_plate(s)
                instruction = expr["name"]
                if len(expr.get("output")) > 0:
                    action_node = self.make_action_node(instruction)
                    self.nodes.append(action_node)
                if expr.get("input") is not None:
                    # This is a regular operation node
                    for i in expr["input"]:
                        if i.get("variable") is not None:
                            if (
                                isinstance(self, LoopScope)
                                and int(i["index"]) == -1
                            ) or (
                                isinstance(self, ContainerScope)
                                and int(i["index"]) == 0
                            ):
                                found = False
                                for var in input_vars:
                                    if i["variable"] == var.name:
                                        inp_node = var
                                        found = True
                                        break
                                if not found:
                                    inp_node = self.make_var_node(
                                        i["variable"], i["index"], self.name
                                    )
                            else:
                                inp_node = self.make_var_node(
                                    i["variable"], i["index"], self.name
                                )
                            self.nodes.append(inp_node)
                            self.edges.append((inp_node, action_node))
                            action_node.inputs.append(inp_node)
                elif expr.get("inputs") is not None:
                    # This is a loop_plate node
                    plate_vars = list()
                    plate_index = len(self.child_vars)
                    loop_scope = self.child_scopes[plate_index]
                    for var in expr["inputs"]:
                        # NOTE: cheating for now
                        new_var = self.make_var_node(
                            var,
                            "2",
                            self.name,
                            inp_node=True,
                            child_loop=loop_scope,
                        )
                        plate_vars.append(new_var)
                    self.child_vars.append(plate_vars)

                o = expr["output"]
                if o.get("variable") is not None:
                    found = False
                    for node in self.nodes:
                        if o["variable"] == node.name:
                            scope = self.name
                            found = True
                            break
                    if (
                        not found
                        and int(o["index"]) == 0
                        and o["variable"] in self.inputs
                        and self.parent_scope is not None
                    ):
                        scope = self.parent_scope.name
                    else:
                        scope = self.name
                    out_node = self.make_var_node(
                        o["variable"], o["index"], scope
                    )
                    self.nodes.append(out_node)
                    self.edges.append((action_node, out_node))
                    action_node.output = out_node
            elif (
                expr.get("function") is not None
                and expr["function"] in self.child_names
            ):
                # Do this for function calls
                call_vars = list()
                for var in expr["input"]:
                    if int(var["index"]) < 1:
                        scope = self.parent_scope.name
                    else:
                        scope = self.name

                    # NOTE: cheating for now
                    idx = "2" if int(var["index"]) == -1 else "0"
                    inode = int(var["index"]) == -1
                    inp_node = self.make_var_node(
                        var["variable"], idx, scope, inp_node=inode
                    )
                    call_vars.append(inp_node)
                    self.child_vars.append(call_vars)

    def add_nodes(self, sub):
        for node in self.nodes:
            sub.add_node(
                node.unique_name(),
                shape=node.shape,
                color=node.color,
                node_type=type(node).__name__,
                lambda_fn=getattr(node, "lambda_fn", None),
                label=node.get_label(),
                cag_label=node.name,
                index_var=getattr(node, "loop_var", None),
                is_index=getattr(node, "is_index", None),
                start=getattr(node, "start", None),
                end=getattr(node, "end", None),
                index=int(getattr(node, "index", None)),
                parent=f"cluster_{self.name}",
            )

    def add_edges(self, sub):
        edges = [
            (src.unique_name(), dst.unique_name()) for src, dst in self.edges
        ]

        sub.add_edges_from(edges)

    def build_containment_graph(self, graph):
        sub = graph.add_subgraph(
            name=f"cluster_{self.name}",
            label=self.name,
            style="bold, rounded",
            border_color=self.border_color,
        )

        self.add_nodes(sub)
        self.add_edges(sub)

        for child in self.child_scopes:
            child.build_containment_graph(sub)


class LoopScope(Scope):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.border_color = "blue"

        self.index_var = LoopVariableNode(
            name=self.json["index_variable"],
            idx="0",
            scp=self.name,
            is_index=True,
            start=self.json["index_iteration_range"]["start"]["value"],
            end=self.json["index_iteration_range"]["end"]["value"],
        )

    def setup_from_json(self, vars=[]):
        self.inputs += self.json["input"]

        super().setup_from_json(vars)

        for child, vars in zip(self.child_scopes, self.child_vars):
            child.setup_from_json(vars)


class ContainerScope(Scope):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.border_color = "green"

    def setup_from_json(self, vars=[]):
        for var in self.json["input"]:
            self.inputs.append(var["name"])

        super().setup_from_json(vars)

        for child, vars in zip(self.child_scopes, self.child_vars):
            child.setup_from_json(vars)


class Node(metaclass=ABCMeta):
    def __init__(self, name="", idx="", scp=""):
        self.name = name
        self.index = idx
        self.scope = scp
        self.color = rv_maroon
        self.shape = "ellipse"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.unique_name()

    def unique_name(self):
        return f"{self.name}_{self.index}__{self.scope}"

    def get_label(self):
        return '_'.join((self.name, str(self.index)))


class FuncVariableNode(Node):
    def __init__(self, name="", idx="", scp=""):
        super().__init__(name=name, idx=idx, scp=scp)


class ActionNode(Node):
    def __init__(self, name="", idx="", scp=""):
        super().__init__(name=name, idx=idx, scp=scp)
        start = name.find("__")
        end = name.rfind("__")
        self.action = name[start : end + 2]
        self.inputs: List = []
        self.output = None
        self.color = "black"
        self.shape = "rectangle"
        self.lambda_fn = (
            "_".join((name, idx))
        )

    def get_label(self):
        return self.action


class LoopVariableNode(Node):
    def __init__(
        self,
        name="",
        idx="",
        scp="",
        is_index=False,
        loop_var="",
        loop_index=0,
        start="",
        end="",
    ):

        super().__init__(name=name, idx=idx, scp=scp)

        self.start = start
        self.end = end
        self.is_index = is_index
        if not self.is_index:
            self.loop_var = loop_var
            self.loop_index = loop_index

    def get_label(self):
        if not self.is_index:
            return f"{self.name}\n@{self.loop_var}={self.loop_index}"
        else:
            return self.name


def scope_tree_from_json(json_data: Dict) -> Scope:
    """
    Parses through a dictionary of data from a JSON spec of a FORTRAN program
    and returns a tree of Scope objects that represent the properly nested
    scopes found in the JSON data.

    Args:
        json_data: A dict of all data found when parsing the FORTRAN program

    Returns:
        A Scope that serves as the root of a tree of Scopes
    """

    # Build a new scope object for each function and loop_plate object. Index
    # scopes into a dict by (scope_name |-> scope)

    scope_types_dict = {"container": ContainerScope, "loop_plate": LoopScope}

    scopes = {
        f["name"]: scope_types_dict[f["type"]](f["name"], f)
        for f in json_data["functions"]
        if f["type"] in scope_types_dict
    }

    # Make a list of all scopes by scope names
    scope_names = list(scopes.keys())

    # Remove pseudo-scopes we wish to not display (such as print)
    for scope in scopes.values():
        scope.remove_non_scope_children(scope_names)

    # Build the nested tree of scopes using recursion
    root = scopes[json_data["start"]]
    root.build_scope_tree(scopes)
    root.setup_from_json()
    return root
