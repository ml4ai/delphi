from typing import List, Dict, Iterable, Set, Union, Any
from abc import ABC, abstractmethod
from functools import singledispatch
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4, UUID
import importlib
import inspect
import json
import os

import networkx as nx
import numpy as np
from networkx.algorithms.simple_paths import all_simple_paths

from delphi.GrFN.structures import (
    GenericContainer,
    FuncContainer,
    CondContainer,
    LoopContainer,
    LambdaType,
    GenericStmt,
    CallStmt,
    LambdaStmt,
    ContainerIdentifier,
    VariableIdentifier,
    TypeIdentifier,
    VariableDefinition,
    TypeDefinition,
)
from delphi.utils.misc import choose_font
from delphi.translators.for2py import f2grfn


FONT = choose_font()

dodgerblue3 = "#1874CD"
forestgreen = "#228b22"


class ComputationalGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FCG = self.to_FCG()
        self.function_sets = self.build_function_sets()

    @staticmethod
    def var_shortname(long_var_name):
        (
            module,
            var_scope,
            container_name,
            container_index,
            var_name,
            var_index,
        ) = long_var_name.split("::")
        return var_name

    def get_input_nodes(self) -> List[str]:
        """ Get all input nodes from a network. """
        return [n for n, d in self.in_degree() if d == 0]

    def get_output_nodes(self) -> List[str]:
        """ Get all output nodes from a network. """
        return [n for n, d in self.out_degree() if d == 0]

    def to_FCG(self):
        G = nx.DiGraph()
        for (name, attrs) in self.nodes(data=True):
            if attrs["type"] == "function":
                for predecessor_variable in self.predecessors(name):
                    for predecessor_function in self.predecessors(
                        predecessor_variable
                    ):
                        G.add_edge(predecessor_function, name)
        return G

    def build_function_sets(self):
        initial_funcs = [n for n, d in self.FCG.in_degree() if d == 0]
        distances = dict()

        def find_distances(funcs, dist):
            all_successors = list()
            for func in funcs:
                distances[func] = dist
                all_successors.extend(self.FCG.successors(func))
            if len(all_successors) > 0:
                find_distances(list(set(all_successors)), dist + 1)

        find_distances(initial_funcs, 0)
        call_sets = dict()
        for func_name, call_dist in distances.items():
            if call_dist in call_sets:
                call_sets[call_dist].add(func_name)
            else:
                call_sets[call_dist] = {func_name}

        function_set_dists = sorted(
            call_sets.items(), key=lambda t: (t[0], len(t[1]))
        )
        function_sets = [func_set for _, func_set in function_set_dists]
        return function_sets

    def run(
        self, inputs: Dict[str, Union[float, Iterable]],
    ) -> Union[float, Iterable]:
        """Executes the GrFN over a particular set of inputs and returns the
        result.

        Args:
            inputs: Input set where keys are the names of input nodes in the
                GrFN and each key points to a set of input values (or just one)

        Returns:
            A set of outputs from executing the GrFN, one for every set of
            inputs.
        """
        full_inputs = {self.input_name_map[n]: v for n, v in inputs.items()}
        # Set input values
        for i in self.inputs:
            value = full_inputs[i]
            if isinstance(value, float):
                value = np.array([value], dtype=np.float32)
            if isinstance(value, int):
                value = np.array([value], dtype=np.int32)
            elif isinstance(value, list):
                value = np.array(value, dtype=np.float32)

            self.nodes[i]["value"] = value
        for func_set in self.function_sets:
            for func_name in func_set:
                lambda_fn = self.nodes[func_name]["lambda_fn"]
                output_node = list(self.successors(func_name))[0]

                signature = self.nodes[func_name]["func_inputs"]
                input_values = [self.nodes[n]["value"] for n in signature]
                res = lambda_fn(*input_values)

                # Convert output to a NumPy matrix if a constant was returned
                if len(input_values) == 0:
                    res = np.array(res, dtype=np.float32)

                self.nodes[output_node]["value"] = res

        # Return the output
        return [self.nodes[o]["value"] for o in self.outputs]

    def to_CAG(self):
        """ Export to a Causal Analysis Graph (CAG) PyGraphviz AGraph object.
        The CAG shows the influence relationships between the variables and
        elides the function nodes."""

        G = nx.DiGraph()
        for (name, attrs) in self.nodes(data=True):
            if attrs["type"] == "variable":
                cag_name = attrs["cag_label"]
                G.add_node(cag_name, **attrs)
                for pred_fn in self.predecessors(name):
                    for pred_var in self.predecessors(pred_fn):
                        v_attrs = self.nodes[pred_var]
                        v_name = v_attrs["cag_label"]
                        G.add_node(v_name, **self.nodes[pred_var])
                        G.add_edge(v_name, cag_name)

        return G


class GroundedFunctionNetwork(ComputationalGraph):
    """
    Representation of a GrFN model as a DiGraph with a set of input nodes and
    currently a single output. The DiGraph is composed of variable nodes and
    function nodes. Function nodes store an actual Python function with the
    expected set of ordered input arguments that correspond to the variable
    inputs of that node. Variable nodes store a value. This value can be any
    data type found in Python. When no value exists for a variable the value
    key will be set to None. Importantly only function nodes can be children or
    parents of variable nodes, and the reverse is also true. Both variable and
    function nodes can be inputs, but the output will always be a variable
    node.
    """

    def __init__(self, G, scope_tree, outputs):
        super().__init__(G)
        self.outputs = outputs
        self.inputs = [
            n
            for n, d in self.in_degree()
            if d == 0 and self.nodes[n]["type"] == "variable"
        ]
        self.input_name_map = {
            self.var_shortname(name): name for name in self.inputs
        }
        # self.outputs = outputs
        self.scope_tree = scope_tree

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "\n".join(self.traverse_nodes(self.inputs))

    def traverse_nodes(self, node_set, depth=0):
        """BFS traversal of nodes that returns name traversal as large string.

        Args:
            node_set: Set of input nodes to begin traversal.
            depth: Current traversal depth for child node viewing.

        Returns:
            type: String containing tabbed traversal view.
        """

        tab = "  "
        result = list()
        for n in node_set:
            repr = (
                n
                if self.nodes[n]["type"] == "variable"
                else f"{n}{inspect.signature(self.nodes[n]['lambda_fn'])}"
            )

            result.append(f"{tab * depth}{repr}")
            result.extend(
                self.traverse_nodes(self.successors(n), depth=depth + 1)
            )
        return result

    @classmethod
    def from_json_and_lambdas(cls, file: str, lambdas):
        """Builds a GrFN from a JSON object.

        Args:
            cls: The class variable for object creation.
            file: Filename of a GrFN JSON file.
            lambdas: A lambdas module

        Returns:
            type: A GroundedFunctionNetwork object.

        """
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, lambdas)

    @classmethod
    def from_dict(cls, data: Dict, lambdas_path):
        """Builds a GrFN object from a set of extracted function data objects
        and an associated file of lambda functions.

        Args:
            cls: The class variable for object creation.
            data: A set of function data object that specify the wiring of a
                  GrFN object.
            lambdas_path: Path to a lambdas file containing functions to be
                computed during GrFN execution.

        Returns:
            A GroundedFunctionNetwork object.

        """
        lambdas = importlib.__import__(str(Path(lambdas_path).stem))
        # functions = {d["name"]: d for d in data["containers"]}
        occurrences = {}
        G = nx.DiGraph()
        scope_tree = nx.DiGraph()

        def identity(x):
            return x

        def add_variable_node(
            parent: str, basename: str, index: str, is_exit: bool = False
        ):
            full_var_name = ""  # make_variable_name(parent, basename, index)
            G.add_node(
                full_var_name,
                type="variable",
                color="crimson",
                fontcolor="white" if is_exit else "black",
                fillcolor="crimson" if is_exit else "white",
                style="filled" if is_exit else "",
                parent=parent,
                label=f"{basename}::{index}",
                cag_label=f"{basename}",
                basename=basename,
                padding=15,
                value=None,
            )
            return full_var_name

        @singledispatch
        def process_statment(stmt: GenericStmt, inputs: list) -> None:
            raise TypeError(f"Unrecognized statment type: {type(stmt)}")

        @process_statment.register
        def _(stmt: LambdaStmt, inputs: list):
            for output in stmt.outputs:
                exit_node = output.basename == "EXIT"
                node_name = add_variable_node(
                    stmt.parent.name,
                    output.basename,
                    output.index,
                    is_exit=exit_node,
                )
                G.add_edge(stmt.lambda_node_name, node_name)

            ordered_inputs = list()
            # Dummy code to stop warnings
            lambda_node_name = ""
            lambda_name = ""
            stmt_type = list()
            scope = None
            for inp in stmt["input"]:
                if inp.endswith("-1"):
                    (parent, var_name, idx) = inputs[inp]
                else:
                    parent = scope.name
                    (_, var_name, idx) = inp.split("::")

                node_name = add_variable_node(parent, var_name, idx)
                ordered_inputs.append(node_name)
                G.add_edge(node_name, lambda_node_name)

            G.add_node(
                lambda_node_name,
                type="function",
                lambda_fn=getattr(lambdas, lambda_name),
                func_inputs=ordered_inputs,
                visited=False,
                shape="rectangle",
                parent=scope.name,
                label=stmt_type[0].upper(),
                padding=10,
            )

        def process_call_statement(stmt, scope, inputs, cname):
            container_name = stmt["function"]["name"]
            if container_name not in occurrences:
                occurrences[container_name] = 0

            # new_container = functions[container_name]
            new_scope = None
            # new_scope = create_container_node(
            #     new_container, occurrences[container_name], parent=scope
            # )

            input_values = list()
            for inp in stmt["input"]:
                if inp.endswith("-1"):
                    (parent, var_name, idx) = inputs[inp]
                else:
                    parent = scope.name
                    (_, var_name, idx) = inp.split("::")
                input_values.append((parent, var_name, idx))

            callee_ret, callee_up = process_container(
                new_scope, input_values, container_name
            )

            caller_ret, caller_up = list(), list()
            for var in stmt["output"]:
                parent = scope.name
                (_, var_name, idx) = var.split("::")
                node_name = add_variable_node(parent, var_name, idx)
                caller_ret.append(node_name)

            for var in stmt["updated"]:
                parent = scope.name
                (_, var_name, idx) = var.split("::")
                node_name = add_variable_node(parent, var_name, idx)
                caller_up.append(node_name)

            for callee_var, caller_var in zip(callee_ret, caller_ret):
                lambda_node_name = f"{callee_var}-->{caller_var}"
                G.add_node(
                    lambda_node_name,
                    type="function",
                    lambda_fn=identity,
                    func_inputs=[callee_var],
                    shape="rectangle",
                    parent=scope.name,
                    label="A",
                    padding=10,
                )
                G.add_edge(callee_var, lambda_node_name)
                G.add_edge(lambda_node_name, caller_var)

            for callee_var, caller_var in zip(callee_up, caller_up):
                lambda_node_name = f"{callee_var}-->{caller_var}"
                G.add_node(
                    lambda_node_name,
                    type="function",
                    lambda_fn=identity,
                    func_inputs=[callee_var],
                    shape="rectangle",
                    parent=scope.name,
                    label="A",
                    padding=10,
                )
                G.add_edge(callee_var, lambda_node_name)
                G.add_edge(lambda_node_name, caller_var)
            occurrences[container_name] += 1

        @singledispatch
        def process_container(
            scope: GenericContainer, scope_inputs: list
        ) -> tuple:
            raise TypeError(f"Unrecognized container scope: {type(scope)}")

        @process_container.register
        def _(scope: FuncContainer, scope_inputs: list) -> tuple:
            if len(scope.arguments) == len(scope_inputs):
                input_vars = {
                    a: v for a, v in zip(scope.arguments, scope_inputs)
                }
            elif len(scope.arguments) > 0:
                input_vars = {
                    a: (scope.name,) + tuple(a.split("::")[1:])
                    for a in scope.arguments
                }

            for stmt in scope.body:
                func_def = stmt["function"]
                func_type = func_def["type"]
                if func_type == "lambda":
                    pass
                    # process_wiring_statement(
                    #     stmt, scope, input_vars, scope.name
                    # )
                elif func_type == "container":
                    process_call_statement(stmt, scope, input_vars, scope.name)
                else:
                    raise ValueError(f"Undefined function type: {func_type}")

            scope_tree.add_node(scope.name, color="forestgreen")
            if scope.parent is not None:
                scope_tree.add_edge(scope.parent.name, scope.name)

            return_list, updated_list = list(), list()
            for var_name in scope.returns:
                (_, basename, idx) = var_name.split("::")
                return_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )

            for var_name in scope.updated:
                (_, basename, idx) = var_name.split("::")
                updated_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )
            return return_list, updated_list

        @process_container.register
        def _(scope: CondContainer, scope_inputs: list) -> tuple:
            if len(scope.arguments) == len(scope_inputs):
                input_vars = {
                    a: v for a, v in zip(scope.arguments, scope_inputs)
                }
            elif len(scope.arguments) > 0:
                input_vars = {
                    a: (scope.name,) + tuple(a.split("::")[1:])
                    for a in scope.arguments
                }

            conditions = list()
            for cond_obj in scope.body:
                cond = cond_obj["condition"]
                stmts = cond_obj["statements"]

                if cond is None:
                    conditions.append(None)
                else:
                    # FIXME: generalize cond handling in the future
                    # cond_stmt = cond[0]
                    pass
                    # process_wiring_statement(
                    #     cond_stmt, scope, input_vars, scope.name
                    # )

                for stmt in stmts:
                    func_def = stmt["function"]
                    func_type = func_def["type"]
                    if func_type == "lambda":
                        # process_wiring_statement(
                        #     stmt, scope, input_vars, scope.name
                        # )
                        pass
                    elif func_type == "container":
                        process_call_statement(
                            stmt, scope, input_vars, scope.name
                        )
                    else:
                        raise ValueError(
                            f"Undefined function type: {func_type}"
                        )

            return_list, updated_list = list(), list()
            for var_name in scope.returns:
                (_, basename, idx) = var_name.split("::")
                return_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )

            for var_name in scope.updated:
                (_, basename, idx) = var_name.split("::")
                updated_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )
            return return_list, updated_list

        @process_container.register
        def _(scope: LoopContainer, scope_inputs: list) -> tuple:
            if len(scope.arguments) == len(scope_inputs):
                input_vars = {
                    a: v for a, v in zip(scope.arguments, scope_inputs)
                }
            elif len(scope.arguments) > 0:
                input_vars = {
                    a: (scope.name,) + tuple(a.split("::")[1:])
                    for a in scope.arguments
                }

            for stmt in scope.body:
                func_def = stmt["function"]
                func_type = func_def["type"]
                if func_type == "lambda":
                    pass
                    # process_wiring_statement(
                    #     stmt, scope, input_vars, scope.name
                    # )
                elif func_type == "container":
                    process_call_statement(stmt, scope, input_vars, scope.name)
                else:
                    raise ValueError(f"Undefined function type: {func_type}")

            scope_tree.add_node(scope.name, color="navyblue")
            if scope.parent is not None:
                scope_tree.add_edge(scope.parent.name, scope.name)

            return_list, updated_list = list(), list()
            for var_name in scope.returns:
                (_, basename, idx) = var_name.split("::")
                return_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )

            for var_name in scope.updated:
                (_, basename, idx) = var_name.split("::")
                updated_list.append(
                    # make_variable_name(scope.name, basename, idx)
                )
            return return_list, updated_list

        root = data["start"][0]
        occurrences[root] = 0
        # cur_scope = create_container_node(functions[root], occurrences[root])
        # returns, updates = process_container(cur_scope, [])
        # return cls(G, scope_tree, returns + updates)

    @classmethod
    def from_python_file(
        cls, python_file, lambdas_path, json_filename: str, stem: str
    ):
        """Builds GrFN object from Python file."""
        with open(python_file, "r") as f:
            pySrc = f.read()
        return cls.from_python_src(pySrc, lambdas_path, json_filename, stem)

    @classmethod
    def from_python_src(
        cls,
        pySrc,
        python_file: str,
        fortran_file: str,
        module_log_file_path: str,
        mod_mapper_dict: list,
        processing_modules: bool,
    ):
        lambdas_path = python_file.replace(".py", "_lambdas.py")
        # Builds GrFN object from Python source code.
        pgm_dict = f2grfn.generate_grfn(
            pySrc,
            python_file,
            lambdas_path,
            mod_mapper_dict,
            fortran_file,
            module_log_file_path,
            processing_modules,
        )

        G = cls.from_dict(pgm_dict, lambdas_path)

        # Cleanup intermediate files.
        variable_map_filename = python_file.replace(".py", "_variable_map.pkl")
        os.remove(variable_map_filename)
        rectified_xml_filename = "rectified_" + str(Path(python_file)).replace(
            ".py", ".xml"
        )
        os.remove(rectified_xml_filename)
        return G

    @classmethod
    def from_fortran_file(cls, fortran_file: str, tmpdir: str = "."):
        """Builds GrFN object from a Fortran program."""

        root_dir = os.path.abspath(tmpdir)

        (
            python_sources,
            translated_python_files,
            mod_mapper_dict,
            fortran_filename,
            module_log_file_path,
            processing_modules,
        ) = f2grfn.fortran_to_grfn(
            fortran_file,
            temp_dir=str(tmpdir),
            root_dir_path=root_dir,
            processing_modules=False,
        )

        # For now, just taking the first translated file.
        # TODO - generalize this.
        python_file = translated_python_files[0]
        G = cls.from_python_src(
            python_sources[0][0],
            python_file,
            fortran_file,
            module_log_file_path,
            mod_mapper_dict,
            processing_modules,
        )

        return G

    @classmethod
    def from_fortran_src(cls, fortran_src: str, dir: str = "."):
        """ Create a GroundedFunctionNetwork instance from a string with raw
        Fortran code.

        Args:
            fortran_src: A string with Fortran source code.
            dir: (Optional) - the directory in which the temporary Fortran file
                will be created (make sure you have write permission!) Defaults
                to the current directory.
        Returns:
            A GroundedFunctionNetwork instance
        """
        import tempfile

        fp = tempfile.NamedTemporaryFile("w+t", delete=False, dir=dir)
        fp.writelines(fortran_src)
        fp.close()
        G = cls.from_fortran_file(fp.name, dir)
        os.remove(fp.name)
        return G

    def to_json(self):
        """Experimental outputting a GrFN to a JSON file."""
        containers = {
            name: {"name": name, "parent": None, "exit": True, "nodes": list()}
            for name in self.scope_tree.nodes
        }

        nodes_json = list()
        for name, data in self.nodes(data=True):
            containers[data["parent"]]["nodes"].append(name)
            if data["type"] == "variable":
                nodes_json.append(
                    {
                        "name": name,
                        "type": "variable",
                        "reference": None,
                        "data-type": {
                            "name": "float32",
                            "domain": [("-inf", "inf")],
                        },
                    }
                )
            elif data["type"] == "function":
                (source_list, _) = inspect.getsourcelines(data["lambda_fn"])
                source_code = "".join(source_list)
                nodes_json.append(
                    {
                        "name": name,
                        "type": "function",
                        "reference": None,
                        "inputs": data["func_inputs"],
                        "lambda": source_code,
                    }
                )
            else:
                raise ValueError(f"Unrecognized node type: {data['type']}")

        return json.dumps(
            {
                "nodes": nodes_json,
                "edges": list(self.edges),
                "containers": list(containers.values()),
            }
        )

    def to_json_file(self, filename):
        GrFN_json = self.to_json()
        json.dump(GrFN_json, open(filename, "w"))

    def to_AGraph(self):
        """ Export to a PyGraphviz AGraph object. """
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update(
            {"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "LR"}
        )
        A.node_attr.update({"fontname": "Menlo"})

        def build_tree(cluster_name, node_attrs, root_graph):
            subgraph_nodes = [
                node_name
                for node_name, node_data in self.nodes(data=True)
                if node_data["parent"] == cluster_name
            ]
            root_graph.add_nodes_from(subgraph_nodes)
            subgraph = root_graph.add_subgraph(
                subgraph_nodes,
                name=f"cluster_{cluster_name}",
                label=cluster_name,
                style="bold, rounded",
                rankdir="LR",
                color=node_attrs[cluster_name]["color"],
            )
            for n in self.scope_tree.successors(cluster_name):
                build_tree(n, node_attrs, subgraph)

        root = [n for n, d in self.scope_tree.in_degree() if d == 0][0]
        node_data = {n: d for n, d in self.scope_tree.nodes(data=True)}
        build_tree(root, node_data, A)
        return A

    def CAG_to_AGraph(self):
        """Returns a variable-only view of the GrFN in the form of an AGraph.

        Returns:
            type: A CAG constructed via variable influence in the GrFN object.

        """
        CAG = self.to_CAG()
        for name, data in CAG.nodes(data=True):
            CAG.nodes[name]["label"] = data["cag_label"]
        A = nx.nx_agraph.to_agraph(CAG)
        A.graph_attr.update(
            {"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "LR"}
        )
        A.node_attr.update(
            {
                "shape": "rectangle",
                "color": "#650021",
                "style": "rounded",
                "fontname": "Menlo",
            }
        )
        A.edge_attr.update({"color": "#650021", "arrowsize": 0.5})
        return A

    def FCG_to_AGraph(self):
        """ Build a PyGraphviz AGraph object corresponding to a call graph of
        functions. """

        A = nx.nx_agraph.to_agraph(self.FCG)
        A.graph_attr.update(
            {"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "TB"}
        )
        A.node_attr.update(
            {"shape": "rectangle", "color": "#650021", "style": "rounded"}
        )
        A.edge_attr.update({"color": "#650021", "arrowsize": 0.5})
        return A


class ForwardInfluenceBlanket(ComputationalGraph):
    """
    This class takes a network and a list of a shared nodes between the input
    network and a secondary network. From this list a shared nodes and blanket
    network is created including all of the nodes between any input/output pair
    in the shared nodes, as well as all nodes required to blanket the network
    for forward influence. This class itself becomes the blanket and inherits
    from the ComputationalGraph class.
    """

    def __init__(self, G: GroundedFunctionNetwork, shared_nodes: Set[str]):
        # super().__init__()
        outputs = G.outputs
        inputs = set(G.inputs).intersection(shared_nodes)

        # Get all paths from shared inputs to shared outputs
        path_inputs = shared_nodes - set(outputs)
        io_pairs = [(inp, G.output_node) for inp in path_inputs]
        paths = [p for (i, o) in io_pairs for p in all_simple_paths(G, i, o)]

        # Get all edges needed to blanket the included nodes
        main_nodes = {node for path in paths for node in path}
        main_edges = {
            (n1, n2) for path in paths for n1, n2 in zip(path, path[1:])
        }
        blanket_nodes = set()
        add_nodes, add_edges = list(), list()

        def place_var_node(var_node):
            prev_funcs = list(G.predecessors(var_node))
            if len(prev_funcs) > 0 and G.nodes[prev_funcs[0]]["label"] == "L":
                prev_func = prev_funcs[0]
                add_nodes.extend([var_node, prev_func])
                add_edges.append((prev_func, var_node))
            else:
                blanket_nodes.add(var_node)

        for node in main_nodes:
            if G.nodes[node]["type"] == "function":
                for var_node in G.predecessors(node):
                    if var_node not in main_nodes:
                        add_edges.append((var_node, node))
                        if "::IF_" in var_node:
                            if_func = list(G.predecessors(var_node))[0]
                            add_nodes.extend([if_func, var_node])
                            add_edges.append((if_func, var_node))
                            for new_var_node in G.predecessors(if_func):
                                add_edges.append((new_var_node, if_func))
                                place_var_node(new_var_node)
                        else:
                            place_var_node(var_node)

        main_nodes |= set(add_nodes)
        main_edges |= set(add_edges)
        main_nodes = main_nodes - inputs - set(outputs)

        orig_nodes = G.nodes(data=True)

        F = nx.DiGraph()

        F.add_nodes_from([(n, d) for n, d in orig_nodes if n in inputs])
        for node in inputs:
            F.nodes[node]["color"] = dodgerblue3
            F.nodes[node]["fontcolor"] = dodgerblue3
            F.nodes[node]["penwidth"] = 3.0
            F.nodes[node]["fontname"] = FONT

        F.inputs = list(F.inputs)

        F.add_nodes_from([(n, d) for n, d in orig_nodes if n in blanket_nodes])
        for node in blanket_nodes:
            F.nodes[node]["fontname"] = FONT
            F.nodes[node]["color"] = forestgreen
            F.nodes[node]["fontcolor"] = forestgreen

        F.add_nodes_from([(n, d) for n, d in orig_nodes if n in main_nodes])
        for node in main_nodes:
            F.nodes[node]["fontname"] = FONT

        for out_var_node in outputs:
            F.add_node(out_var_node, **G.nodes[out_var_node])
            F.nodes[out_var_node]["color"] = dodgerblue3
            F.nodes[out_var_node]["fontcolor"] = dodgerblue3

        F.add_edges_from(main_edges)
        super().__init__(F, outputs)

        # self.FCG = self.to_FCG()
        # self.function_sets = self.build_function_sets()

    @classmethod
    def from_GrFN(cls, G1, G2):
        """ Creates a ForwardInfluenceBlanket object representing the
        intersection of this model with the other input model.

        Args:
            G1: The GrFN model to use as the basis for this FIB
            G2: The GroundedFunctionNetwork object to compare this model to.

        Returns:
            A ForwardInfluenceBlanket object to use for model comparison.
        """

        if not (
            isinstance(G1, GroundedFunctionNetwork)
            and isinstance(G2, GroundedFunctionNetwork)
        ):
            raise TypeError(
                f"Expected two GrFNs, but got ({type(G1)}, {type(G2)})"
            )

        def shortname(var):
            return var[var.find("::") + 2 : var.rfind("_")]

        def shortname_vars(graph, shortname):
            return [v for v in graph.nodes() if shortname in v]

        g1_var_nodes = {
            shortname(n)
            for (n, d) in G1.nodes(data=True)
            if d["type"] == "variable"
        }
        g2_var_nodes = {
            shortname(n)
            for (n, d) in G2.nodes(data=True)
            if d["type"] == "variable"
        }

        shared_vars = {
            full_var
            for shared_var in g1_var_nodes.intersection(g2_var_nodes)
            for full_var in shortname_vars(G1, shared_var)
        }

        return cls(G1, shared_vars)

    def run(
        self,
        inputs: Dict[str, Union[float, Iterable]],
        covers: Dict[str, Union[float, Iterable]],
    ) -> Union[float, Iterable]:
        """Executes the FIB over a particular set of inputs and returns the
        result.
        Args:
            inputs: Input set where keys are the names of input nodes in the
              GrFN and each key points to a set of input values (or just one).
        Returns:
            A set of outputs from executing the GrFN, one for every set of
            inputs.
        """
        # Abort run if covers does not match our expected cover set
        if len(covers) != len(self.blanket_nodes):
            raise ValueError("Incorrect number of cover values.")

        # Set the cover node values
        for node_name, val in covers.items():
            self.nodes[node_name]["value"] = val

        return super().run(inputs)

    def to_AGraph(self):
        A = nx.nx_agraph.to_AGraph(self)
        A.graph_attr.update({"dpi": 227, "fontsize": 20})
        A.node_attr.update({"shape": "rectangle", "style": "rounded"})
        A.edge_attr.update({"arrowsize": 0.5})
        return A


@dataclass(repr=False, frozen=False)
class GenericNode(ABC):
    uid: UUID

    def __hash__(self):
        return hash(self.uid)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        return self.uid

    @staticmethod
    def create_node_id() -> UUID:
        return uuid4()

    @abstractmethod
    def get_kwargs(self):
        return NotImplemented


@dataclass(repr=False, frozen=False)
class VariableNode(GenericNode):
    identifier: VariableIdentifier
    value: Any

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name

    @classmethod
    def from_id(cls, id: VariableIdentifier):
        return cls(GenericNode.create_node_id(), id, None)

    def get_fullname(self):
        return f"{self.name}\n({self.index})"

    def get_kwargs(self):
        is_exit = self.name == "EXIT"
        return {
            "color": "crimson",
            "fontcolor": "white" if is_exit else "black",
            "fillcolor": "crimson" if is_exit else "white",
            "style": "filled" if is_exit else "",
            "padding": 15,
            "value": None,
        }


@dataclass(repr=False, frozen=False)
class LambdaNode(GenericNode):
    func_type: LambdaType
    func_str: str
    function: callable

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.func_type.shortname()

    def __call__(self, *values) -> Iterable[Any]:
        expected_num_args = len(self.get_signature())
        input_num_args = len(values)
        if expected_num_args != input_num_args:
            raise RuntimeError(
                f"""Incorrect number of inputs
                (expected {expected_num_args} found {input_num_args})
                for lambda:\n{self.func_str}"""
            )
        try:
            res = self.function(*values)
            if not isinstance(res, Iterable):
                res = list(res)
            return res
        except Exception as e:
            print(f"Exception occured in {self.func_str}")
            raise e

    def get_kwargs(self):
        return {"shape": "rectangle", "padding": 10}

    def get_signature(self):
        return self.function.__code__.co_varnames


@dataclass
class HyperEdge:
    inputs: Iterable[VariableNode]
    lambda_fn: LambdaNode
    outputs: Iterable[VariableNode]

    def __call__(self):
        result = self.lambda_fn(*[var.value for var in self.inputs])
        for i, out_val in enumerate(result):
            self.outputs[i].value = out_val


@dataclass(repr=False)
class GrFNSubgraph:
    namespace: str
    scope: str
    basename: str
    occurrence_num: int
    border_color: str
    nodes: Iterable[GenericNode]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.basename}\n({self.occurrence_num})"

    @classmethod
    def from_container(cls, con: GenericContainer, occ: int):
        if isinstance(con, CondContainer):
            clr = "orange"
        elif isinstance(con, FuncContainer):
            clr = "forestgreen"
        elif isinstance(con, LoopContainer):
            clr = "navyblue"
        else:
            # TODO: perhaps use this in the future
            # clr = "lightskyblue"
            raise TypeError(f"Unrecognized container type: {type(con)}")
        id = con.identifier
        return cls(id.namespace, id.scope, id.basename, occ, clr, [])


class GroundedFactorNetwork(nx.DiGraph):
    def __init__(
        self,
        id: ContainerIdentifier,
        G: nx.DiGraph,
        H: List[HyperEdge],
        S: List[GrFNSubgraph],
    ):
        super().__init__(G)
        self.hyper_edges = H
        self.subgraphs = S

        self.namespace = id.namespace
        self.scope = id.scope
        self.name = id.con_name
        self.label = f"{self.name} ({self.namespace}.{self.scope})"

        self.variables = [n for n in self.nodes if isinstance(n, VariableNode)]
        self.lambdas = [n for n in self.nodes if isinstance(n, LambdaNode)]
        self.inputs = [
            n
            for n in self.nodes
            if len(list(self.predecessors(n))) == 0
            and isinstance(n, VariableNode)
        ]
        self.outputs = [
            n
            for n in self.nodes
            if len(list(self.successors(n))) == 0
            and isinstance(n, VariableNode)
        ]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        L_sz = str(len(self.lambda_nodes))
        V_sz = str(len(self.variable_nodes))
        I_sz = str(len(self.input_variables))
        O_sz = str(len(self.output_variables))
        size_str = f"< |L|: {L_sz}, |V|: {V_sz}, |I|: {I_sz}, |O|: {O_sz} >"
        return f"{self.label}\n{size_str}"

    def __call__(self, inputs: Dict[str, Any]) -> Iterable[Any]:
        """Executes the GrFN over a particular set of inputs and returns the
        result.

        Args:
            inputs: Input set where keys are the names of input nodes in the
                GrFN and each key points to a set of input values (or just one)

        Returns:
            A set of outputs from executing the GrFN, one for every set of
            inputs.
        """
        # TODO: update this function to work with new GrFN object
        full_inputs = {self.input_name_map[n]: v for n, v in inputs.items()}
        # Set input values
        for i in self.inputs:
            value = full_inputs[i]
            if isinstance(value, float):
                value = np.array([value], dtype=np.float32)
            if isinstance(value, int):
                value = np.array([value], dtype=np.int32)
            elif isinstance(value, list):
                value = np.array(value, dtype=np.float32)

            self.nodes[i]["value"] = value
        for func_set in self.function_sets:
            for func_name in func_set:
                signature = self.nodes[func_name]["func_inputs"]
                input_values = [self.nodes[n]["value"] for n in signature]
                lambda_fn = self.nodes[func_name]["lambda_fn"]
                res = lambda_fn(*input_values)

                # Convert output to a NumPy matrix if a constant was returned
                if len(input_values) == 0:
                    res = np.array(res, dtype=np.float32)

                output_node = list(self.successors(func_name))[0]
                self.nodes[output_node]["value"] = res

        # Return the output
        return [self.nodes[o]["value"] for o in self.outputs]

    @classmethod
    def from_AIR(
        cls,
        con_id: ContainerIdentifier,
        containers: Dict[ContainerIdentifier, GenericContainer],
        variables: Dict[VariableIdentifier, VariableDefinition],
        types: Dict[TypeIdentifier, TypeDefinition],
    ):
        network = nx.DiGraph()
        hyper_edges = list()
        Occs = dict()
        subgraphs = list()

        def add_variable_node(id: VariableIdentifier) -> VariableNode:
            node = VariableNode.from_id(id)
            network.add_node(node, **(node.get_kwargs()))
            return node

        def add_lambda_node(
            lambda_type: LambdaType, lambda_str: str
        ) -> LambdaNode:
            lambda_fn = eval(lambda_str)
            node_id = GenericNode.create_node_id()
            node = LambdaNode(node_id, lambda_type, lambda_str, lambda_fn)
            network.add_node(node, **(node.get_kwargs()))
            return node

        def add_hyper_edge(
            inputs: Iterable[VariableNode],
            lambda_node: LambdaNode,
            outputs: Iterable[VariableNode],
        ) -> None:
            network.add_edges_from(
                [(in_node, lambda_node) for in_node in inputs]
            )
            network.add_edges_from(
                [(lambda_node, out_node) for out_node in outputs]
            )
            hyper_edges.append(HyperEdge(inputs, lambda_node, outputs))

        def translate_container(
            con: GenericContainer, inputs: Iterable[VariableNode],
        ) -> Iterable[VariableNode]:
            # raise ValueError(f"Unsupported container type: {type(con)}")
            if con.name not in Occs:
                Occs[con.name] = 0

            con_subgraph = GrFNSubgraph(con.name, Occs[con.name])
            live_variables = dict()
            if len(inputs) > 0:
                in_var_names = [n.name for n in inputs]
                in_var_str = ",".join(in_var_names)
                pass_func_str = f"lambda {in_var_str}:({in_var_str})"
                func = add_lambda_node(LambdaType.PASS, pass_func_str)
                out_nodes = [add_variable_node(id) for id in con.arguments]
                add_hyper_edge(inputs, func, out_nodes)
                con_subgraph.nodes.append(func)

                live_variables.update(
                    {id: node for id, node in zip(con.arguments, out_nodes)}
                )
            else:
                live_variables.update(
                    {id: add_variable_node(id) for id in con.arguments}
                )

            for stmt in con.statements:
                returned_nodes = translate_stmt(stmt, inputs, live_variables)
                con_subgraph.nodes.extend(returned_nodes)

            subgraphs.nodes.extend(list(live_variables.values()))
            subgraphs.append(con_subgraph)

            if len(inputs) > 0:
                # Do this only if this is not the starting container
                returned_vars = [live_variables[id] for id in con.returns]
                update_vars = [live_variables[id] for id in con.updated]
                output_vars = returned_vars + update_vars

                out_var_names = [n.name for n in output_vars]
                out_var_str = ",".join(out_var_names)
                pass_func_str = f"lambda {out_var_str}:({out_var_str})"
                func = network.add_lambda_node(LambdaType.PASS, pass_func_str)
                return (output_vars, func)

        @singledispatch
        def translate_stmt(
            stmt: GenericStmt,
            live_variables: Dict[VariableIdentifier, VariableNode],
        ) -> Iterable[GenericNode]:
            raise ValueError(f"Unsupported statement type: {type(stmt)}")

        @translate_stmt.register
        def _(
            stmt: CallStmt,
            live_variables: Dict[VariableIdentifier, VariableNode],
        ) -> Iterable[GenericNode]:
            new_con = containers[stmt.call_id]
            if stmt.call_id not in Occs:
                Occs[stmt.call_id] = 0

            inputs = [live_variables[id] for id in stmt.inputs]
            (con_outputs, pass_func) = translate_container(new_con, inputs)
            out_nodes = [add_variable_node(var) for var in stmt.outputs]
            add_hyper_edge(con_outputs, pass_func, out_nodes)
            for output_node in out_nodes:
                var_id = output_node.identifier
                live_variables[var_id] = output_node

            Occs[stmt.call_id] += 1
            return out_nodes

        @translate_stmt.register
        def _(
            stmt: LambdaStmt,
            live_variables: Dict[VariableIdentifier, VariableNode],
        ) -> Iterable[GenericNode]:
            inputs = [live_variables[id] for id in stmt.inputs]
            out_nodes = [add_variable_node(var) for var in stmt.outputs]
            func = add_lambda_node(stmt.type, stmt.func_str)
            add_hyper_edge(inputs, func, out_nodes)
            for output_node in out_nodes:
                var_id = output_node.identifier
                live_variables[var_id] = output_node

            return [func] + out_nodes

        start_container = containers[con_id]
        Occs[con_id] = 0
        translate_container(start_container, [])
        return cls(network, hyper_edges, subgraphs)
