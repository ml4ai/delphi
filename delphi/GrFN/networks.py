from abc import ABCMeta
from pathlib import Path
from typing import Dict, Iterable, Set, Union
from typing import List
import importlib
import inspect
import json
import os
import itertools

from SALib.analyze import sobol, fast, rbd_fast
from SALib.sample import saltelli, fast_sampler, latin
import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths

from delphi.translators.for2py.types_ext import Float32
# from delphi.GrFN.analysis import get_max_s2_sensitivity
import delphi.GrFN.utils as utils
from delphi.GrFN.utils import ScopeNode
from delphi.utils.misc import choose_font
from delphi.translators.for2py import (
    genPGM,
    f2grfn,
)
import numpy as np

FONT = choose_font()

dodgerblue3 = "#1874CD"
forestgreen = "#228b22"


class ComputationalGraph(nx.DiGraph):
    __metaclass__ = ABCMeta

    def __init__(self, network, output_vars, *args, **kwargs):
        super().__init__(network, *args, **kwargs)
        self.outputs = output_vars
        self.inputs = [
            n
            for n, d in self.in_degree()
            if d == 0 and self.nodes[n]["type"] == "variable"
        ]
        self.input_name_map = {
            self.var_shortname(name): name for name in self.inputs
        }
        self.FCG = self.to_FCG()
        self.function_sets = self.build_function_sets()

    @staticmethod
    def var_shortname(long_var_name):
        (module,
         var_scope,
         container_name,
         container_index,
         var_name,
         var_index) = long_var_name.split("::")
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
        self,
        inputs: Dict[str, Union[float, Iterable]],
    ) -> Union[float, Iterable]:
        """Executes the GrFN over a particular set of inputs and returns the
        result.

        Args:
            inputs: Input set where keys are the names of input nodes in the
                GrFN and each key points to a set of input values (or just one).

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
        # for o in self.outputs:
            # print(self.nodes[o]["value"])
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
        super().__init__(G, outputs)
        # self.outputs = outputs
        self.scope_tree = scope_tree
        # self.inputs = [
        #     n
        #     for n, d in self.in_degree()
        #     if d == 0 and self.nodes[n]["type"] == "variable"
        # ]

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

        Returns:
            type: A GroundedFunctionNetwork object.

        """
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data, lambdas)

    @classmethod
    def from_dict(cls, data: Dict, lambdas):
        """Builds a GrFN object from a set of extracted function data objects
        and an associated file of lambda functions.

        Args:
            cls: The class variable for object creation.
            data: A set of function data object that specify the wiring of a
                  GrFN object.
            lambdas: [Module] A python module containing actual python
                     functions to be computed during GrFN execution.

        Returns:
            A GroundedFunctionNetwork object.

        """
        functions = {d["name"]: d for d in data["containers"]}
        occurrences = {}
        G = nx.DiGraph()
        scope_tree = nx.DiGraph()

        def identity(x):
            return x

        def make_identifier(scope: str, var: str):
            (_, name, idx) = var.split("::")
            return make_variable_name(scope, name, idx)

        def make_variable_name(parent: str, basename: str, index: str):
            return f"{parent}::{basename}::{index}"

        def add_variable_node(parent: str, basename: str, index: str, is_exit: bool = False):
            full_var_name = make_variable_name(parent, basename, index)
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

        def process_wiring_statement(stmt, scope, inputs, cname):
            lambda_name = stmt["function"]["name"]
            lambda_node_name = f"{scope.name}::" + lambda_name

            stmt_type = lambda_name.split("__")[-3]
            if stmt_type == "assign" and len(stmt["input"]) == 0:
                stmt_type = "literal"

            for output in stmt["output"]:
                (_, var_name, idx) = output.split("::")
                node_name = add_variable_node(scope.name, var_name, idx, is_exit=var_name == "EXIT")
                G.add_edge(lambda_node_name, node_name)

            ordered_inputs = list()
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

            new_container = functions[container_name]
            container_color = "navyblue" if new_container["repeat"] else "forestgreen"
            new_scope = ScopeNode(new_container, occurrences[container_name], parent=scope)
            scope_tree.add_node(new_scope.name, color=container_color)
            scope_tree.add_edge(scope.name, new_scope.name)

            input_values = list()
            for inp in stmt["input"]:
                if inp.endswith("-1"):
                    (parent, var_name, idx) = inputs[inp]
                else:
                    parent = scope.name
                    (_, var_name, idx) = inp.split("::")
                input_values.append((parent, var_name, idx))

            callee_ret, callee_up = process_container(new_scope, input_values, container_name)

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

        def process_container(scope, input_vals, cname):
            if len(scope.arguments) == len(input_vals):
                input_vars = {a: v for a, v in zip(scope.arguments, input_vals)}
            elif len(scope.arguments) > 0:
                input_vars = {a: (scope.name,) + tuple(a.split("::")[1:]) for a in scope.arguments}

            for stmt in scope.body:
                func_def = stmt["function"]
                func_type = func_def["type"]
                if func_type == "lambda":
                    process_wiring_statement(stmt, scope, input_vars, cname)
                elif func_type == "container":
                    process_call_statement(stmt, scope, input_vars, cname)
                else:
                    raise ValueError(f"Undefined function type: {func_type}")

            return_list, updated_list = list(), list()
            for var_name in scope.returns:
                (_, basename, idx) = var_name.split("::")
                return_list.append(make_variable_name(scope.name, basename, idx))

            for var_name in scope.updated:
                (_, basename, idx) = var_name.split("::")
                updated_list.append(make_variable_name(scope.name, basename, idx))
            return return_list, updated_list

        root = data["start"][0]
        occurrences[root] = 0
        cur_scope = ScopeNode(functions[root], occurrences[root])
        scope_tree.add_node(cur_scope.name, color="forestgreen")
        returns, updates = process_container(cur_scope, [], root)
        return cls(G, scope_tree, returns+updates)

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
        lambdas_path,
        json_filename: str,
        stem: str,
        python_file: str,
        fortran_file: str,
        module_log_file_path: str,
        mode_mapper_dict: list,
        processing_modules: bool,
        generated_files_list: list,
        save_file: bool = False,
    ):
        module_file_exist = False
        module_import_paths = {}

        """Builds GrFN object from Python source code."""
        pgm_dict = f2grfn.generate_grfn(
            pySrc,
            python_file,
            lambdas_path,
            mode_mapper_dict,
            fortran_file,
            module_log_file_path,
            processing_modules,
        )
        lambdas = importlib.__import__(stem + "_lambdas")
        """Add generated GrFN and lambdas file paths to the list"""
        return cls.from_dict(pgm_dict, lambdas)

    @classmethod
    def from_fortran_file(cls, fortran_file: str, tmpdir: str = ".", save_file: bool = False):
        """Builds GrFN object from a Fortran program."""

        lambda_file_suffix = "_lambdas.py"

        if tmpdir == "." and "/" in fortran_file:
            tmpdir_path = Path(fortran_file).parent
        root_dir = os.path.abspath(tmpdir)

        (
            pySrc,
            json_filename,
            translated_python_files,
            stem,
            mode_mapper_dict,
            fortran_filename,
            module_log_file_path,
            processing_modules,
        ) = f2grfn.fortran_to_grfn(
            fortran_file,
            temp_dir=str(tmpdir),
            root_dir_path=root_dir,
            processing_modules=False,
        )

        generated_files = []
        for python_file in translated_python_files:
            lambdas_path = python_file[0:-3] + lambda_file_suffix
            G = cls.from_python_src(
                pySrc[0][0],
                lambdas_path,
                json_filename,
                stem,
                python_file,
                fortran_file,
                module_log_file_path,
                mode_mapper_dict,
                processing_modules,
                generated_files,
                save_file=save_file
            )

            """Return GrFN object"""
            return G

    @classmethod
    def from_fortran_src(cls, fortran_src: str, dir: str = "."):
        """ Create a GroundedFunctionNetwork instance from a string with raw
        Fortran code.

        Args:
            fortran_src: A string with Fortran source code.
            dir: (Optional) - the directory in which the temporary Fortran file
                will be created (make sure you have write permission!) Defaults to
                the current directory.
        Returns:
            A GroundedFunctionNetwork instance
        """
        import tempfile
        fp = tempfile.NamedTemporaryFile('w+t', delete=False, dir=dir)
        fp.writelines(fortran_src)
        fp.close()
        G = cls.from_fortran_file(fp.name, dir)
        os.remove(fp.name)
        return G

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
                color=node_attrs[cluster_name]["color"]
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
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "LR"})
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
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "TB"})
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
        paths = [
            p for (i, o) in io_pairs for p in all_simple_paths(G, i, o)
        ]

        # Get all edges needed to blanket the included nodes
        main_nodes = {node for path in paths for node in path}
        main_edges = {
            (n1, n2) for path in paths for n1, n2 in zip(path, path[1:])
        }
        blanket_nodes = set()
        add_nodes, add_edges = list(), list()

        def place_var_node(var_node):
            prev_funcs = list(G.predecessors(var_node))
            if (
                len(prev_funcs) > 0
                and G.nodes[prev_funcs[0]]["label"] == "L"
            ):
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

        F.add_nodes_from([(n, d) for n, d in orig_nodes
                            if n in blanket_nodes])
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

        if not (isinstance(G1, GroundedFunctionNetwork)
                and isinstance(G2, GroundedFunctionNetwork)):
            raise TypeError(
                f"Expected two GrFNs, but got ({type(G1)}, {type(G2)})"
            )

        def shortname(var):
            return var[var.find("::") + 2: var.rfind("_")]

        def shortname_vars(graph, shortname):
            return [v for v in graph.nodes() if shortname in v]

        g1_var_nodes = {
            shortname(n)
            for (n, d) in G1.nodes(data=True) if d["type"] == "variable"
        }
        g2_var_nodes = {
            shortname(n)
            for (n, d) in G2.nodes(data=True) if d["type"] == "variable"
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
        covers: Dict[str, Union[float, Iterable]]
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
        if len(covers) != len(blanket_nodes):
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
