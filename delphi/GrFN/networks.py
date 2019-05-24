import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Union, Set, Optional
import subprocess as sp
import importlib
import inspect
import json
import os
import ast
import itertools

import torch
from SALib.analyze import sobol, fast, rbd_fast
from SALib.sample import saltelli, fast_sampler, latin
import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths

from delphi.GrFN.analysis import get_max_s2_sensitivity
import delphi.GrFN.utils as utils
from delphi.GrFN.utils import ScopeNode
from delphi.utils.misc import choose_font
from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    rectify,
)
import numpy as np
import torch

FONT = choose_font()

dodgerblue3 = "#1874CD"
forestgreen = "#228b22"


class ComputationalGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = [
            n
            for n, d in self.out_degree()
            if d == 0 and self.nodes[n]["type"] == "variable"
        ]

    def build_call_graph(self):
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
        # TODO - this fails when there is only one function node in the graph -
        # need to fix!
        initial_funcs = [n for n, d in self.call_graph.in_degree() if d == 0]
        distances = dict()

        def find_distances(funcs, dist):
            all_successors = list()
            for func in funcs:
                distances[func] = dist
                all_successors.extend(self.call_graph.successors(func))
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
        torch_size: Optional[int] = None,
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
        # Set input values
        for i in self.inputs:
            self.nodes[i]["value"] = inputs[i]

        for func_set in self.function_sets:
            for func_name in func_set:
                lambda_fn = self.nodes[func_name]["lambda_fn"]
                output_node = list(self.successors(func_name))[0]

                signature = self.nodes[func_name]["func_inputs"]
                input_values = [self.nodes[n]["value"] for n in signature]
                res = lambda_fn(*input_values)

                if torch_size is not None and len(signature) == 0:
                    self.nodes[output_node]["value"] = torch.tensor(
                        [res] * torch_size, dtype=torch.double
                    )
                else:
                    self.nodes[output_node]["value"] = res

        # Return the output
        return self.nodes[self.output_node]["value"]

    def to_CAG(self):
        """ Export to a Causal Analysis Graph (CAG) PyGraphviz AGraph object.
        The CAG shows the influence relationships between the variables and
        elides the function nodes."""

        G = nx.DiGraph()
        for (name, attrs) in self.nodes(data=True):
            if attrs["type"] == "variable":
                for pred_fn in self.predecessors(name):
                    if not any(
                        fn_type in pred_fn
                        for fn_type in ("condition", "decision")
                    ):
                        for pred_var in self.predecessors(pred_fn):
                            G.add_node(
                                self.nodes[pred_var]["basename"],
                                **self.nodes[pred_var],
                            )
                            G.add_node(attrs["basename"], **attrs)
                            G.add_edge(
                                self.nodes[pred_var]["basename"],
                                attrs["basename"],
                            )
                if attrs["is_loop_index"]:
                    G.add_edge(attrs["basename"], attrs["basename"])

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

    def __init__(self, G, scope_tree):
        super().__init__(G)
        self.scope_tree = scope_tree
        self.inputs = [
            n
            for n, d in self.in_degree()
            if d == 0 and self.nodes[n]["type"] == "variable"
        ]
        self.output_node = self.outputs[-1]
        self.call_graph = self.build_call_graph()
        self.function_sets = self.build_function_sets()

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
        functions = {d["name"]: d for d in data["functions"]}
        G = nx.DiGraph()
        scope_tree = nx.DiGraph()

        def add_variable_node(
            basename: str, parent: str, index: int, is_loop_index: bool = False
        ):
            G.add_node(
                f"{parent}::{basename}_{index}",
                type="variable",
                color="maroon",
                parent=parent,
                label=f"{basename}_{index}",
                basename=basename,
                is_loop_index=is_loop_index,
                padding=15,
                value=None,
            )

        def find_correct_scope(cur_scope, var_name):
            if cur_scope.parent is None:
                return cur_scope.name, -1

            new_scope = cur_scope.parent

            if var_name in new_scope.variables:
                return new_scope.name, new_scope.variables[var_name]
            else:
                return find_correct_scope(new_scope, var_name)

        def process_container(
            scope, loop_index_variable: Optional[str] = None
        ):
            for stmt in scope.body:
                if "name" in stmt:
                    stmt_type = functions[stmt["name"]]["type"]
                    if stmt_type in ("assign", "condition", "decision"):
                        if stmt_type == "assign":
                            if "body" in functions[stmt["name"]]:
                                stmt_type = "literal"

                        output = stmt["output"]
                        scope.variables[output["variable"]] = output["index"]
                        add_variable_node(
                            output["variable"], scope.name, output["index"]
                        )
                        G.add_edge(
                            stmt["name"],
                            f"{scope.name}::{output['variable']}_{output['index']}",
                        )

                        ordered_inputs = list()
                        for input in stmt.get("input", []):
                            parent = scope.name
                            index = input["index"]
                            base_name = input["variable"]
                            if (
                                index == -1
                                and base_name != loop_index_variable
                            ):
                                parent, index = find_correct_scope(
                                    scope, base_name
                                )

                            add_variable_node(
                                base_name,
                                parent,
                                index,
                                base_name == loop_index_variable,
                            )
                            node_name = f"{parent}::{base_name}_{index}"
                            ordered_inputs.append(node_name)
                            G.add_edge(node_name, stmt["name"])

                        G.add_node(
                            stmt["name"],
                            type="function",
                            lambda_fn=getattr(lambdas, stmt["name"]),
                            func_inputs=ordered_inputs,
                            shape="rectangle",
                            parent=scope.name,
                            label=stmt_type[0].upper(),
                            padding=10,
                        )

                    elif stmt_type == "loop_plate":
                        index_variable = functions[stmt["name"]][
                            "index_variable"
                        ]
                        scope_tree.add_node(stmt["name"], color="blue")
                        scope_tree.add_edge(scope.name, stmt["name"])
                        new_scope = ScopeNode(
                            functions[stmt["name"]], parent=scope
                        )
                        process_container(
                            new_scope, loop_index_variable=index_variable
                        )
                    else:
                        pass
                elif "function" in stmt and stmt["function"] != "print":
                    scope_tree.add_node(stmt["function"], color="green")
                    scope_tree.add_edge(scope.name, stmt["function"])
                    new_scope = ScopeNode(
                        functions[stmt["function"]], parent=scope
                    )
                    process_container(new_scope)

        root = data["start"]
        scope_tree.add_node(root, color="green")
        cur_scope = ScopeNode(functions[root])
        process_container(cur_scope)
        return cls(G, scope_tree)

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
        save_file: bool = False,
    ):
        """Builds GrFN object from Python source code."""
        asts = [ast.parse(pySrc)]
        pgm_dict = genPGM.create_pgm_dict(
            lambdas_path,
            asts,
            json_filename,
            {"FileName": f"{stem}.py"},  # HACK
        )
        lambdas = importlib.__import__(stem + "_lambdas")
        return cls.from_dict(pgm_dict, lambdas)



    @classmethod
    def from_fortran_file(cls, fortran_file: str, tmpdir: str = "."):
        """Builds GrFN object from a Fortran program."""
        stem = Path(fortran_file).stem
        if tmpdir == "." and "/" in fortran_file:
            tmpdir = Path(fortran_file).parent
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
        tree = rectify.buildNewASTfromXMLString(xml_string)
        trees = [tree]
        comments = get_comments.get_comments(preprocessed_fortran_file)
        os.remove(preprocessed_fortran_file)
        xml_to_json_translator = translate.XMLToJSONTranslator()
        outputDict = xml_to_json_translator.analyze(trees, comments)
        pySrc = pyTranslate.create_python_source_list(outputDict)[0][0]

        G = cls.from_python_src(pySrc, lambdas_path, json_filename, stem)
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

    def clear(self):
        """Clear variable nodes for next computation."""
        for n in self.nodes():
            if self.nodes[n]["type"] == "variable":
                self.nodes[n]["value"] = None
            elif self.nodes[n]["type"] == "function":
                self.nodes[n]["func_visited"] = False

    def sobol_analysis(
        self, num_samples, prob_def, use_torch=False, var_types=None
    ):
        def create_input_tensor(name, samples):
            type_info = var_types[name]
            if type_info[0] == str:
                (val1, val2) = type_info[1]
                return np.where(samples >= 0.5, val1, val2)
            else:
                return torch.tensor(samples)

        samples = saltelli.sample(
            prob_def, num_samples, calc_second_order=True
        )
        if use_torch:
            samples = np.split(samples, samples.shape[1], axis=1)
            samples = [s.squeeze() for s in samples]
            if var_types is None:
                values = {
                    n: torch.tensor(s)
                    for n, s in zip(prob_def["names"], samples)
                }
            else:
                values = {
                    n: create_input_tensor(n, s)
                    for n, s in zip(prob_def["names"], samples)
                }
            Y = self.run(values, torch_size=len(samples[0])).numpy()
        else:
            Y = np.zeros(samples.shape[0])
            for i, sample in enumerate(samples):
                values = {n: val for n, val in zip(prob_def["names"], sample)}
                Y[i] = self.run(values)

        return sobol.analyze(prob_def, Y)

    def S2_surface(self, sizes, bounds, presets, use_torch=False,
            num_samples=10):
        """Calculates the sensitivity surface of a GrFN for the two variables with
        the highest S2 index.

        Args:
            num_samples: Number of samples for sensitivity analysis.
            sizes: Tuple of (number of x inputs, number of y inputs).
            bounds: Set of bounds for GrFN inputs.
            presets: Set of standard values for GrFN inputs.

        Returns:
            Tuple:
                Tuple: The names of the two variables that were selected
                Tuple: The X, Y vectors of eval values
                Z: The numpy matrix of output evaluations

        """
        args = self.inputs
        Si = self.sobol_analysis(
            num_samples,
            {
                "num_vars": len(args),
                "names": args,
                "bounds": [bounds[arg] for arg in args],
            },
        )
        S2 = Si["S2"]
        (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

        x_var = args[v1]
        y_var = args[v2]
        search_space = [(x_var, bounds[x_var]), (y_var, bounds[y_var])]
        preset_vals = {
            arg: presets[arg]
            for i, arg in enumerate(args)
            if i != v1 and i != v2
        }

        X = np.linspace(*search_space[0][1], sizes[0])
        Y = np.linspace(*search_space[1][1], sizes[1])

        if use_torch:
            Xm, Ym = torch.meshgrid(torch.tensor(X), torch.tensor(Y))
            inputs = {n: torch.full_like(Xm, v) for n, v in presets.items()}
            inputs.update({search_space[0][0]: Xm, search_space[1][0]: Ym})
            Z = self.run(inputs).numpy()
        else:
            Xm, Ym = np.meshgrid(X, Y)
            Z = np.zeros((len(X), len(Y)))
            for x, y in itertools.product(range(len(X)), range(len(Y))):
                inputs = {n: v for n, v in presets.items()}
                inputs.update({search_space[0][0]: x, search_space[1][0]: y})
                Z[x][y] = self.run(inputs)

        return X, Y, Z, x_var, y_var


    def to_FIB(self, other):
        """ Creates a ForwardInfluenceBlanket object representing the
        intersection of this model with the other input model.

        Args:
            other: The GroundedFunctionNetwork object to compare this model to.

        Returns:
            A ForwardInfluenceBlanket object to use for model comparison.
        """

        if not isinstance(other, GroundedFunctionNetwork):
            raise TypeError(
                f"Expected GroundedFunctionNetwork, but got {type(other)}"
            )

        def shortname(var):
            return var[var.find("::") + 2 : var.rfind("_")]

        def shortname_vars(graph, shortname):
            return [v for v in graph.nodes() if shortname in v]

        this_var_nodes = [
            shortname(n)
            for (n, d) in self.nodes(data=True)
            if d["type"] == "variable"
        ]
        other_var_nodes = [
            shortname(n)
            for (n, d) in other.nodes(data=True)
            if d["type"] == "variable"
        ]

        shared_vars = set(this_var_nodes).intersection(set(other_var_nodes))
        full_shared_vars = {
            full_var
            for shared_var in shared_vars
            for full_var in shortname_vars(self, shared_var)
        }

        return ForwardInfluenceBlanket(self, full_shared_vars)

    def to_agraph(self):
        """ Export to a PyGraphviz AGraph object. """
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update(
            {"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "TB"}
        )
        A.node_attr.update({"fontname": "Menlo"})

        def build_tree(cluster_name, root_graph):
            subgraph_nodes = [
                n[0]
                for n in self.nodes(data=True)
                if n[1]["parent"] == cluster_name
            ]
            root_graph.add_nodes_from(subgraph_nodes)
            subgraph = root_graph.add_subgraph(
                subgraph_nodes,
                name=f"cluster_{cluster_name}",
                label=cluster_name,
                style="bold, rounded",
            )
            for n in self.scope_tree.successors(cluster_name):
                build_tree(n, subgraph)

        root = [n for n, d in self.scope_tree.in_degree() if d == 0][0]
        build_tree(root, A)
        return A

    def to_CAG_agraph(self):
        """Returns a variable-only view of the GrFN in the form of an AGraph.

        Returns:
            type: A CAG constructed via variable influence in the GrFN object.

        """
        CAG = self.to_CAG()
        A = nx.nx_agraph.to_agraph(CAG)
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
        A.node_attr.update(
            {
                "shape": "rectangle",
                "color": "#650021",
                "style": "rounded",
                "fontname": "Gill Sans",
            }
        )
        A.edge_attr.update({"color": "#650021", "arrowsize": 0.5})
        return A

    def to_call_agraph(self):
        """ Build a PyGraphviz AGraph object corresponding to a call graph of
        functions. """

        A = nx.nx_agraph.to_agraph(self.call_graph)
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
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
        super().__init__()
        self.output_node=G.output_node
        self.inputs = set(G.inputs).intersection(shared_nodes)

        # Get all paths from shared inputs to shared outputs
        path_inputs = shared_nodes - {self.output_node}
        io_pairs = [(inp, G.output_node) for inp in path_inputs]
        paths = [
            p for (i, o) in io_pairs for p in all_simple_paths(G, i, o)
        ]

        # Get all edges needed to blanket the included nodes
        main_nodes = {node for path in paths for node in path}
        main_edges = {
            (n1, n2) for path in paths for n1, n2 in zip(path, path[1:])
        }
        self.cover_nodes = set()
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
                self.cover_nodes.add(var_node)

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
        main_nodes = main_nodes - self.inputs - {self.output_node}

        orig_nodes = G.nodes(data=True)

        self.add_nodes_from([(n, d) for n, d in orig_nodes if n in self.inputs])
        for node in self.inputs:
            self.nodes[node]["color"] = dodgerblue3
            self.nodes[node]["fontcolor"] = dodgerblue3
            self.nodes[node]["penwidth"] = 3.0
            self.nodes[node]["fontname"] = FONT

        self.inputs = list(self.inputs)

        self.add_nodes_from([(n, d) for n, d in orig_nodes
                            if n in self.cover_nodes])
        for node in self.cover_nodes:
            self.nodes[node]["fontname"] = FONT
            self.nodes[node]["color"] = forestgreen
            self.nodes[node]["fontcolor"] = forestgreen

        self.add_nodes_from([(n, d) for n, d in orig_nodes if n in main_nodes])
        for node in main_nodes:
            self.nodes[node]["fontname"] = FONT

        self.add_node(self.output_node, **G.nodes[self.output_node])
        self.nodes[self.output_node]["color"] = dodgerblue3
        self.nodes[self.output_node]["fontcolor"] = dodgerblue3

        self.add_edges_from(main_edges)
        self.call_graph = self.build_call_graph()
        self.function_sets = self.build_function_sets()

    def run(
        self,
        inputs: Dict[str, Union[float, Iterable]],
        covers: Dict[str, Union[float, Iterable]],
        torch_size: Optional[int] = None,
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
        if len(covers) != len(self.cover_nodes):
            raise ValueError("Incorrect number of cover values.")

        # Set the cover node values
        for node_name, val in covers.items():
            self.nodes[node_name]["value"] = val

        return super().run(inputs, torch_size)

    def sobol_analysis(
        self, num_samples, prob_def, covers, use_torch=False, var_types=None
    ):
        def create_input_tensor(name, samples):
            type_info = var_types[name]
            if type_info[0] == str:
                (val1, val2) = type_info[1]
                return np.where(samples >= 0.5, val1, val2)
            else:
                return torch.tensor(samples)

        samples = saltelli.sample(
            prob_def, num_samples, calc_second_order=True
        )
        if use_torch:
            samples = np.split(samples, samples.shape[1], axis=1)
            samples = [s.squeeze() for s in samples]
            if var_types is None:
                values = {
                    n: torch.tensor(s)
                    for n, s in zip(prob_def["names"], samples)
                }
            else:
                values = {
                    n: create_input_tensor(n, s)
                    for n, s in zip(prob_def["names"], samples)
                }
            Y = self.run(values, covers, torch_size=len(samples[0])).numpy()
        else:
            Y = np.zeros(samples.shape[0])
            for i, sample in enumerate(samples):
                values = {n: val for n, val in zip(prob_def["names"], sample)}
                Y[i] = self.run(values, covers)

        return sobol.analyze(prob_def, Y)

    def S2_surface(self, sizes, bounds, presets, covers, use_torch=False,
            num_samples = 10):
        """Calculates the sensitivity surface of a GrFN for the two variables with
        the highest S2 index.

        Args:
            num_samples: Number of samples for sensitivity analysis.
            sizes: Tuple of (number of x inputs, number of y inputs).
            bounds: Set of bounds for GrFN inputs.
            presets: Set of standard values for GrFN inputs.

        Returns:
            Tuple:
                Tuple: The names of the two variables that were selected
                Tuple: The X, Y vectors of eval values
                Z: The numpy matrix of output evaluations

        """
        args = self.inputs
        Si = self.sobol_analysis(
            num_samples,
            {
                "num_vars": len(args),
                "names": args,
                "bounds": [bounds[arg] for arg in args],
            },
            covers
        )
        S2 = Si["S2"]
        (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

        x_var = args[v1]
        y_var = args[v2]
        search_space = [(x_var, bounds[x_var]), (y_var, bounds[y_var])]
        preset_vals = {
            arg: presets[arg]
            for i, arg in enumerate(args)
            if i != v1 and i != v2
        }

        X = np.linspace(*search_space[0][1], sizes[0])
        Y = np.linspace(*search_space[1][1], sizes[1])

        if use_torch:
            Xm, Ym = torch.meshgrid(torch.tensor(X), torch.tensor(Y))
            inputs = {n: torch.full_like(Xm, v) for n, v in presets.items()}
            inputs.update({search_space[0][0]: Xm, search_space[1][0]: Ym})
            Z = self.run(inputs, covers).numpy()
        else:
            Xm, Ym = np.meshgrid(X, Y)
            Z = np.zeros((len(X), len(Y)))
            for x, y in itertools.product(range(len(X)), range(len(Y))):
                inputs = {n: v for n, v in presets.items()}
                inputs.update({search_space[0][0]: x, search_space[1][0]: y})
                Z[x][y] = self.run(inputs, covers)

        return X, Y, Z, x_var, y_var

    def to_agraph(self):
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update({"dpi": 227, "fontsize": 20})
        A.node_attr.update({"shape": "rectangle", "style": "rounded"})
        A.edge_attr.update({"arrowsize": 0.5})
        return A
