import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Union, Set
import subprocess as sp
import importlib
import inspect
import json
import os
import ast

import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths

import delphi.GrFN.utils as utils
from delphi.GrFN.utils import NodeType
from delphi.utils.misc import choose_font
from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)

FONT = choose_font()

dodgerblue3 = "#1874CD"
forestgreen = "#228b22"


class GroundedFunctionNetwork(nx.DiGraph):
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

    def __init__(self, nodes, edges, subgraphs):
        super(GroundedFunctionNetwork, self).__init__()
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        self.scopes = subgraphs

        self.inputs = [n for n, d in self.in_degree() if d == 0]
        self.outputs = [n for n, d in self.out_degree() if d == 0]

        self.model_inputs = [n for n in self.inputs
                             if self.nodes[n]["type"] == NodeType.VARIABLE]
        self.output_node = self.outputs[-1]

        A = self.to_agraph()
        A.draw("petasce.pdf", prog="dot")

        self.build_call_graph()
        self.build_function_sets()

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
            repr = self.nodes[n]["name"] \
                if self.nodes[n]["type"] == NodeType.VARIABLE else \
                f"{self.nodes[n]['name']}{inspect.signature(self.nodes[n]['lambda'])}"

            result.append(f"{tab * depth}{repr}")
            result.extend(self.traverse_nodes(self.successors(n), depth=depth+1))
        return result

    @classmethod
    def from_json(cls, file: str):
        """Builds a GrFN from a JSON object.

        Args:
            cls: The class variable for object creation.
            file: Filename of a GrFN JSON file.

        Returns:
            type: A GroundedFunctionNetwork object.

        """
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

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
        nodes, edges, subgraphs = list(), list(), dict()

        # Get a list of all container/loop plates contained in the data object
        containers = {obj["name"]: obj for obj in data["functions"]
                      if obj["type"] in ["container", "loop_plate"]}

        def process_container(container: Dict, inputs: Dict[str, Dict[str, str]]) -> None:
            """Wires the body statements found in a given container/loop plate.

            Args:
                container: The container object containing the body
                    statements that specify GrFN wiring.
                inputs: A dict of input variables from the outer container.

            Returns:
                None

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
                    stmt_type = utils.get_node_type(short_type)
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
                        var["variable"]: utils.get_variable_name(var, con_name)
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
                            input_node_name = utils.get_variable_name(var, con_name)

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
                        "func_visited": False,
                        "lambda": getattr(lambdas, stmt_name),  # Gets the lambda function
                        "func_inputs": ordered_inputs,          # saves indexed arg ordering
                        "scope": con_name
                    }))

                    # Add output node and node unique name to edges, subgraph set, and arg set
                    out_node_name = utils.get_variable_name(stmt["output"], con_name)
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
    def from_python_file(cls, python_file, lambdas_path, json_filename: str, stem: str):
        """Builds GrFN object from Python file."""
        with open(python_file, "r") as f:
            pySrc = f.read()
        return cls.from_python_src(pySrc, lambdas_path, json_filename, stem)

    @classmethod
    def from_python_src(cls, pySrc, lambdas_path, json_filename: str, stem: str):
        """Builds GrFN object from Python source code."""
        asts = [ast.parse(pySrc)]
        pgm_dict = genPGM.create_pgm_dict(
            lambdas_path, asts, json_filename, {"FileName": f"{stem}.py"}, save_file=True    # HACK
        )
        lambdas = importlib.__import__(stem + "_lambdas")
        return cls.from_dict(pgm_dict, lambdas)

    @classmethod
    def from_fortran_file(cls, fortran_file: str, tmpdir: str="."):
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
        trees = [ET.fromstring(xml_string)]
        comments = get_comments.get_comments(preprocessed_fortran_file)
        os.remove(preprocessed_fortran_file)
        xml_to_json_translator = translate.XMLToJSONTranslator()
        outputDict = xml_to_json_translator.analyze(trees, comments)
        pySrc = pyTranslate.create_python_source_list(outputDict)[0][0]

        return cls.from_python_src(pySrc, lambdas_path, json_filename, stem)

    def clear(self):
        """Clear variable nodes for next computation."""
        for n in self.nodes():
            if self.nodes[n]["type"] == NodeType.VARIABLE:
                self.nodes[n]["value"] = None
            elif self.nodes[n]["type"].is_function_node():
                self.nodes[n]["func_visited"] = False

    def build_call_graph(self):
        edges = list()

        def update_edge_set(cur_fns):
            for c in cur_fns:
                nxt_fns = [p for v in self.successors(c)
                           for p in self.successors(v)]
                edges.extend([(c, n) for n in nxt_fns])
                update_edge_set(list(set(nxt_fns)))

        for inp in self.model_inputs:
            print(inp)
        print(len(self.model_inputs))
        update_edge_set(list({n for v in self.model_inputs for n in self.successors(v)}))
        self.call_graph = nx.DiGraph()
        self.call_graph.add_edges_from(edges)

    def build_function_sets(self):
        initial_funcs = [n for n, d in self.call_graph.in_degree() if d == 0]
        distances = dict()

        def find_distances(funcs, dist):
            all_successors = list()
            for func in funcs:
                distances[func] = dist
                all_successors.extend(self.call_graph.successors(func))
            if len(all_successors) > 0:
                find_distances(list(set(all_successors)), dist+1)

        find_distances(initial_funcs, 0)
        call_sets = dict()
        for func_name, call_dist in distances.items():
            if call_dist in call_sets:
                call_sets[call_dist].add(func_name)
            else:
                call_sets[call_dist] = {func_name}

        function_set_dists = sorted(call_sets.items(), key=lambda t: (t[0], len(t[1])))
        self.function_sets = [func_set for _, func_set in function_set_dists]

    @utils.timeit
    def run(self, inputs: Dict[str, Union[float, Iterable]]) -> Union[float, Iterable]:
        """Executes the GrFN over a particular set of inputs and returns the
        result.

        Args:
            inputs: Input set where keys are the names of input nodes in the
              GrFN and each key points to a set of input values (or just one).

        Returns:
            A set of outputs from executing the GrFN, one for every set of
            inputs.

        """
        # Abort run if inputs does not match our expected input set
        if len(inputs) != len(self.model_inputs):
            raise ValueError("Incorrect number of inputs.")

        # Set input values
        for node_name, val in inputs.items():
            self.nodes[node_name]["value"] = val

        for func_set in self.function_sets:
            for func_name in func_set:
                # Get function arguments via signature derived from JSON
                signature = self.nodes[func_name]["func_inputs"]
                lambda_fn = self.nodes[func_name]["lambda"]
                output_node = list(self.successors(func_name))[0]

                # Run the lambda function and save the output
                res = lambda_fn(*(self.nodes[n]["value"] for n in signature))
                self.nodes[output_node]["value"] = res

        # return the output
        return self.nodes[self.output_node]["value"]

    def to_CAG(self):
        nodes, edges = list(), list()

        def gather_nodes_and_edges(prev_name, inputs):
            """Recursively constructs CAG node and edge sets via variable lists.

            Args:
                prev_name: Parent variable of the input variable set.
                inputs: set of current variables.

            Returns:
                type: None.

            """
            for name in inputs:
                uniq_name = name[name.find("::") + 2: name.rfind("_")]
                nodes.append(uniq_name)

                if prev_name is not None:
                    edges.append((prev_name, uniq_name))

                next_inputs = list(set([v for f in self.successors(name)
                                        for v in self.successors(f)]))
                gather_nodes_and_edges(uniq_name, next_inputs)

        gather_nodes_and_edges(None, self.model_inputs)
        CAG = nx.DiGraph()
        CAG.add_nodes_from(nodes)
        CAG.add_edges_from(edges)
        return CAG

    def to_FIB(self, other):
        if not isinstance(other, GroundedFunctionNetwork):
            raise TypeError(f"Expected GroundedFunctionNetwork, but got {type(other)}")

        def shortname(var):
            return var[var.find("::") + 2: var.rfind("_")]

        def shortname_vars(graph, shortname):
            return [v for v in graph.nodes() if shortname in v]

        this_var_nodes = [shortname(n) for (n, d) in self.nodes(data=True)
                          if d["type"] == NodeType.VARIABLE]
        other_var_nodes = [shortname(n) for (n, d) in other.nodes(data=True)
                           if d["type"] == NodeType.VARIABLE]

        shared_vars = set(this_var_nodes).intersection(set(other_var_nodes))
        full_shared_vars = {full_var for shared_var in shared_vars
                            for full_var in shortname_vars(self, shared_var)}

        return ForwardInfluenceBlanket(self, full_shared_vars)

    def to_agraph(self):
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
        A.node_attr.update({ "fontname": FONT })
        for n in A.nodes():
            if self.nodes[n]["type"] == NodeType.VARIABLE:
                A.add_node(n, color="maroon", shape="ellipse")
            else:
                A.add_node(n, color="black", shape="rectangle")

        return A

    def to_CAG_agraph(self):
        """Returns a variable-only view of the GrFN in the form of an AGraph.

        Returns:
            type: A CAG constructed via variable influence in the GrFN object.

        """
        CAG = self.to_CAG()
        A = nx.nx_agraph.to_agraph(CAG)
        A.graph_attr.update({
            "dpi": 227,
            "fontsize": 20,
            "fontname": "Menlo"
        })
        A.node_attr.update({
            "shape": "rectangle",
            "color": "#650021",
            "style": "rounded",
            "fontname": "Gill Sans",
        })
        A.edge_attr.update({
            "color": "#650021",
            "arrowsize": 0.5
        })
        return A

    def to_call_agraph(self) -> nx.DiGraph:
        A = nx.nx_agraph.to_agraph(self.call_graph)
        A.graph_attr.update({
            "dpi": 227,
            "fontsize": 20,
            "fontname": "Menlo"
        })
        A.node_attr.update({
            "shape": "rectangle",
            "color": "#650021",
            "style": "rounded",
            "fontname": "Gill Sans",
        })
        A.edge_attr.update({
            "color": "#650021",
            "arrowsize": 0.5
        })
        return A


class ForwardInfluenceBlanket(nx.DiGraph):
    """
    This class takes a network and a list of a shared nodes between the input
    network and a secondary network. From this list a shared nodes and blanket
    network is created including all of the nodes between any input/output pair
    in the shared nodes, as well as all nodes required to blanket the network
    for forward influence. This class itself becomes the blanket and inherits
    from the NetworkX DiGraph class.
    """
    def __init__(self, G: GroundedFunctionNetwork, shared_nodes: Set[str]):
        super(ForwardInfluenceBlanket, self).__init__()
        self.inputs = set(G.model_inputs).intersection(shared_nodes)
        self.output_node = G.output_node

        # Get all paths from shared inputs to shared outputs
        path_inputs = shared_nodes - {self.output_node}
        io_pairs = [(inp, self.output_node) for inp in path_inputs]
        paths = [p for (i, o) in io_pairs for p in all_simple_paths(G, i, o)]

        # Get all edges needed to blanket the included nodes
        main_nodes = {node for path in paths for node in path} - self.inputs - {self.output_node}
        main_edges = {(n1, n2) for p in paths for n1, n2 in zip(p, p[1:])}
        self.cover_nodes, cover_edges = set(), set()
        for path in paths:
            first_node = path[0]
            for func_node in G.predecessors(first_node):
                for var_node in G.predecessors(func_node):
                    if var_node not in main_nodes:
                        self.cover_nodes.add(var_node)
                        main_nodes.add(func_node)
                        cover_edges.add((var_node, func_node))
                        main_edges.add((func_node, first_node))

        orig_nodes = G.nodes(data=True)
        self.add_nodes_from(
            [(n, d) for n, d in orig_nodes if n in self.inputs],
            color=dodgerblue3, fontcolor=dodgerblue3, fontname=FONT,
            penwidth=3.0
        )
        self.add_node(
            (self.output_node, G.nodes[self.output_node]),
            color=dodgerblue3, fontcolor=dodgerblue3, fontname=FONT,
        )
        self.add_nodes_from(
            [(n, d) for n, d in orig_nodes if n in main_nodes],
            fontname=FONT
        )
        self.add_nodes_from(
            [(n, d) for n, d in orig_nodes if n in self.cover_nodes],
            color=forestgreen, fontcolor=forestgreen, fontname=FONT,
        )
        self.add_edges_from(main_edges)
        self.add_edges_from(
            cover_edges, color=forestgreen,
            fontcolor=forestgreen, fontname=FONT,
        )

        for node_name in shared_nodes:
            for dest in self.successors(node_name):
                self[node_name][dest]["color"] = dodgerblue3
                self[node_name][dest]["fontcolor"] = dodgerblue3

        # NOTE: Adding cut nodes as needed for display only!!
        # cut_nodes = [n for n in G.nodes if n not in self.nodes]
        # cut_edges = [e for e in G.edges if e not in self.edges]
        #
        # self.add_nodes_from(cut_nodes)
        # self.add_edges_from(cut_edges)
        #
        # for node_name in cut_nodes:
        #     self.nodes[node_name]["color"] = "orange"
        #     self.nodes[node_name]["fontcolor"] = "orange"
        #
        # for source, dest in cut_edges:
        #     self[source][dest]["color"] = "orange"

        self.build_call_graph()
        self.build_function_sets()

    def build_call_graph(self):
        edges = list()

        def update_edge_set(cur_fns):
            for c in cur_fns:
                nxt_fns = [p for v in self.successors(c)
                           for p in self.successors(v)]
                edges.extend([(c, n) for n in nxt_fns])
                update_edge_set(list(set(nxt_fns)))

        update_edge_set(
            list({
                n for v in self.inputs for n in self.successors(v)
            }.union({
                n for v in self.cover_nodes for n in self.successors(v)
            }))
        )
        self.call_graph = nx.DiGraph()
        self.call_graph.add_edges_from(edges)

    def build_function_sets(self):
        initial_funcs = [n for n, d in self.call_graph.in_degree() if d == 0]
        distances = dict()

        def find_distances(funcs, dist):
            all_successors = list()
            for func in funcs:
                distances[func] = dist
                all_successors.extend(self.call_graph.successors(func))
            if len(all_successors) > 0:
                find_distances(list(set(all_successors)), dist+1)

        find_distances(initial_funcs, 0)
        call_sets = dict()
        for func_name, call_dist in distances.items():
            if call_dist in call_sets:
                call_sets[call_dist].add(func_name)
            else:
                call_sets[call_dist] = {func_name}

        function_set_dists = sorted(call_sets.items(), key=lambda t: (t[0], len(t[1])))
        self.function_sets = [func_set for _, func_set in function_set_dists]

    @utils.timeit
    def run(self, inputs: Dict[str, Union[float, Iterable]], covers: Dict[str, Union[float, Iterable]]) -> Union[float, Iterable]:
        """Executes the GrFN over a particular set of inputs and returns the
        result.

        Args:
            inputs: Input set where keys are the names of input nodes in the
              GrFN and each key points to a set of input values (or just one).

        Returns:
            A set of outputs from executing the GrFN, one for every set of
            inputs.

        """
        # Abort run if inputs does not match our expected input set
        if len(inputs) != len(self.model_inputs):
            raise ValueError("Incorrect number of inputs.")

        if len(covers) != len(self.cover_nodes):
            raise ValueError("Incorrect number of cover values.")

        # Set input values
        for node_name, val in inputs.items():
            self.nodes[node_name]["value"] = val

        for node_name, val in covers.items():
            self.nodes[node_name]["value"] = val

        for func_set in self.function_sets:
            for func_name in func_set:
                # Get function arguments via signature derived from JSON
                signature = self.nodes[func_name]["func_inputs"]
                lambda_fn = self.nodes[func_name]["lambda"]
                output_node = list(self.successors(func_name))[0]

                # Run the lambda function and save the output
                res = lambda_fn(*(self.nodes[n]["value"] for n in signature))
                self.nodes[output_node]["value"] = res

        # return the output
        return self.nodes[self.output_node]["value"]

    def to_agraph(self):
        A = nx.nx_agraph.to_agraph(self)
        A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo"})
        A.node_attr.update({
            "shape": "rectangle",
            # "color": "#650021",
            "style": "rounded",
            # "fontname": "Gill Sans",
        })
        A.edge_attr.update({
            # "color": "#650021",
            "arrowsize": 0.5
        })
        return A
