from abc import ABCMeta, abstractmethod
from typing import Dict


rv_maroon = "#650021"


class ScopeNode(metaclass=ABCMeta):
    def __init__(self, name, data):
        self.name = name
        self.child_names = list()
        self.child_nodes = list()
        self.inputs = list()
        self.variables = list()
        self.nodes = list()
        self.calls = dict()
        self.node_pairs = list()
        self.node_types = dict()
        self.is_root = False
        self.parent_scope = None

        self.build_child_names(data)
        self.find_child_calls(data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        root = "(R) " if self.is_root else ""
        return f"{root}{self.name}: {self.child_names}"

    def build_child_names(self, data):
        for expr in data["body"]:
            if (expr.get("name") is not None and "loop_plate" in expr["name"]):
                self.child_names.append(expr["name"])
            if expr.get("function") is not None:
                self.child_names.append(expr["function"])

    def find_child_calls(self, data):
        for expr in data["body"]:
            if expr.get("function") is not None:
                self.calls[expr["function"]] = [
                    f"{inp_obj['variable']}_{inp_obj['index']}"
                    for inp_obj in expr["input"]
                ]

    def build_scope_tree(self, all_scopes):
        for name in self.child_names:
            new_scope = all_scopes[name]
            new_scope.build_scope_tree(all_scopes)
            new_scope.parent_scope = self
            self.child_nodes.append(new_scope)

    def remove_non_scope_children(self, scopes):
        self.child_names = [c for c in self.child_names if c in scopes]

    @abstractmethod
    def setup_from_json(self, data):
        for expr in data["body"]:
            if expr.get("name") is not None:
                instruction = expr["name"]
                if len(expr.get("output")) > 0:
                    self.node_types[instruction] = "factor"
                if expr.get("input") is not None:
                    for i in expr["input"]:
                        if i.get("variable") is not None:
                            iname = f"{i['variable']}_{i['index']}"
                            self.node_types[iname] = "variable"
                            self.node_pairs.append((iname, instruction))

                output = expr["output"]
                if output.get("variable") is not None:
                    oname = f"{output['variable']}_{output['index']}"
                    self.node_types[oname] = "variable"
                    self.node_pairs.append((instruction, oname))

    def get_node_name(self, node):
        if '__assign__' in node:
            blocks = node.split('__assign__')
            node = '\n__assign__\n'.join(blocks)
        if '__condition__' in node:
            blocks = node.split('__condition__')
            node = '\n__condition__\n'.join(blocks)
        if '__decision__' in node:
            blocks = node.split('__decision__')
            node = '\n__decision__\n'.join(blocks)

        return f"{node}\n{self.name}"

    def add_nodes(self, sub):
        for node, n_type in self.node_types.items():
            clr = "black" if n_type == "factor" else rv_maroon
            shape = "rectangle" if n_type == "factor" else "ellipse"
            name = self.get_node_name(node)
            sub.add_node(name, shape=shape, color=clr)

    @abstractmethod
    def build_containment_graph(self, graph, border_clr):
        sub = graph.add_subgraph(name=f"cluster_{self.name}",
                                 color=border_clr)
        sub.graph_attr["label"] = self.name
        self.add_nodes(sub)

        for src, dst in self.node_pairs:
            src_name = f"{src}\n{self.name}"
            dst_name = f"{dst}\n{self.name}"
            sub.add_edge(src_name, dst_name)

        for child in self.child_nodes:
            child.build_containment_graph(sub)

    def build_linked_graph(self, graph):
        sub = graph.add_subgraph(name=f"cluster_{self.name}",
                                 color=self.edge_color,
                                 style='bold, rounded',
                                 label=self.name)

        self.add_nodes(sub)
        edges = [(self.get_node_name(src), self.get_node_name(dst))
                 for src, dst in self.node_pairs]

        sub.add_edges_from(edges)

        for child in self.child_nodes:
            child.build_linked_graph(sub)


class LoopScopeNode(ScopeNode):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.edge_color = "blue"
        self.setup_from_json(json_data)

    def setup_from_json(self, data):
        self.inputs.append(data["index_variable"])
        self.variables.append(data["index_variable"])

        for var in data["variables"]:
            self.variables.append(var["name"])

        super().setup_from_json(data)

    def build_containment_graph(self, graph):
        super().build_containment_graph(graph, self.edge_color)

    def get_node_name(self, node):
        if '__assign__' in node:
            blocks = node.split('__assign__')
            node = '\n__assign__\n'.join(blocks)
        if '__condition__' in node:
            blocks = node.split('__condition__')
            node = '\n__condition__\n'.join(blocks)
        if '__decision__' in node:
            blocks = node.split('__decision__')
            node = '\n__decision__\n'.join(blocks)

        return f"{node}\n{self.parent_scope.name}"


class FuncScopeNode(ScopeNode):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.edge_color = "forestgreen"
        self.setup_from_json(json_data)

    def setup_from_json(self, data):
        for input_obj in data["input"]:
            self.inputs.append(input_obj["name"])

        for var in data["variables"]:
            self.variables.append(var["name"])

        super().setup_from_json(data)

    def build_containment_graph(self, graph):
        super().build_containment_graph(graph, self.edge_color)

    def get_node_name(self, node):
        if node.endswith("0") and self.parent_scope is not None:
            possible_vars = self.parent_scope.calls[self.name]
            for var in possible_vars:
                prefix = var[:var.rindex("_")]
                if node.startswith(prefix):
                    if prefix in self.parent_scope.variables:
                        return f"{var}\n{self.parent_scope.name}"
                    else:
                        return f"{var}\n{self.parent_scope.parent_scope.name}"

        if '__assign__' in node:
            blocks = node.split('__assign__')
            node = '\n__assign__\n'.join(blocks)
        if '__condition__' in node:
            blocks = node.split('__condition__')
            node = '\n__condition__\n'.join(blocks)
        if '__decision__' in node:
            blocks = node.split('__decision__')
            node = '\n__decision__\n'.join(blocks)

        return f"{node}\n{self.name}"


def scope_tree_from_json(json_data: Dict) -> ScopeNode:
    """
    Parses through a dictionary of data from a JSON spec of a FORTRAN program
    and returns a tree of ScopeNode objects that represent the properly nested
    scopes found in the JSON data.

    Args:
        json_data: A dict of all data found when parsing the FORTRAN program

    Returns:
        A ScopeNode that serves as the root of a tree of ScopeNodes
    """

    # Build a new scope object for each function and loop_plate object. Index
    # scopes into a dict by (scope_name |-> scope)

    scope_types_dict = {'container': FuncScopeNode, 'loop_plate': LoopScopeNode}

    scopes = {f['name']: scope_types_dict[f['type']](f['name'], f)
              for f in json_data['functions']
              if f['type'] in scope_types_dict}

    # Make a list of all scopes by scope names
    scope_names = list(scopes.keys())

    # Remove pseudo-scopes we wish to not display (such as print)
    for scope in scopes.values():
        scope.remove_non_scope_children(scope_names)

    # Find the root scope to properly nest scopes in a tree. NOTE: This will no
    # longer work when we have functions with the same name at different scope
    # levels. Not sure if this is possible in FORTRAN but we should be prepared.
    root = None
    for name in scope_names:
        dependent = False
        for c1 in scopes.values():
            if name in c1.child_names:
                dependent = True
                break
        if not dependent:
            root = scopes[name]
            break

    # Build the nested tree of scopes using recursion
    root.build_scope_tree(scopes)
    return root
