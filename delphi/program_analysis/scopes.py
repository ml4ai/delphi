from abc import ABCMeta, abstractmethod
from typing import Dict


rv_maroon = "#650021"


class Scope(metaclass=ABCMeta):
    def __init__(self, name, data):
        self.name = name
        self.parent_scope = None
        self.child_names = list()
        self.child_nodes = list()
        self.child_vars = list()

        self.inputs = list()
        self.calls = dict()

        self.nodes = list()
        self.edges = list()

        self.json = data

        # NOTE: default edge color for debugging
        self.edge_color = "red"

        self.build_child_names()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        root = "(R) " if self.parent_scope is None else ""
        return f"{root}{self.name}: {self.child_names}"

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
            self.child_nodes.append(new_scope)

    def make_var_node(self, name, idx, scp):
        if isinstance(self, FuncScope):
            return FuncVariableNode(name=name, idx=idx, scp=scp)
        else:
            if name == self.index_var.name:
                return LoopVariableNode(
                    name=name, idx=idx, scp=scp, is_index=True
                )
            return LoopVariableNode(
                name=name, idx=idx, scp=scp, loop_var=self.index_var.name
            )

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
                # Do this for regular assignment/decision/condition operations and loop_plate(s)
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
                                isinstance(self, FuncScope)
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
                elif expr.get("inputs") is not None:
                    # This is a loop_plate node
                    plate_vars = list()
                    for var in expr["inputs"]:
                        # NOTE: cheating for now
                        new_var = self.make_var_node(var, "2", self.name)
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
                    inp_node = self.make_var_node(var["variable"], idx, scope)
                    call_vars.append(inp_node)
                    self.child_vars.append(call_vars)

    def add_nodes(self, sub):
        for node in self.nodes:
            clr = "black" if isinstance(node, ActionNode) else rv_maroon
            shape = "rectangle" if isinstance(node, ActionNode) else "ellipse"
            name = node.unique_name()
            label = node.get_label()
            sub.add_node(name, shape=shape, color=clr, label=label)

    def add_edges(self, sub):
        edges = [
            (src.unique_name(), dst.unique_name()) for src, dst in self.edges
        ]

        sub.add_edges_from(edges)

    @abstractmethod
    def build_containment_graph(self, graph, border_clr):
        sub = graph.add_subgraph(
            name=f"cluster_{self.name}",
            label=self.name,
            style="bold, rounded",
            color=border_clr,
        )

        self.add_nodes(sub)
        self.add_edges(sub)

        for child in self.child_nodes:
            child.build_containment_graph(sub)


class LoopScope(Scope):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.edge_color = "blue"

        self.index_var = LoopVariableNode(
            name=self.json["index_variable"],
            idx="0",
            scp=self.name,
            is_index=True,
        )

    def setup_from_json(self, vars=[]):
        self.inputs += self.json["input"]

        super().setup_from_json(vars)

        for child, vars in zip(self.child_nodes, self.child_vars):
            child.setup_from_json(vars)

    def build_containment_graph(self, graph):
        super().build_containment_graph(graph, self.edge_color)


class FuncScope(Scope):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.edge_color = "forestgreen"

    def setup_from_json(self, vars=[]):
        for var in self.json["input"]:
            self.inputs.append(var["name"])

        super().setup_from_json(vars)

        for child, vars in zip(self.child_nodes, self.child_vars):
            child.setup_from_json(vars)

    def build_containment_graph(self, graph):
        super().build_containment_graph(graph, self.edge_color)


class Node(metaclass=ABCMeta):
    def __init__(self, name="", idx="", scp=""):
        self.name = name
        self.index = idx
        self.scope = scp

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.unique_name()

    def unique_name(self):
        return "{}_{}__{}".format(self.name, self.index, self.scope)

    @abstractmethod
    def get_label(self):
        return NotImplemented


class FuncVariableNode(Node):
    def __init__(self, name="", idx="", scp=""):
        super().__init__(name=name, idx=idx, scp=scp)

    def get_label(self):
        return self.name


class ActionNode(Node):
    def __init__(self, name="", idx="", scp=""):
        super().__init__(name=name, idx=idx, scp=scp)
        start = name.find("__")
        end = name.rfind("__")
        self.action = name[start : end + 2]

    def get_label(self):
        return self.action


class LoopVariableNode(Node):
    def __init__(self, name="", idx="", scp="", is_index=False, loop_var=""):
        super().__init__(name=name, idx=idx, scp=scp)
        self.is_index = is_index
        if not self.is_index:
            self.loop_var = loop_var
            if int(idx) < 0:
                self.loop_index = -1
            else:
                self.loop_index = 0

    def get_label(self):
        if not self.is_index:
            return "{}\n@{}={}".format(
                self.name, self.loop_var, self.loop_index
            )
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

    scope_types_dict = {"container": FuncScope, "loop_plate": LoopScope}

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
