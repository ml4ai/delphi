from abc import ABCMeta, abstractmethod


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
        return "{}{}: {}".format(root, self.name, self.child_names)

    def build_child_names(self, data):
        for expr in data["body"]:
            if (expr.get("name") is not None and "loop_plate" in expr["name"]):
                self.child_names.append(expr["name"])
            if expr.get("function") is not None:
                self.child_names.append(expr["function"])

    def find_child_calls(self, data):
        for expr in data["body"]:
            if expr.get("function") is not None:
                name = expr["function"]
                variables = list()
                for inp_obj in expr["input"]:
                    variables.append("{}_{}".format(inp_obj["variable"], inp_obj["index"]))
                self.calls[name] = variables

    def build_scope_tree(self, all_scopes):
        for name in self.child_names:
            new_scope = all_scopes[name]
            new_scope.build_scope_tree(all_scopes)
            new_scope.parent_scope = self
            self.child_nodes.append(new_scope)

    def remove_non_scope_children(self, scopes):
        saved_children = list()
        for child in self.child_names:
            if child in scopes:
                saved_children.append(child)

        self.child_names = saved_children

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
                            iname = "{}_{}".format(i["variable"], i["index"])
                            self.node_types[iname] = "variable"
                            self.node_pairs.append((iname, instruction))

                output = expr["output"]
                if output.get("variable") is not None:
                    oname = "{}_{}".format(output["variable"], output["index"])
                    self.node_types[oname] = "variable"
                    self.node_pairs.append((instruction, oname))

    @abstractmethod
    def build_containment_graph(self, graph, border_clr):
        sub = graph.add_subgraph(name="cluster_{}".format(self.name),
                                 color=border_clr)
        sub.graph_attr["label"] = self.name

        for node, n_type in self.node_types.items():
            clr = "red" if n_type == "factor" else "black"
            shape = "rectangle" if n_type == "factor" else "ellipse"
            name = "{}\n{}".format(node, self.name)
            sub.add_node(name, shape=shape, color=clr)

        for src, dst in self.node_pairs:
            src_name = "{}\n{}".format(src, self.name)
            dst_name = "{}\n{}".format(dst, self.name)
            sub.add_edge(src_name, dst_name)

        for child in self.child_nodes:
            child.build_containment_graph(sub)

    @abstractmethod
    def build_linked_graph(self, graph):
        return NotImplemented


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

    def build_linked_graph(self, graph):
        sub = graph.add_subgraph(name="cluster_{}".format(self.name),
                                 color=self.edge_color)
        sub.graph_attr["label"] = self.name

        def get_node_name(node):
            # if self.node_types[node] == "factor":
            #     return "{}\n{}".format(node, self.name)
            # else:
            #     return "{}\n{}".format(node, self.parent_scope.name)
            return "{}\n{}".format(node, self.parent_scope.name)

        for node, n_type in self.node_types.items():
            clr = "red" if n_type == "factor" else "black"
            shape = "rectangle" if n_type == "factor" else "ellipse"
            name = get_node_name(node)
            sub.add_node(name, shape=shape, color=clr)

        for src, dst in self.node_pairs:
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)
            sub.add_edge(src_name, dst_name)

        for child in self.child_nodes:
            child.build_linked_graph(sub)


class FuncScopeNode(ScopeNode):
    def __init__(self, name, json_data):
        super().__init__(name, json_data)
        self.edge_color = "green"
        self.setup_from_json(json_data)

    def setup_from_json(self, data):
        for input_obj in data["input"]:
            self.inputs.append(input_obj["name"])

        for var in data["variables"]:
            self.variables.append(var["name"])

        super().setup_from_json(data)

    def build_containment_graph(self, graph):
        super().build_containment_graph(graph, self.edge_color)

    def build_linked_graph(self, graph):
        sub = graph.add_subgraph(name="cluster_{}".format(self.name),
                                 color=self.edge_color)
        sub.graph_attr["label"] = self.name

        def get_node_name(node):
            if node.endswith("0") and self.parent_scope is not None:
                possible_vars = self.parent_scope.calls[self.name]
                for var in possible_vars:
                    prefix = var[:var.rindex("_")]
                    if node.startswith(prefix):
                        if prefix in self.parent_scope.variables:
                            return "{}\n{}".format(var, self.parent_scope.name)
                        else:
                            return "{}\n{}".format(var, self.parent_scope.parent_scope.name)

            return "{}\n{}".format(node, self.name)

        for node, n_type in self.node_types.items():
            clr = "red" if n_type == "factor" else "black"
            shape = "rectangle" if n_type == "factor" else "ellipse"
            name = get_node_name(node)
            sub.add_node(name, shape=shape, color=clr)

        for src, dst in self.node_pairs:
            src_name = get_node_name(src)
            dst_name = get_node_name(dst)
            sub.add_edge(src_name, dst_name)

        for child in self.child_nodes:
            child.build_linked_graph(sub)
