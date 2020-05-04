import networkx as nx


class GroundedFunctionNetwork(nx.DiGraph):
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
        functions = {d["name"]: d for d in data["containers"]}
        variables = {v["name"]: v for v in data["variables"]}
        varname2id = {name: str(uuid.uuid4()) for name in variables.keys()}
        occurrences = {}
        G = nx.DiGraph()
        scope_tree = nx.DiGraph()

        def identity(x):
            return x

        def get_variable_reference(parent: str, basename: str, index: str):
            (namespace, context, container, _) = parent.split("::")
            if context != "@global":
                container = f"{context}.{container}"
            return f"@variable::{namespace}::{container}::{basename}::{index}"

        def add_variable_node(
            parent: str, basename: str, index: str, is_exit: bool = False
        ):
            var_identifier = varname2id[
                get_variable_reference(parent, basename, index)
            ]
            G.add_node(
                var_identifier,
                type="variable",
                color="crimson",
                fontcolor="white" if is_exit else "black",
                fillcolor="crimson" if is_exit else "white",
                style="filled" if is_exit else "",
                parent=parent,
                basename=basename,
                index=index,
                label=f"{basename}\n({index})",
                cag_label=f"{basename}",
                padding=15,
                value=None,
            )
            return var_identifier

        def process_wiring_statement(stmt, scope, inputs, cname):
            lambda_name = stmt["function"]["name"]
            lambda_identifier = str(uuid.uuid4())

            stmt_type = lambda_name.split("__")[-3]
            if stmt_type == "assign" and len(stmt["input"]) == 0:
                stmt_type = "literal"

            for output in stmt["output"]:
                (_, var_name, idx) = output.split("::")
                node_name = add_variable_node(
                    scope.name, var_name, idx, is_exit=var_name == "EXIT"
                )
                G.add_edge(lambda_identifier, node_name)

            ordered_inputs = list()
            for inp in stmt["input"]:
                if inp.endswith("-1"):
                    (parent, var_name, idx) = inputs[inp]
                else:
                    parent = scope.name
                    (_, var_name, idx) = inp.split("::")

                node_name = add_variable_node(parent, var_name, idx)
                ordered_inputs.append(node_name)
                G.add_edge(node_name, lambda_identifier)

            G.add_node(
                lambda_identifier,
                type="function",
                func_type=stmt_type,
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
            container_color = (
                "navyblue" if new_container["repeat"] else "forestgreen"
            )
            new_scope = ScopeNode(
                new_container, occurrences[container_name], parent=scope
            )
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
                lambda_identifier = str(uuid.uuid4())
                G.add_node(
                    lambda_identifier,
                    type="function",
                    func_type="assign",
                    lambda_fn=identity,
                    func_inputs=[callee_var],
                    shape="rectangle",
                    parent=scope.name,
                    label="A",
                    padding=10,
                )
                G.add_edge(callee_var, lambda_identifier)
                G.add_edge(lambda_identifier, caller_var)

            for callee_var, caller_var in zip(callee_up, caller_up):
                lambda_identifier = str(uuid.uuid4())
                G.add_node(
                    lambda_identifier,
                    type="function",
                    func_type="assign",
                    lambda_fn=identity,
                    func_inputs=[callee_var],
                    shape="rectangle",
                    parent=scope.name,
                    label="A",
                    padding=10,
                )
                G.add_edge(callee_var, lambda_identifier)
                G.add_edge(lambda_identifier, caller_var)
            occurrences[container_name] += 1

        def process_container(scope, input_vals, cname):
            if len(scope.arguments) == len(input_vals):
                input_vars = {
                    a: v for a, v in zip(scope.arguments, input_vals)
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
                    process_wiring_statement(stmt, scope, input_vars, cname)
                elif func_type == "container":
                    process_call_statement(stmt, scope, input_vars, cname)
                else:
                    raise ValueError(f"Undefined function type: {func_type}")

            return_list, updated_list = list(), list()
            for var_name in scope.returns:
                (_, basename, idx) = var_name.split("::")
                return_list.append(
                    varname2id[
                        get_variable_reference(scope.name, basename, idx)
                    ]
                )

            for var_name in scope.updated:
                (_, basename, idx) = var_name.split("::")
                updated_list.append(
                    varname2id[
                        get_variable_reference(scope.name, basename, idx)
                    ]
                )
            return return_list, updated_list

        root = data["start"][0]
        occurrences[root] = 0
        cur_scope = ScopeNode(functions[root], occurrences[root])
        scope_tree.add_node(cur_scope.name, color="forestgreen")
        returns, updates = process_container(cur_scope, [], root)
        return cls(G, scope_tree, returns + updates)
