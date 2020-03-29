import importlib
from pathlib import Path

import networkx as nx

from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork
from delphi.GrFN.utils import ScopeNode


def extract_GrFN(
    con_name: str, containers: dict, variables: dict, lambdas: dict
):
    """Builds the GrFN for container con_name given all containers, variables,
    and lambdas that were defined in the AutoMATES Intermediate Representation
    (AIR).

    NOTE: The lambdas for a particular container are referenced via the
    namespace portion of the container name string.

    Args:
        con_name: name of the container that is the root of the GrFN
        containers: All container objects from the AIR
        variables: All variable definitions from the AIR
        lambdas: container namespace --> string path to associated lambdas

    Returns:
        A GroundedFunctionNetwork object.

    """
    # TODO Adarsh: This is the old code from GroundedFunctionNetwork.from_dict()
    # it needs to be updated to work with the inputs provided by the interpreter
    lambdas = importlib.__import__(str(Path(lambdas_path).stem))
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

    def add_variable_node(
        parent: str, basename: str, index: str, is_exit: bool = False
    ):
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
            node_name = add_variable_node(
                scope.name, var_name, idx, is_exit=var_name == "EXIT"
            )
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
            return_list.append(make_variable_name(scope.name, basename, idx))

        for var_name in scope.updated:
            (_, basename, idx) = var_name.split("::")
            updated_list.append(make_variable_name(scope.name, basename, idx))
        return return_list, updated_list

    occurrences[con_name] = 0
    cur_scope = ScopeNode(functions[con_name], occurrences[con_name])
    scope_tree.add_node(cur_scope.name, color="forestgreen")
    returns, updates = process_container(cur_scope, [], con_name)
    return GroundedFunctionNetwork(G, scope_tree, returns + updates)
