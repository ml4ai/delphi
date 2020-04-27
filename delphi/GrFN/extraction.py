import importlib
from pathlib import Path
from functools import singledispatch

import networkx as nx

from typing import Dict
from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.structures import (
    GenericSubgraph,
    FuncSubgraph,
    CondSubgraph,
    LoopSubgraph,
    GenericStmt,
    LambdaStmt,
    CallStmt,
)


def extract_GrFN(
    con_name: str,
    containers: dict,
    variables: dict,
    container_lambdas_map: Dict[str, str],
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
        container_lambdas_map: A map from each container namespace with to the string path to
            the corresponding associated lambdas.py file.

    Returns:
        A GroundedFunctionNetwork object.

    """
    # TODO Adarsh: This is the old code from GroundedFunctionNetwork.from_dict()
    # it needs to be updated to work with the inputs provided by the interpreter
    lambdas = importlib.__import__(str(Path(lambdas_path).stem))
    occurrences = {}
    G = nx.DiGraph()
    scope_tree = nx.DiGraph()

    def identity(x):
        return x

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

    @singledispatch
    def process_statment(stmt: GenericStmt, inputs: list) -> None:
        raise TypeError(f"Unrecognized statment type: {type(stmt)}")

    @process_statment.register
    def _(stmt: LambdaStmt, inputs: list) -> None:
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

    @process_statment.register
    def _(stmt: LambdaStmt, inputs: list) -> None:
        container_name = stmt["function"]["name"]
        if container_name not in occurrences:
            occurrences[container_name] = 0

        new_container = containers[container_name]
        new_scope = create_container_node(
            new_container, occurrences[container_name], parent=scope
        )

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
    def process_container(scope: GenericSubgraph, scope_inputs: list) -> tuple:
        raise TypeError(f"Unrecognized container scope: {type(scope)}")

    @process_container.register
    def _(scope: FuncSubgraph, scope_inputs: list) -> tuple:
        if len(scope.arguments) == len(scope_inputs):
            input_vars = {a: v for a, v in zip(scope.arguments, scope_inputs)}
        elif len(scope.arguments) > 0:
            input_vars = {
                a: (scope.name,) + tuple(a.split("::")[1:])
                for a in scope.arguments
            }

        for stmt in scope.body:
            func_def = stmt["function"]
            func_type = func_def["type"]
            if func_type == "lambda":
                process_wiring_statement(stmt, scope, input_vars, scope.name)
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
            return_list.append(make_variable_name(scope.name, basename, idx))

        for var_name in scope.updated:
            (_, basename, idx) = var_name.split("::")
            updated_list.append(make_variable_name(scope.name, basename, idx))
        return return_list, updated_list

    @process_container.register
    def _(scope: CondSubgraph, scope_inputs: list) -> tuple:
        if len(scope.arguments) == len(scope_inputs):
            input_vars = {a: v for a, v in zip(scope.arguments, scope_inputs)}
        elif len(scope.arguments) > 0:
            input_vars = {
                a: (scope.name,) + tuple(a.split("::")[1:])
                for a in scope.arguments
            }

        conditions, statements = list(), list()
        for cond_obj in scope.body:
            cond = cond_obj["condition"]
            stmts = cond_obj["statements"]

            if cond is None:
                conditions.append(None)
            else:
                # FIXME: generalize cond handling in the future
                cond_stmt = cond[0]
                process_wiring_statement(
                    cond_stmt, scope, input_vars, scope.name
                )

            for stmt in stmts:
                func_def = stmt["function"]
                func_type = func_def["type"]
                if func_type == "lambda":
                    process_wiring_statement(
                        stmt, scope, input_vars, scope.name
                    )
                elif func_type == "container":
                    process_call_statement(stmt, scope, input_vars, scope.name)
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

    @process_container.register
    def _(scope: LoopSubgraph, scope_inputs: list) -> tuple:
        if len(scope.arguments) == len(scope_inputs):
            input_vars = {a: v for a, v in zip(scope.arguments, scope_inputs)}
        elif len(scope.arguments) > 0:
            input_vars = {
                a: (scope.name,) + tuple(a.split("::")[1:])
                for a in scope.arguments
            }

        for stmt in scope.body:
            func_def = stmt["function"]
            func_type = func_def["type"]
            if func_type == "lambda":
                process_wiring_statement(stmt, scope, input_vars, scope.name)
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
            return_list.append(make_variable_name(scope.name, basename, idx))

        for var_name in scope.updated:
            (_, basename, idx) = var_name.split("::")
            updated_list.append(make_variable_name(scope.name, basename, idx))
        return return_list, updated_list

    occurrences[con_name] = 0
    cur_scope = GenericSubgraph.create_scope(
        containers[con_name], occurrences[con_name]
    )
    returns, updates = process_container(cur_scope.name, [])
    return GroundedFunctionNetwork(G, scope_tree, returns + updates)
