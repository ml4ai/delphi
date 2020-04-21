from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from types import ModuleType

from delphi.GrFN.networks import (
    GenericNetwork,
    CondNetwork,
    FuncNetwork,
    LoopNetwork,
)
from delphi.GrFN.code_types import CodeType


class GenericContainer(ABC):
    def __init__(self, name: str, data: dict):
        self.name = name
        self.arguments = data["arguments"]
        self.updated = data["updated"]
        self.returns = data["return_value"]

        # NOTE: store base name as key and update index during wiring
        self.variables = dict()
        self.code_type = CodeType.UNKNOWN
        self.code_stats = {
            "num_calls": 0,
            "max_call_depth": 0,
            "num_math_assgs": 0,
            "num_data_changes": 0,
            "num_var_access": 0,
            "num_assgs": 0,
            "num_switches": 0,
            "num_loops": 0,
            "max_loop_depth": 0,
            "num_conditionals": 0,
            "max_conditional_depth": 0,
        }

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self, other):
        args_str = "\n".join([f"\t{arg}" for arg in self.arguments])
        vars_str = "\n".join(
            [f"\t{k} -> {v}" for k, v in self.variables.items()]
        )
        return f"Inputs:\n{args_str}\nVariables:\n{vars_str}"

    def dump_containers(self):
        print(self)
        if self.parent is not None:
            if not isinstance(self.parent, GenericContainer):
                raise TypeError(
                    f"Unrecognized container node parent: {type(self.parent)}"
                )
            self.parent.dump_containers()

    @staticmethod
    def create_container(data: dict):
        con_type = data["type"]
        if con_type == "function":
            return FuncContainer(data)
        elif con_type == "loop":
            return LoopContainer(data)
        elif con_type == "if-block":
            return CondContainer(data)
        elif con_type == "select-block":
            return CondContainer(data)
        else:
            raise ValueError(f"Unrecognized container type value: {con_type}")

    @staticmethod
    def create_statement_list(stmts: list):
        return [GenericStmt.create_statement(stmt) for stmt in stmts]

    @abstractmethod
    def translate(
        self, call_inputs: list, containers: dict, occurrences: dict
    ) -> GenericNetwork:
        return NotImplemented


class CondContainer(GenericContainer):
    def __init__(self, data: dict):
        name = data["name"]
        (_, namespace, sub_name, con_name) = name.split("::")
        n = f"{namespace}::{sub_name}::{con_name}"
        super().__init__(n, data)

        self.guarded_statements = [
            {
                "guard": GenericContainer.create_statement_list(
                    blocks["condition"]
                ),
                "statments": GenericContainer.create_statement_list(
                    blocks["statements"]
                ),
            }
            for blocks in data["body"]
        ]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<COND Con> -- {self.name}\n{base_str}\n"

    def translate(
        self, call_inputs: list, containers: dict, occurrences: dict
    ) -> CondNetwork:
        return NotImplemented


class FuncContainer(GenericContainer):
    def __init__(self, data: dict):
        name = data["name"]
        (_, namespace, scope, sub_name) = name.split("::")
        n = f"{namespace}::{scope}::{sub_name}"
        super().__init__(n, data)
        self.statement_list = GenericContainer.create_statement_list(
            data["body"]
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<FUNC Con> -- {self.name}\n{base_str}\n"

    def translate(
        self, call_inputs: list, containers: dict, occurrences: dict
    ) -> FuncNetwork:
        if self.name not in occurrences:
            occurrences[self.name] = 0

        network_idx = occurrences[self.name]
        new_network = FuncNetwork(self.name, network_idx, parent=None)


class LoopContainer(GenericContainer):
    def __init__(self, data: dict):
        name = data["name"]
        (_, namespace, sub_name, loop_name) = name.split("::")
        n = f"{namespace}::{sub_name}::{loop_name}"
        super().__init__(n, data)

        self.statement_list = GenericContainer.create_statement_list(
            data["body"]
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<LOOP Con> -- {self.name}\n{base_str}\n"

    def translate(
        self, call_inputs: list, containers: dict, occurrences: dict
    ) -> LoopNetwork:
        if len(self.arguments) == len(call_inputs):
            input_vars = {a: v for a, v in zip(self.arguments, call_inputs)}
        elif len(self.arguments) > 0:
            input_vars = {
                a: (self.name,) + tuple(a.split("::")[1:])
                for a in self.arguments
            }

        for stmt in self.statement_list:
            # TODO: pickup translation here
            stmt.translate(self, input_vars)
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


@unique
class LambdaType(Enum):
    ASSIGN = auto()
    LITERAL = auto()
    CONDITION = auto()
    DECISION = auto()

    def __str__(self):
        return str(self.name)

    def shortname(self):
        return self.__str__()[0]

    @classmethod
    def get_lambda_type(cls, lambda_name: str, num_inputs: int):
        type_str = lambda_name.split("__")[-3]
        if type_str == "assign":
            if num_inputs == 0:
                return cls.LITERAL
            return cls.ASSIGN
        elif type_str == "condition":
            return cls.CONDITION
        elif type_str == "decision":
            return cls.DECISION
        else:
            raise ValueError(f"Unrecognized lambda type name: {type_str}")


class GenericStmt(ABC):
    def __init__(
        self, n: str, inputs: list, outputs: list, p: GenericContainer
    ):
        self.name = n
        self.container = p

        self.inputs = [GenericStmt.__get_variable_data(i) for i in inputs]
        self.outputs = [GenericStmt.__get_variable_data(o) for o in outputs]

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        inputs_str = "\n\t".join(
            [f'{i["basename"]}::{i["index"]}' for i in self.inputs]
        )
        outputs_str = "\n\t".join(
            [f'{i["basename"]}::{i["index"]}' for i in self.outputs]
        )
        return f"Inputs:\n{inputs_str}\nOutputs:\n{outputs_str}"

    @staticmethod
    def create_statement(stmt_data: dict, container: GenericContainer):
        func_type = stmt_data["function"]["type"]
        if func_type == "lambda":
            return LambdaStmt(stmt_data, container)
        elif func_type == "container":
            return CallStmt(stmt_data, container)
        else:
            raise ValueError(f"Undefined statement type: {func_type}")

    @staticmethod
    def __get_variable_data(var_repr: str) -> dict:
        (_, var_name, idx) = var_repr.split("::")
        return {"basename": var_name, "index": int(idx)}

    # def make_variable_name(self):
    #     return f"{parent}::{basename}::{index}"


def CallStmt(GenericStmt):
    def __init__(self, stmt: dict, con: GenericContainer):
        super().__init__(
            stmt["function"]["name"],
            stmt["input"],
            stmt["output"] + stmt["updated"],
            con,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        generic_str = super().__str__()
        return f"<CallStmt>: {self.name}\n{generic_str}"

    def translate(self, containers: dict, occurrences: dict) -> GenericNetwork:
        new_con_name = self.name
        if new_con_name not in occurrences:
            occurrences[new_con_name] = 0

        call_inputs = list()
        for in_var in self.inputs:
            if in_var["index"] == -1:
                pass
            else:
                pass

        new_container = containers[new_con_name]
        new_container.translate(call_inputs, containers, occurrences)

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


def LambdaStmt(GenericStmt):
    def __init__(self, stmt: dict, con: GenericContainer):
        super().__init__(
            stmt["function"]["name"], stmt["input"], stmt["output"], con
        )
        self.lambda_node_name = f"{self.parent.name}::" + self.name
        self.type = LambdaType.get_lambda_type(self.name, len(self.inputs))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        generic_str = super().__str__()
        return f"<LambdaStmt>: {self.name}\n{generic_str}"

    def translate(
        self, inputs: dict, lambdas: ModuleType, network: GenericNetwork
    ):
        corrected_inputs = [
            inp if inp["index"] != "-1" else inputs[inp["basename"]]
            for inp in self.inputs
        ]
        in_ids = [network.add_variable_node(var) for var in corrected_inputs]
        out_ids = [network.add_variable_node(var) for var in self.outputs]

        fn = getattr(lambdas, self.lambda_node_name)
        lambda_id = network.add_lambda_node(self.type, fn, self.inputs)
        network.add_hyper_edge(in_ids, lambda_id, out_ids)
