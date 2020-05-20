from abc import ABC, abstractmethod
from enum import Enum, auto, unique
from dataclasses import dataclass
from types import ModuleType

# from delphi.GrFN.networks import (
#     GroundedFactorNetwork,
#     GrFNSubgraph,
# )
from delphi.GrFN.code_types import CodeType


@dataclass(repr=False, frozen=True)
class GenericIdentifier(ABC):
    namespace: str
    scope: str

    @staticmethod
    def from_str(data: str):
        components = data.split("::")
        type_str = components[0]
        if type_str == "@container":
            (_, ns, sc, n) = components
            return ContainerIdentifier(ns, sc, n)
        elif type_str == "@type":
            (_, ns, sc, n) = components
            return TypeIdentifier(ns, sc, n)
        elif type_str == "@variable":
            (_, ns, sc, n, idx) = components
            return VariableIdentifier(ns, sc, n, int(idx))

    def is_global_scope(self):
        return self.scope == "@global"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        return NotImplemented


@dataclass(repr=False, frozen=True)
class ContainerIdentifier(GenericIdentifier):
    con_name: str

    def __str__(self):
        return f"Con -- {self.con_name} ({self.namespace}.{self.scope})"


@dataclass(repr=False, frozen=True)
class TypeIdentifier(GenericIdentifier):
    type_name: str

    def __str__(self):
        return f"Type -- {self.type_name} ({self.namespace}.{self.scope})"


@dataclass(repr=False, frozen=True)
class VariableIdentifier(GenericIdentifier):
    var_name: str
    index: int

    @classmethod
    def from_str_and_con(cls, data: str, con: ContainerIdentifier):
        (_, name, idx) = data.split("::")
        return cls(con.namespace, con.scope, name, int(idx))

    def __str__(self):
        var_str = f"{self.var_name}::{self.index}"
        return f"Var -- {var_str} ({self.namespace}.{self.scope})"


@dataclass(frozen=True)
class GenericDefinition(ABC):
    identifier: GenericIdentifier
    type: str

    @staticmethod
    def from_dict(data: dict):
        print(data)
        if "domain" in data:
            if "dimensions" in data["domain"]:
                type_str = "type"
                name_str = "array"
            else:
                name_str = data["domain"]["name"]
                type_str = data["domain"]["type"]
            return VariableDefinition(
                GenericIdentifier.from_str(data["name"]),
                type_str,
                data["domain"]["mutable"],
                name_str,
                data["domain_constraint"],
                tuple(data["source_refs"]),
            )
        else:
            return TypeDefinition(
                GenericIdentifier.from_str(data["name"]),
                data["type"],
                tuple(data["attributes"]),
            )


@dataclass(frozen=True)
class VariableDefinition(GenericDefinition):
    is_mutable: bool
    domain_name: str
    domain_constraint: str
    source_refs: tuple


@dataclass(frozen=True)
class TypeDefinition(GenericDefinition):
    attributes: tuple


class GenericContainer(ABC):
    def __init__(self, data: dict):
        self.identifier = GenericIdentifier.from_str(data["name"])
        self.arguments = [
            VariableIdentifier.from_str_and_con(var_str, self.identifier)
            for var_str in data["arguments"]
        ]
        self.updated = [
            VariableIdentifier.from_str_and_con(var_str, self.identifier)
            for var_str in data["updated"]
        ]
        self.returns = [
            VariableIdentifier.from_str_and_con(var_str, self.identifier)
            for var_str in data["return_value"]
        ]
        self.statements = [
            GenericStmt.create_statement(stmt, self) for stmt in data["body"]
        ]

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
    def __str__(self):
        args_str = "\n".join([f"\t{arg}" for arg in self.arguments])
        outputs_str = "\n".join([f"\t{var}" for var in self.returns + self.updated])
        return f"Inputs:\n{args_str}\nVariables:\n{outputs_str}"

    @staticmethod
    def from_dict(data: dict):
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

    # @abstractmethod
    # def translate(
    #     self, call_inputs: list, containers: dict, occurrences: dict
    # ) -> GrFNSubgraph:
    #     return NotImplemented


class CondContainer(GenericContainer):
    def __init__(self, data: dict):
        super().__init__(data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<COND Con> -- {self.identifier.con_name}\n{base_str}\n"

    # def translate(
    #     self, call_inputs: list, containers: dict, occurrences: dict
    # ) -> GrFNSubgraph:
    #     return NotImplemented


class FuncContainer(GenericContainer):
    def __init__(self, data: dict):
        super().__init__(data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<FUNC Con> -- {self.identifier.con_name}\n{base_str}\n"

    # def translate(
    #     self, call_inputs: list, containers: dict, occurrences: dict
    # ) -> GrFNSubgraph:
    #     if self.name not in occurrences:
    #         occurrences[self.name] = 0
    #
    #     network_idx = occurrences[self.name]
    #     new_network = GrFNSubgraph(self.name, network_idx, parent=None)


class LoopContainer(GenericContainer):
    def __init__(self, data: dict):
        super().__init__(data)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        base_str = super().__str__()
        return f"<LOOP Con> -- {self.identifier.con_name}\n{base_str}\n"

    # def translate(
    #     self, call_inputs: list, containers: dict, occurrences: dict
    # ) -> GrFNSubgraph:
    #     if len(self.arguments) == len(call_inputs):
    #         input_vars = {a: v for a, v in zip(self.arguments, call_inputs)}
    #     elif len(self.arguments) > 0:
    #         input_vars = {
    #             a: (self.name,) + tuple(a.split("::")[1:])
    #             for a in self.arguments
    #         }
    #
    #     for stmt in self.statement_list:
    #         # TODO: pickup translation here
    #         stmt.translate(self, input_vars)
    #         func_def = stmt["function"]
    #         func_type = func_def["type"]
    #         if func_type == "lambda":
    #             process_wiring_statement(stmt, scope, input_vars, scope.name)
    #         elif func_type == "container":
    #             process_call_statement(stmt, scope, input_vars, scope.name)
    #         else:
    #             raise ValueError(f"Undefined function type: {func_type}")
    #
    #     scope_tree.add_node(scope.name, color="forestgreen")
    #     if scope.parent is not None:
    #         scope_tree.add_edge(scope.parent.name, scope.name)
    #
    #     return_list, updated_list = list(), list()
    #     for var_name in scope.returns:
    #         (_, basename, idx) = var_name.split("::")
    #         return_list.append(make_variable_name(scope.name, basename, idx))
    #
    #     for var_name in scope.updated:
    #         (_, basename, idx) = var_name.split("::")
    #         updated_list.append(make_variable_name(scope.name, basename, idx))
    #     return return_list, updated_list


@unique
class LambdaType(Enum):
    ASSIGN = auto()
    LITERAL = auto()
    CONDITION = auto()
    DECISION = auto()
    PASS = auto()

    def __str__(self):
        return str(self.name)

    def shortname(self):
        return self.__str__()[0]

    @classmethod
    def get_lambda_type(cls, type_str: str, num_inputs: int):
        if type_str == "assign":
            if num_inputs == 0:
                return cls.LITERAL
            return cls.ASSIGN
        elif type_str == "condition":
            return cls.CONDITION
        elif type_str == "decision":
            return cls.DECISION
        elif type_str == "pass":
            return cls.PASS
        else:
            raise ValueError(f"Unrecognized lambda type name: {type_str}")


class GenericStmt(ABC):
    def __init__(self, stmt: dict, p: GenericContainer):
        self.container = p
        self.inputs = [
            VariableIdentifier.from_str_and_con(i, self.container.identifier)
            for i in stmt["input"]
        ]
        self.outputs = [
            VariableIdentifier.from_str_and_con(o, self.container.identifier)
            for o in (stmt["output"] + stmt["updated"])
        ]

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self):
        inputs_str = ", ".join([f"{id.var_name} ({id.index})" for id in self.inputs])
        outputs_str = ", ".join([f"{id.var_name} ({id.index})" for id in self.outputs])
        return f"Inputs: {inputs_str}\nOutputs: {outputs_str}"

    @staticmethod
    def create_statement(stmt_data: dict, container: GenericContainer):
        func_type = stmt_data["function"]["type"]
        if func_type == "lambda":
            return LambdaStmt(stmt_data, container)
        elif func_type == "container":
            return CallStmt(stmt_data, container)
        else:
            raise ValueError(f"Undefined statement type: {func_type}")

    # def correct_input_list(
    #     self, alt_inputs: Dict[VariableIdentifier, VariableNode]
    # ) -> List[VariableNode]:
    #     return [v if v.index != -1 else alt_inputs[v] for v in self.inputs]


class CallStmt(GenericStmt):
    def __init__(self, stmt: dict, con: GenericContainer):
        super().__init__(stmt, con)
        self.call_id = GenericIdentifier.from_str(stmt["function"]["name"])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        generic_str = super().__str__()
        return f"<CallStmt>: {self.call_id}\n{generic_str}"

    # def translate(
    #     self,
    #     container_inputs: dict,
    #     network: GroundedFactorNetwork,
    #     containers: dict,
    #     occurrences: dict,
    # ) -> None:
    #     new_container = containers[self.call_id]
    #     if self.call_id not in occurrences:
    #         occurrences[self.call_id] = 0
    #
    #     call_inputs = self.correct_input_list(container_inputs)
    #     outputs = new_container.translate(call_inputs, containers, occurrences)
    #
    #     out_var_names = [n.name for n in outputs]
    #     out_var_str = ",".join(out_var_names)
    #     pass_func_str = f"lambda {out_var_str}:({out_var_str})"
    #     func = network.add_lambda_node(LambdaType.PASS, pass_func_str, outputs)
    #
    #     out_nodes = [network.add_variable_node(var) for var in self.outputs]
    #     network.add_hyper_edge(outputs, func, out_nodes)
    #
    #     caller_ret, caller_up = list(), list()
    #
    #     occurrences[self.call_id] += 1


class LambdaStmt(GenericStmt):
    def __init__(self, stmt: dict, con: GenericContainer):
        super().__init__(stmt, con)
        # NOTE Want to use the form below eventually
        # type_str = stmt["function"]["lambda_type"]

        name_components = stmt["function"]["name"].split("__")
        type_str = name_components[-3]

        # NOTE: we shouldn't need this since we will use UUIDs
        # self.lambda_node_name = f"{self.parent.name}::" + self.name
        self.type = LambdaType.get_lambda_type(type_str, len(self.inputs))
        self.func_str = stmt["function"]["code"]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        generic_str = super().__str__()
        return f"<LambdaStmt>: {self.type}\n{generic_str}"

    # def translate(
    #     self, container_inputs: dict, network: GroundedFactorNetwork
    # ) -> None:
    #     corrected_inputs = self.correct_input_list(container_inputs)
    #     in_nodes = [network.add_variable_node(var) for var in corrected_inputs]
    #     out_nodes = [network.add_variable_node(var) for var in self.outputs]
    #     func = network.add_lambda_node(self.type, self.func_str, in_nodes)
    #     network.add_hyper_edge(in_nodes, func, out_nodes)
