import os
import re
import sys
import importlib
from pathlib import Path
from typing import Set, Dict
from abc import ABC, abstractmethod

from delphi.translators.for2py import f2grfn
from delphi.GrFN.networks import GroundedFactorNetwork
from delphi.GrFN.structures import (
    GenericContainer,
    GenericStmt,
    GenericIdentifier,
    GenericDefinition,
)
from delphi.GrFN.code_types import CodeType, build_code_type_decision_tree


class SourceInterpreter(ABC):
    def __init__(self, C: Dict, V: Dict, T: Dict, D: Dict):
        self.containers = C
        self.variables = V
        self.types = T
        self.documentation = D
        self.decision_tree = build_code_type_decision_tree()

    @classmethod
    @abstractmethod
    def from_src_file(cls, filepath):
        pass

    @classmethod
    @abstractmethod
    def from_src_dir(cls, dirpath):
        pass

    @staticmethod
    @abstractmethod
    def interp_file_IR(filepath):
        pass


class ImperativeInterpreter(SourceInterpreter):
    def __init__(self, C, V, T, D):
        super().__init__(C, V, T, D)

    @classmethod
    def from_src_file(cls, file):
        if not (file.endswith(".for") or file.endswith(".f")):
            raise ValueError(f"Unsupported file type ending for: {file}")

        (C, V, T, D) = cls.interp_file_IR(file)
        return cls(C, V, T, D)

    @classmethod
    def from_src_dir(cls, dirpath):
        src_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dirpath)
            for file in files
            if file.endswith(".for") or file.endswith(".f")
        ]

        C, V, T, D = {}, {}, {}, {}
        for src_path in src_paths:
            (C_new, V_new, T_new, D_new) = cls.interp_file_IR(src_path)
            C.update(C_new)
            V.update(V_new)
            T.update(T_new)
            D.update(D_new)

        return cls(C, V, T, D)

    @staticmethod
    def interp_file_IR(fortran_file):
        (
            python_sources,
            translated_python_files,
            mod_mapper_dict,
            fortran_filename,
            module_log_file_path,
            processing_modules,
        ) = f2grfn.fortran_to_grfn(fortran_file, save_intermediate_files=True,)
        python_file = translated_python_files[0]
        lambdas_path = python_file.replace(".py", "_lambdas.py")
        ir_dict = f2grfn.generate_grfn(
            python_sources[0][0],
            python_file,
            lambdas_path,
            mod_mapper_dict[0],
            fortran_file,
            module_log_file_path,
            processing_modules,
        )

        C = dict()
        for con_data in ir_dict["containers"]:
            new_container = GenericContainer.from_dict(con_data)
            C[new_container.identifier] = new_container

        V = dict()
        for var_data in ir_dict["variables"]:
            new_var = GenericDefinition.from_dict(var_data)
            V[new_var.identifier] = new_var

        T = dict()
        for type_data in ir_dict["types"]:
            new_type = GenericDefinition.from_dict(type_data)
            T[new_type.identifier] = new_type

        filename = ir_dict["source"][0]

        # TODO Paul - is it fine to switch from keying by filename to keying by
        # container name? Also, lowercasing? - Adarsh
        container_name = Path(filename).stem.lower()
        D = {
            n if not n.startswith("$") else container_name + n: data
            for n, data in ir_dict["source_comments"].items()
        }

        return C, V, T, D

    def __find_max_call_depth(self, depth, container, visited: Set[str]):
        # TODO Adarsh: implement this
        # NOTE: use the visited list to avoid an infinite loop

        for stmt in container["body"]:
            function = stmt["function"]
            if (
                function["type"] in ("container", "function")
                and function["name"] not in visited
            ):
                visited.add(function["name"])
                depth = self.__find_max_call_depth(
                    depth + 1, self.containers[function["name"]], visited
                )

        return depth

    def __find_max_cond_depth(self, depth, curr_con):
        # NOTE: @Adarsh you can hold off on implementing this
        return NotImplemented

    def __find_max_loop_depth(self, depth, curr_con):
        # NOTE: @Adarsh you can hold off on implementing this
        return NotImplemented

    def __process_container_stmt_stats(self, stmt, con_name):
        """
        Processes through a container call statement gathering stats for the
        container referenced by con_name.
        """
        # TODO Adarsh: this may need some debugging
        child_con_name = stmt["function"]["name"]

        child_con = self.containers[child_con_name]
        child_con_type = child_con["type"]
        if child_con_type in ("container", "function"):
            self.container_stats[con_name]["num_calls"] += 1
            visited = {child_con_name}
            temp = self.__find_max_call_depth(1, child_con, visited)
            if temp >= self.container_stats[con_name]["max_call_depth"]:
                self.container_stats[con_name]["max_call_depth"] = temp
        elif child_con_type == "if-block":
            self.container_stats[con_name]["num_conditionals"] += 1
            temp = self.__find_max_cond_depth(1, child_con)
            # if temp >= self.container_stats[con_name]["max_conditional_depth"]:
            # self.container_stats[con_name]["max_conditional_depth"] = temp
        elif child_con_type == "select-block":
            self.container_stats[con_name]["num_switches"] += 1
        elif child_con_type == "loop":
            self.container_stats[con_name]["num_loops"] += 1
            temp = self.__find_max_loop_depth(1, child_con)
            if temp >= self.container_stats[con_name]["max_loop_depth"]:
                self.container_stats[con_name]["max_loop_depth"] = temp
        else:
            raise ValueError(f"Unidentified container type: {child_con_type}")

    def __is_data_access(lambda_str):
        """
        Returns true if this lambda represents a data access, false otherwise.
        Common Fortran pattern of data access to search for:
        some_var = some_struct % some_attr
        NOTE: regex for the "%" on the RHS of the "="
        """
        # TODO Adarsh: implement this
        return NotImplemented

    def __is_math_assg(lambda_str):
        """
        Returns true if any math operator func is found, false otherwise

        NOTE: need to consider refining to deal with unary minus and divison
        operators as sometimes being constant creation instead of a math op
        """
        # TODO Adarsh: debug this
        rhs_lambda = lambda_str[lambda_str.find("=") + 1 :]
        math_ops = r"\+|-|/|\*\*|\*|%"
        math_funcs = (
            r"np\.maximum|np\.minimum|np\.exp|np\.log|np\.sqrt|np\.log10"
        )
        trig_funcs = (
            r"np\.sin|np\.cos|np\.tan|np\.arccos|np\.arcsin|np\.arctan"
        )
        math_search = re.search(math_ops, rhs_lambda)
        if math_search is not None:
            return True

        func_search = re.search(math_funcs, rhs_lambda)
        if func_search is not None:
            return True

        trig_search = re.search(trig_funcs, rhs_lambda)
        if trig_search is not None:
            return True

        return False

    def __process_lambda_stmt_stats(self, stmt, con_name):
        # TODO finish this implementation
        self.container_stats[con_name]["num_assgs"] += 1
        lambda_name = stmt["function"]["name"]
        lambdas_dir = str(lambda_path.parent.resolve())
        if lambdas_dir not in sys.path:
            sys.path.insert(0, lambdas_dir)
        lambdas = importlib.import_module(lambda_path.stem)
        # NOTE: use inspect.getsource(<lambda-ref>) in order to get the string source
        # NOTE: We need to search for:
        #   (1) assignment vs condition
        #   (2) accessor assignment vs math assignment
        #   (3) data change assignment vs regular math assignment
        return NotImplemented

    def gather_container_stats(self):
        """
        Analysis code that gathers container statistics used to determine the
        code-type of this container.
        """
        for con_name, con_data in self.containers.items():
            for stmt in con_data["body"]:
                # TODO Paul/Adarsh - extend the below to deal with statements that don't
                # have the 'function' key - e.g. ones that have 'condition' as
                # a key.
                if stmt.get("function") is not None:
                    stmt_type = stmt["function"]["type"]
                    if stmt_type == "container":
                        self.__process_container_stmt_stats(stmt, con_name)
                    elif stmt_type == "lambda":
                        self.__process_lambda_stmt_stats(stmt, con_name)
                    else:
                        raise ValueError(
                            f"Unidentified statement type: {stmt_type}"
                        )

    def label_container_code_type(self, current_node, stats):
        G = self.decision_tree
        satisfied = G.nodes[current_node]["func"](stats)
        for successor in G.successors(current_node):
            if G.get_edge_data(current_node, successor)["type"] == satisfied:
                label = (
                    G.nodes[successor]["type"]
                    if G.nodes[successor]["type"] != "condition"
                    else self.label_container_code_type(successor, stats)
                )

        return label

    def label_container_code_types(self):
        # TODO Adarsh: Implement the code-type decision tree here
        root = "C0"
        for container, stats in self.container_stats.items():
            self.container_code_types[
                container
            ] = self.label_container_code_type(root, stats)

    def build_GrFNs(self):
        """
        Creates the GrFNs for each container that has been determined to be
        represent a scientific model.
        """
        return {
            name: GroundedFactorNetwork.from_AIR(
                name, self.containers, self.variables, self.types,
            )
            for name in self.containers.keys()
            if self.container_code_types[name] is CodeType.MODEL
        }
