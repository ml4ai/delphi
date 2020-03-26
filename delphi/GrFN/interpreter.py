import os
import re
import importlib
import enum as enum
from pathlib import Path
from abc import ABC, abstractmethod

from delphi.translators.for2py import f2grfn


@enum.unique
class CodeType(enum.Enum):
    ACCESSOR = enum.auto()
    CALCULATION = enum.auto()
    CONVERSION = enum.auto()
    FILEIO = enum.auto()
    HELPER = enum.auto()
    LOGGING = enum.auto()
    MODEL = enum.auto()
    PIPELINE = enum.auto()
    UNKNOWN = enum.auto()


class SourceInterpreter(ABC):
    def __init__(self, L, C, V, T, D):
        self.lambda_paths = L
        self.containers = C
        self.variables = V
        self.types = T
        self.documentation = D
        self.container_code_types = {
            name: CodeType.UNKNOWN for name in self.containers
        }
        self.container_stats = {
            con_name: {
                "num_calls": 0,
                "max_call_depth": 0,
                "num_math_assgs": 0,
                "num_data_change_math_assgs": 0,
                "num_simple_assgs": 0,
                "num_assgs": 0,
                "num_switches": 0,
                "num_loops": 0,
                "max_loop_depth": 0,
                "num_conditionals": 0,
                "max_conditional_depth": 0,
            }
            for con_name in self.containers
        }

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
    def extract_IR(filepath):
        pass


class ImperativeInterpreter(SourceInterpreter):
    def __init__(self, L, C, V, T, D):
        super().__init__(L, C, V, T, D)

    @classmethod
    def from_src_file(cls, file):
        if not (file.endswith(".for") or file.endswith(".f")):
            raise ValueError("Unsupported file type ending for: {file}")

        (C, V, T, D) = cls.extract_IR(file)
        return cls(C, V, T, D)

    @classmethod
    def from_src_dir(cls, dirpath):
        src_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dirpath)
            for file in files
            if file.endswith(".for") or file.endswith(".f")
        ]

        L, C, V, T, D = {}, {}, {}, {}, {}
        for src_path in src_paths:
            (L_new, C_new, V_new, T_new, D_new) = cls.extract_IR(src_path)
            L.update(L_new)
            C.update(C_new)
            V.update(V_new)
            T.update(T_new)
            D.update(D_new)

        return cls(L, C, V, T, D)

    @staticmethod
    def extract_IR(fortran_file):
        (
            python_sources,
            translated_python_files,
            mod_mapper_dict,
            fortran_filename,
            module_log_file_path,
            processing_modules,
        ) = f2grfn.fortran_to_grfn(fortran_file, processing_modules=False,)
        python_file = translated_python_files[0]
        lambdas_path = python_file.replace(".py", "_lambdas.py")
        ir_dict = f2grfn.generate_grfn(
            python_sources[0][0],
            python_file,
            lambdas_path,
            mod_mapper_dict,
            fortran_file,
            module_log_file_path,
            processing_modules,
        )

        C = {c["name"]: c for c in ir_dict["containers"]}
        V = {v["name"]: v for v in ir_dict["variables"]}
        T = {t["name"]: t for t in ir_dict["types"]}

        fname = ir_dict["source"][0]
        L = {fname: lambdas_path}
        D = {
            n if not n.startswith("$") else fname + n: data
            for n, data in ir_dict["source_comments"].items()
        }

        return L, C, V, T, D

    def extract_GrFN(self, con_data, lambdas):
        return NotImplemented

    def get_container_lambdas(self, container_name):
        (_, namespace, _, _) = container_name.split("::")
        return self.lambda_paths[namespace]

    def __find_max_call_depth(self, depth, curr_con, visited):
        return NotImplemented

    def __find_max_cond_depth(self, depth, curr_con):
        return NotImplemented

    def __find_max_loop_depth(self, depth, curr_con):
        return NotImplemented

    def __process_container_stmt_stats(self, stmt, con_name):
        child_con_name = stmt["function"]["name"]
        child_con = self.containers[child_con_name]
        child_con_type = child_con["type"]
        if child_con_type == "container" or child_con_type == "function":
            self.container_stats[con_name]["num_calls"] += 1
            called = [child_con_name]
            temp = self.__find_max_call_depth(1, child_con, called)
            if temp >= self.container_stats[con_name]["max_call_depth"]:
                self.container_stats[con_name]["max_call_depth"] = temp
        elif child_con_type == "if-block":
            self.container_stats[con_name]["num_conditionals"] += 1
            temp = self.__find_max_cond_depth(1, child_con)
            if temp >= self.container_stats[con_name]["max_conditional_depth"]:
                self.container_stats[con_name]["max_conditional_depth"] = temp
        elif child_con_type == "select-block":
            self.container_stats[con_name]["num_switches"] += 1
        elif child_con_type == "loop":
            self.container_stats[con_name]["num_loops"] += 1
            temp = self.__find_max_loop_depth(1, child_con)
            if temp >= self.container_stats[con_name]["max_loop_depth"]:
                self.container_stats[con_name]["max_loop_depth"] = temp
        else:
            raise ValueError(f"Unidentified container type: {child_con_type}")

    def __has_math_op(lambda_str):
        math_ops = r"\+|-|/|\*\*|\*|%"
        math_funcs = (
            r"np\.maximum|np\.minimum|np\.exp|np\.log|np\.sqrt|np\.log10"
        )
        trig_funcs = (
            r"np\.sin|np\.cos|np\.tan|np\.arccos|np\.arcsin|np\.arctan"
        )
        math_search = re.search(math_ops, lambda_str)
        if math_search is not None:
            return True

        func_search = re.search(math_funcs, lambda_str)
        if func_search is not None:
            return True

        trig_search = re.search(trig_funcs, lambda_str)
        if trig_search is not None:
            return True

        return False

    def __process_lambda_stmt_stats(self, stmt, con_name):
        self.container_stats[con_name]["num_assgs"] += 1
        lambda_name = stmt["function"]["name"]
        lambda_path = self.get_container_lambdas(con_name)
        lambdas = importlib.__import__(str(Path(lambda_path).stem))
        return NotImplemented

    def gather_container_stats(self):
        """
        Analysis code that gathers container statistics used to determine the
        code-type of this container.
        """
        for con_name, con_data in self.containers:
            for stmt in con_data["body"]:
                stmt_type = stmt["function"]["type"]
                if stmt_type == "container":
                    self.__process_container_stmt_stats(stmt, con_name)
                elif stmt_type == "lambda":
                    self.__process_lambda_stmt_stats(stmt, con_name)
                else:
                    raise ValueError(
                        f"Unidentified statement type: {stmt_type}"
                    )

    def label_container_code_types(self):
        return NotImplemented

    def build_GrFNs(self):
        grfn_containers = {
            n: d
            for n, d in self.containers.items()
            if self.container_code_types[n] is CodeType.MODEL
        }

        GrFNs = list()
        for con_name, con_data in grfn_containers.items():
            lambda_path = self.get_container_lambdas(con_name)
            G = self.extract_GrFN(con_data, lambda_path)
            GrFNs.append(G)
        return GrFNs
