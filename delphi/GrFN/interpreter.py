import os
import re
import sys
import importlib
import enum as enum
from pathlib import Path
from abc import ABC, abstractmethod

from delphi.translators.for2py import f2grfn
from delphi.GrFN.extraction import extract_GrFN


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
    def __init__(self, L: dict, C: dict, V: dict, T: dict, D: dict):
        self.container_lambdas_map = L
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
                "num_data_changes": 0,
                "num_var_access": 0,
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

        filename = ir_dict["source"][0]

        # TODO Paul - is it fine to switch from keying by filename to keying by
        # container name? Also, lowercasing? - Adarsh
        container_name = Path(filename).stem.lower()

        L = {container_name: lambdas_path}
        D = {
            n if not n.startswith("$") else container_name + n: data
            for n, data in ir_dict["source_comments"].items()
        }

        return L, C, V, T, D

    def get_container_lambdas(self, container_name: str):
        # When for2py analyzes modules, the produced lambda files have
        # lowercase filenames, e.g.
        #
        #    {'mini_ModuleDefs.for': './tmp/m_mini_moduledefs_lambdas.py'}
        #
        # This makes it difficult to automatically associate namespaces and
        # m_*_lambdas.py files. To deal with it, we lowercase the namespace in
        # this function - but I'm not sure whether this is robust or a hack.
        #
        # - Adarsh
        #
        # TODO Terrence/Pratik: Can you let us know if the lowercasing is necessary?

        namespace = container_name.split("::")[1].lower()
        return self.container_lambdas_map[namespace]

    def __find_max_call_depth(self, depth, curr_con, visited):
        # TODO Adarsh: implement this
        # NOTE: use the visited list to avoid an infinite loop
        return NotImplemented

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
        # TODO Adarsh: finish implementing this.
        self.container_stats[con_name]["num_assgs"] += 1
        lambda_name = stmt["function"]["name"]
        lambda_path = Path(self.get_container_lambdas(con_name))
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
        # TODO Adarsh: Implement the code-type decision tree here
        return NotImplemented

    def build_GrFNs(self):
        """
        Creates the GrFNs for each container that has been determined to be
        represent a scientific model.
        """
        return {
            name: extract_GrFN(
                name,
                self.containers,
                self.variables,
                self.container_lambdas_map,
            )
            for name in self.containers.keys()
            if self.container_code_types[name] is CodeType.MODEL
        }
