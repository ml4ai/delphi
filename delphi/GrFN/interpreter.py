import os
import re
import sys
import importlib
import enum as enum
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Set, Dict
from delphi.translators.for2py import f2grfn
from delphi.GrFN.extraction import extract_GrFN
from inspect import currentframe, getframeinfo


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
    def __init__(self, L: Dict[str, str], C: Dict, V: Dict, T: Dict, D: Dict):
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
        import networkx as nx

        G = nx.DiGraph()
        G.add_node(
            "C0",
            type="condition",
            func=lambda d: d["num_switches"] >= 1,
            shape="rectangle",
            label="num_switches >= 1",
        )
        G.add_node(
            "C1",
            type="condition",
            func=lambda d: d["max_call_depth"] <= 2,
            shape="rectangle",
            label="max_call_depth <= 2",
        )
        G.add_node(
            "C2",
            type="condition",
            func=lambda d: d["num_assgs"] >= 1,
            shape="rectangle",
            label="num_assgs >= 1",
        )
        G.add_node(
            "C3",
            type="condition",
            func=lambda d: d["num_math_assgs"] >= 1,
            shape="rectangle",
            label="num_math_assgs >= 1",
        )
        G.add_node(
            "C4",
            type="condition",
            func=lambda d: d["num_data_changes"] >= 1,
            shape="rectangle",
            label="num_data_changes >= 1",
        )
        G.add_node(
            "C5",
            type="condition",
            func=lambda d: d["num_math_assgs"] >= 5,
            shape="rectangle",
            label="num_math_assgs >= 5",
        )
        G.add_node(
            "C6",
            type="condition",
            func=lambda d: d["num_var_access"] >= 1,
            shape="rectangle",
            label="num_var_access >= 1",
        )
        G.add_node("Accessor", type=CodeType.ACCESSOR, color="blue")
        G.add_node("Calculation", type=CodeType.CALCULATION, color="blue")
        G.add_node("Conversion", type=CodeType.CONVERSION, color="blue")
        G.add_node("File I/O", type=CodeType.FILEIO, color="blue")
        # G.add_node("Helper", type=CodeType.HELPER, color='blue')
        # G.add_node("Logging", type=CodeType.LOGGING, color='blue')
        G.add_node("Model", type=CodeType.MODEL, color="blue")
        G.add_node("Pipeline", type=CodeType.PIPELINE, color="blue")
        G.add_node("Unknown", type=CodeType.UNKNOWN, color="blue")

        G.add_edge("C0", "Pipeline", type=True, color="darkgreen")
        G.add_edge("C0", "C1", type=False, color="red")

        G.add_edge("C1", "C2", type=True, color="darkgreen")
        G.add_edge("C1", "Pipeline", type=False, color="red")

        G.add_edge("C2", "File I/O", type=False, color="red")
        G.add_edge("C2", "C3", type=True, color="darkgreen")

        G.add_edge("C3", "C4", type=True, color="darkgreen")
        G.add_edge("C3", "C5", type=False, color="red")

        G.add_edge("C4", "C6", type=True, color="darkgreen")
        G.add_edge("C4", "Conversion", type=False, color="red")

        G.add_edge("C5", "Accessor", type=True, color="darkgreen")
        G.add_edge("C5", "Unknown", type=False, color="red")

        G.add_edge("C6", "Model", type=True, color="darkgreen")
        G.add_edge("C6", "Calculation", type=False, color="red")

        # A = nx.nx_agraph.to_agraph(G)
        # A.draw('decision_tree.pdf', prog='dot')
        self.decision_tree = G

    @classmethod
    def from_src_file(cls, file):
        if not (file.endswith(".for") or file.endswith(".f")):
            raise ValueError("Unsupported file type ending for: {file}")

        (L, C, V, T, D) = cls.extract_IR(file)
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

        # TODO Paul The line below is a hack - the child_con_name should be a
        # key in self.containers.
        if self.containers.get(child_con_name) is not None:
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
                raise ValueError(
                    f"Unidentified container type: {child_con_type}"
                )

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
            name: extract_GrFN(
                name,
                self.containers,
                self.variables,
                self.container_lambdas_map,
            )
            for name in self.containers.keys()
            if self.container_code_types[name] is CodeType.MODEL
        }