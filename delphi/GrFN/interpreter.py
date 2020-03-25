import os
import json

from abc import ABCMeta, abstractmethod


class SourceInterpreter(ABCMeta):
    def __init__(self, C, V, T, D):
        self.containers = C
        self.variables = V
        self.types = T
        self.documentation = D

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
    def __init__(self, C, V, T, D):
        super.__init__(C, V, T, D)

    @classmethod
    def from_src_file(cls, filepath):
        (C, V, T, D) = cls.extract_IR(filepath)
        return cls(C, V, T, D)

    @classmethod
    def from_src_dir(cls, dirpath):
        src_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dirpath)
            for file in files
            if file.endswith("_GrFN.json")
        ]
        print(src_paths)

        C, V, T, D = {}, {}, {}, {}
        for src_path in src_paths:
            (C_new, V_new, T_new, D_new) = cls.extract_IR(src_path)
            C.update(C_new)
            V.update(V_new)
            T.update(T_new)
            D.update(D_new)

        return cls(C, V, T, D)

    @staticmethod
    def extract_IR(filepath):
        ir_json = json.load(open(filepath, "r"))
        C = {c["name"]: c for c in ir_json["containers"]}
        V = {v["name"]: v for v in ir_json["variables"]}
        T = {t["name"]: t for t in ir_json["types"]}

        fname = ir_json["source"][0]
        D = {
            n if not n.startswith("$") else fname + n: data
            for n, data in ir_json["source_comments"]
        }

        return C, V, T, D

    def get_container_stats(self):
        """
        I expect to find the following code types in Mini-PET
        Pipeline
        Scientific-Model
        Helper-Calculation
        Conversion-Calculation
        Data-access
        """
        num_calls = 0
        max_call_depth = 0
        num_math_assgs = 0
        num_data_change_math_assgs = 0
        num_simple_assgs = 0
        num_switches = 0
        num_loops = 0
        max_loop_depth = 0
        num_conditionals = 0
        max_conditional_depth = 0

        return NotImplemented

    def label_container_code_types(self):
        return NotImplemented

    def build_GrFNs(self, grfn_names_list):
        return NotImplemented
