import time

from SALib.sample import saltelli, fast_sampler, latin
import SALib as SAL
from SALib.analyze import sobol, fast, rbd_fast
import numpy as np
import torch

from delphi.GrFN.networks import ComputationalGraph
from delphi.GrFN.utils import timeit


class InputError(Exception):
    pass


# TODO khan: add documentation to this class (this should help you ensure you
# understand the code)
class SensitivityIndices(object):
    def __init__(self, S: dict):
        """
        Args:
            S: A SALib dictionary from analysis
        """
        self.O1_indices = np.array(S["S1"]) if "S1" in S else None
        self.O2_indices = np.array(S["S2"]) if "S2" in S else None
        self.OT_indices = np.array(S["ST"]) if "ST" in S else None
        self.O1_confidence = S["S1_conf"] if "S1_conf" in S else None
        self.O2_confidence = S["S2_conf"] if "S2_conf" in S else None
        self.OT_confidence = S["ST_conf"] if "ST_conf" in S else None

    def check_first_order(self):
        if self.O1_indices is None:
            raise ValueError("No first order indices present")

    def check_second_order(self):
        if self.O2_indices is None:
            raise ValueError("No second order indices present")

    def check_total_order(self):
        if self.OT_indices is None:
            raise ValueError("No total order indices present")

    @classmethod
    def from_csv(cls, filepath: str):
        # TODO khan: implement this so that Si_dict is a sensitivity index
        # dictionary loaded from the CSV file provided by filepath
        Si_dict = None
        return cls(Si_dict)

    @classmethod
    def from_dict(cls, Si: dict):
        """Creates a SensitivityIndices object from the provided dictionary."""
        return cls(Si)

    @classmethod
    def from_json(cls, filepath: str):
        # TODO khan: implement this so that Si_dict is a sensitivity index
        # dictionary loaded from the JSON file provided by filepath
        Si_dict = None
        return cls(Si_dict)

    @classmethod
    def from_pickle(cls, filepath: str):
        # TODO khan: implement this so that Si_dict is a sensitivity index
        # dictionary loaded from the PKL file provided by filepath
        Si_dict = None
        return cls(Si_dict)

    def get_min_S2(self):
        """Gets the value of the minimum S2 index."""
        self.check_second_order()
        return np.amin(self.O2_indices)

    def get_argmin_S2(self):
        """Gets the location of the minimum S2 index."""
        self.check_second_order()
        full_index = np.argmin(self.O2_indices)
        return np.unravel_index(full_index, self.O2_indices.shape)

    def get_max_S2(self):
        """Gets the value of the maximum S2 index."""
        self.check_second_order()
        return np.amax(self.O2_indices)

    def get_argmax_S2(self):
        """Gets the location of the maximum S2 index."""
        self.check_second_order()
        full_index = np.argmax(self.O2_indices)
        return np.unravel_index(full_index, self.O2_indices.shape)

    def to_csv(self, filepath: str):
        # TODO khan: Save the data in this class to a CSV file
        return NotImplemented

    def to_json(self, filepath: str):
        # TODO khan: Save the data in this class to a JSON file
        return NotImplemented

    def to_pickle(self, filepath: str):
        # TODO khan: Save the data in this class to a PKL file
        return NotImplemented


class SensitivityAnalyzer(object):
    def __init__(self):
        pass

    @staticmethod
    def setup_problem_def(input_vars, B):
        return {
            'num_vars': len(input_vars),
            'names': input_vars,
            'bounds': [B[var] for var in input_vars]
        }

    @staticmethod
    @timeit
    def __run_analysis(analyzer, *args, **kwargs):
        return analyzer(*args, **kwargs)

    @staticmethod
    @timeit
    def __run_sampling(sampler, *args, **kwargs):
        return sampler(*args, **kwargs)

    @staticmethod
    @timeit
    def __execute_CG(CG, samples, problem, C, V, *args, **kwargs):
        def create_input_vector(name, vector, var_types=None):
            if var_types is None:
                return vector

            type_info = var_types[name]
            if type_info[0] != str:
                return vector

            if type_info[0] == str:
                (str1, str2) = type_info[1]
                return np.where(vector >= 0.5, str1, str2)
            else:
                raise ValueError(f"Unrecognized value type: {type_info[0]}")

        # Create vectors of sample inputs to run through the model
        vectorized_sample_list = np.split(samples, samples.shape[1], axis=1)
        vectorized_input_samples = {
            name: create_input_vector(name, vector, var_types=V)
            for name, vector in zip(problem["names"], vectorized_sample_list)
        }

        outputs = CG.run(vectorized_input_samples)
        Y = outputs[0]
        Y = Y.reshape((Y.shape[0],))
        return Y

    @classmethod
    def Si_from_Sobol(
        cls, N: int, G: ComputationalGraph, B: dict, C: dict=None, V: dict=None,
        calc_2nd: bool=True, num_resamples=100, conf_level=0.95, seed=None,
        save_time: bool=False
    ) -> dict:
        """Generates Sensitivity indices using the Sobol method
        Args:
            N: The number of samples to analyze when generating Si
            G: The ComputationalGraph to analyze
            B: A dictionary of bound information for the inputs of G
            C: A dictionary of cover values for use when G is a FIB
            V: A dictionary of GrFN input variable types
            calc_2nd: A boolean that determines whether to include S2 indices
            save_time: Whether to return timing information
        Returns:
            A SensitivityIndices object containing all data from SALib analysis
        """
        prob_def = cls.setup_problem_def(G.inputs, B)

        (samples, sample_time) = cls.__run_sampling(
            SAL.sample.saltelli.sample, prob_def, N,
            calc_second_order=calc_2nd, seed=seed
        )

        (Y, exec_time) = cls.__execute_CG(G, samples, prob_def, C, V)

        (S, analyze_time) = cls.__run_analysis(
            SAL.analyze.sobol.analyze, prob_def, Y,
            calc_second_order=True, num_resamples=100,
            conf_level=0.95, seed=None
        )

        Si = SensitivityIndices(S)
        return Si if not save_time \
            else (Si, (sample_time, exec_time, analyze_time))

    @classmethod
    def Si_from_FAST(
        cls, N: int, G: ComputationalGraph, B: dict, C: dict=None, V: dict=None,
        M: int=4,
        save_time: bool=False, verbose: bool=False
    ) -> dict:
        # TODO Khan: adapt this method to be the same style as Si_from_Sobol
        print("Sampling via FAST sampler...")
        samples = fast_sampler.sample(prob_def, num_samples, M=M)
        samples = np.split(samples, samples.shape[1], axis=1)
        samples = [s.squeeze() for s in samples]
        values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
        print("Running GrFN...")
        Y = network.run(values).numpy()
        print("Analyzing via FAST...")
        return fast.analyze(prob_def, Y, M=M)

    @classmethod
    def Si_from_RBD_FAST(
        cls, N: int, G: ComputationalGraph, B: dict, C: dict=None, V: dict=None,
        M: int=10,
        save_time: bool=False, verbose: bool=False
    ):
        # TODO Khan: adapt this method to be the same style as Si_from_Sobol
        print("Sampling via RBD-FAST...")
        samples = latin.sample(prob_def, num_samples)
        X = samples
        samples = np.split(samples, samples.shape[1], axis=1)
        samples = [s.squeeze() for s in samples]
        values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
        print("Running GrFN..")
        Y = network.run(values).numpy()
        print("Analyzing via RBD ...")
        return rbd_fast.analyze(prob_def, Y, X, M=M)


# class SobolIndex(object):
#     """ This class computes sobol indices from the SALib library
#
#     Attributes:
#         model (str): Name of Model (Upper Case)
#         file_bounds (str): Name of csv file containing Upper and Lower Bounds
#         of Variables
#         sample_size (list): List of sample sizes required for sobol index
#         computation
#     """
#
#     def __init__(self, model, file_bounds, sample_size):
#
#         self.model = model
#         self.file_bounds = file_bounds
#         self.sample_size = sample_size
#
#     def sobol_index_from_GrFN(self, GrFN):
#
#         """
#             Args:
#                 GrFN : GroundedFunctionNetwork
#
#             Returns:
#                     Dictionary with sample sizes, first and second order sobol
#                     indices for a particular model
#         """
#
#         sys.path.insert(0, "../../../tests/data/program_analysis")
#
#         if min(self.sample_size) < 1:
#             raise InputError("Sample Size too small --- Incorrect Sobol Indices Generated -- Min. sample size > 10E+4")
#
#         if self.model == 'PETPT':
#             fortran_code = 'PETPT.for'
#         elif self.model == 'PETASCE':
#             fortran_code = 'PETASCE_simple.for'
#         else:
#             raise InputError("Model Name Invalid!")
#
#         tG = GrFN.from_fortran_file("../../../tests/data/program_analysis/" + fortran_code)
#
#         args = tG.inputs
#         var_names = [args[i].split('::')[4] for i in range(len(args))]
#
#         var_df = pd.read_csv(self.file_bounds, sep=',', header=0)
#         var_dict_lb = pd.Series(var_df.Lower.values, index=var_df.Var).to_dict()
#         var_dict_ub = pd.Series(var_df.Upper.values, index=var_df.Var).to_dict()
#
#         var_bounds = dict()
#         type_dict  = dict()
#         for var_name in var_names:
#
#             if self.model == 'PETPT':
#                 key = self.model + "::@global::" + self.model.lower() + "::0::" + var_name + "::-1"
#             else:
#                 key = self.model + "_simple::@global::" + self.model.lower() + "::0::" + var_name + "::-1"
#
#             val = [var_dict_lb[var_name], var_dict_ub[var_name]]
#
#             var_bounds.update({key:val})
#
#             if var_name != 'meevp' and var_name != 'doy':
#                 type_val = (float, [0.0])
#             elif var_name == 'meevp':
#                 type_val = (str, ["A", "G"])
#             elif  var_name == 'doy':
#                 type_val = (int, list(range(1, 366)))
#
#             type_dict.update({key:type_val})
#
#
#
#         problem = {
#                 'num_vars': len(args),
#                 'names': args,
#                 'bounds': [var_bounds[arg] for arg in args]
#                 }
#
#         Ns = self.sample_size
#
#         sobol_indices_lst = []
#
#         for i in range(len(Ns)):
#
#             Si, sample_time, analysis_time = tG.sobol_analysis(Ns[i], problem, var_types=type_dict)
#             S1_dict = dict(zip(var_names,Si["S1"].tolist()))
#
#             for k in range(Si["S2"].shape[0]):
#                 for l in range(k,Si["S2"].shape[1]):
#                     if k != l:
#                         Si["S2"][l][k] = Si["S2"][k][l]
#
#             Si["S2"] = np.nan_to_num(Si["S2"]).tolist()
#
#             S2_dataframe = pd.DataFrame(data = Si["S2"], columns = var_names).to_json()
#
#             sobol_dict = {"sample size": np.log10(Ns[i]), "First Order": S1_dict, "Second Order (DataFrame)": S2_dataframe, "Sample Time": sample_time, "Analysis Time": analysis_time}
#
#             sobol_indices_lst.append(sobol_dict)
#
#         return sobol_indices_lst
#
#     def generate_json(self):
#
#         """
#             Returns:
#                     JSON file with sample sizes and sobol indices
#         """
#
#         with open("sobol_indices_" + self.model + ".json", "w") as fout:
#             json.dump(self.sobol_index_from_GrFN(), fout)
