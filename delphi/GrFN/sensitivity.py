from numbers import Number
import SALib as SAL
import numpy as np
import json
import csv
import pickle
from delphi.GrFN.networks import ComputationalGraph
from delphi.GrFN.utils import timeit


class InputError(Exception):
    pass


class SensitivityIndices(object):
    """ This class creates an object with first and second order sensitivity
    indices as well as the total sensitivty index for a given sample size. It
    also contains the confidence interval associated with the computation of
    each index. The indices are in the form of a dictionary and they can be saved
    to or read from JSON, pickle, and csv files. In addition, the maximum and
    minimum second order of the indices between any two input variables can be
    determined using the max (min) and argmax (argmin) methods.
    """
    def __init__(self, S: dict, problem: dict):
        """
        Args:
            S: A SALib dictionary from analysis
        """
        self.parameter_list = problem["names"]
        self.O1_indices = np.array(S["S1"]) if "S1" in S else None
        self.O2_indices = np.array(S["S2"]) if "S2" in S else None
        self.OT_indices = np.array(S["ST"]) if "ST" in S else None
        self.O1_confidence = np.array(S["S1_conf"]) if "S1_conf" in S else None
        self.O2_confidence = np.array(S["S2_conf"]) if "S2_conf" in S else None
        self.OT_confidence = np.array(S["ST_conf"]) if "ST_conf" in S else None

    def check_first_order(self):
        if self.O1_indices is None:
            raise ValueError("No first order indices present")
        else:
            return True

    def check_second_order(self):
        if self.O2_indices is None:
            raise ValueError("No second order indices present")
        else:
            return True

    def check_total_order(self):
        if self.OT_indices is None:
            raise ValueError("No total order indices present")
        else:
            return True

    @classmethod
    def from_csv(cls, filepath: str):

        with open(filepath) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                S1 = float(row['S1'])
                S2 = np.array(row['S2'])
                ST = float(row['ST'])
                S1_conf = float(row['S1_conf'])
                S2_conf = float(row['S2_conf'])
                ST_conf = float(row['ST_conf'])

        Si_dict = {'S1': np.array(S1), 'S2': np.array(S2), 'ST': np.array(ST), 'S1_conf': S1_conf, 'S2_conf': S2_conf, 'ST_conf': ST_conf}

        return cls(Si_dict)

    @classmethod
    def from_dicts(cls, Si: dict, P: dict):
        """Creates a SensitivityIndices object from the provided dictionary."""
        return cls(Si, P)

    @classmethod
    def from_json(cls, filepath: str):

        data = open(filepath, encoding='utf-8').read()
        js = json.loads(data)

        Si_dict = {
            'S1': np.array(js['S1']),
            'S2': np.array(js['S2']),
            'ST': np.array(js['ST']),
            'S1_conf': float(js['S1_conf']),
            'S2_conf': float(js['S2_conf']),
            'ST_conf': float(js['ST_conf'])
        }
        problem = js["names"]

        return cls(Si_dict, problem)

    @classmethod
    def from_pickle(cls, filepath: str):

        with open(filepath, 'rb') as fin:
            Si_dict = pickle.load(fin)

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

    def to_csv(self, filepath: str, S: dict):
        try:
            with open(filepath, 'w') as fout:
                writer = csv.DictWriter(fout, fieldnames=S.keys())
                writer.writeheader()
                writer.writerow(S)
        except IOError:
            print("Input File Missing!")

    def to_json(self, filepath: str, S: dict):
        S['S2'] = S['S2'].tolist()
        try:
            with open(filepath, 'w') as fout:
                json.dump(S, fout)
        except IOError:
            print("Input File Missing!")

    def to_pickle(self, filepath: str, S: dict):
        try:
            fout = open(filepath, 'wb')
            pickle.dump(S, fout)
            fout.close()
        except IOError:
            print("Input File Missing!")


class SensitivityAnalyzer(object):
    def __init__(self):
        pass

    @staticmethod
    def setup_problem_def(GrFN, B):
        """
        So not all bounds are created uniformly, we can have different types
        that are represented binarily or categorically and we can also have some
        conditions where a modeler may not want to include one of the inputs
        under sensitivity analysis
        """
        def convert_bounds(bound):
            num_bounds = len(bound)
            if num_bounds == 0:
                raise ValueError(f"Found input variable with 0 bounds")
            elif num_bounds == 1:
                # NOTE: still going to use a zero to 1 range for now
                return [0, 1]
            elif len(bound) == 2:
                if all([isinstance(b, Number) for b in bound]):
                    return bound
                else:
                    return [0, 1]
            else:
                return [0, 1]

        input_vars = list(GrFN.input_name_map.keys())
        return {
            'num_vars': len(input_vars),
            'names': input_vars,
            'bounds': [convert_bounds(B[var]) for var in input_vars]
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

        def reals_to_bools(samples):
            return np.where(samples >= 0.5, True, False)

        def reals_to_strs(samples, str_options):
            num_strs = len(str_options)
            return np.choose((samples*num_strs).astype(np.int64), num_strs)

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
        prob_def = cls.setup_problem_def(G, B)

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

        Si = SensitivityIndices(S, prob_def)
        return Si if not save_time \
            else (Si, (sample_time, exec_time, analyze_time))

    @classmethod
    def Si_from_FAST(
        cls, N: int, G: ComputationalGraph, B: dict, C: dict=None, V: dict=None,
        M: int=4,
        save_time: bool=False, verbose: bool=False, seed: int=None
    ) -> dict:

        prob_def = cls.setup_problem_def(G, B)

        (samples, sample_time) = cls.__run_sampling(
            SAL.sample.fast_sampler.sample, prob_def, N,
            M=M, seed=seed
        )

        (Y, exec_time) = cls.__execute_CG(G, samples, prob_def, C, V)

        (S, analyze_time) = cls.__run_analysis(
            SAL.analyze.fast.analyze, prob_def, Y,
            M=M, print_to_console=False, seed=seed
        )

        Si = SensitivityIndices(S, prob_def)
        return Si if not save_time \
            else (Si, (sample_time, exec_time, analyze_time))

    @classmethod
    def Si_from_RBD_FAST(
        cls, N: int, G: ComputationalGraph, B: dict, C: dict=None, V: dict=None,
        M: int=10,
        save_time: bool=False, verbose: bool=False, seed: int=None
    ):

        prob_def = cls.setup_problem_def(G, B)

        (samples, sample_time) = cls.__run_sampling(
            SAL.sample.latin.sample, prob_def, N, seed=seed
        )

        X = samples

        (Y, exec_time) = cls.__execute_CG(G, samples, prob_def, C, V)

        (S, analyze_time) = cls.__run_analysis(
            SAL.analyze.rbd_fast.analyze, X, Y,
            M=M, print_to_console=False, seed=seed
        )

        Si = SensitivityIndices(S, prob_def)
        return Si if not save_time \
            else (Si, (sample_time, exec_time, analyze_time))
