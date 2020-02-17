import inspect
import importlib
import json
import sys
import numpy as np
import pandas as pd
import time

class InputError(Exception):
    pass

class SobolIndex(object):
    """ This class computes sobol indices from the SALib library 
    
    Attributes:
        model (str): Name of Model (Upper Case)
        file_bounds (str): Name of csv file containing Upper and Lower Bounds
        of Variables
        sample_size (list): List of sample sizes required for sobol index
        computation
    """

    def __init__(self, model, file_bounds, sample_size):

        self.model = model
        self.file_bounds = file_bounds
        self.sample_size = sample_size

    def sobol_index_from_GrFN(self, GrFN):

        """ 
            Args:
                GrFN : GroundedFunctionNetwork

            Returns: 
                    Dictionary with sample sizes, first and second order sobol
                    indices for a particular model
        """

        sys.path.insert(0, "../../../tests/data/program_analysis")

        if min(self.sample_size) < 1:
            raise InputError("Sample Size too small --- Incorrect Sobol Indices Generated -- Min. sample size > 10E+4")

        if self.model == 'PETPT':
            fortran_code = 'PETPT.for'
        elif self.model == 'PETASCE':
            fortran_code = 'PETASCE_simple.for'
        else:
            raise InputError("Model Name Invalid!")

        tG = GrFN.from_fortran_file("../../../tests/data/program_analysis/" + fortran_code)

        args = tG.inputs
        var_names = [args[i].split('::')[4] for i in range(len(args))]

        var_df = pd.read_csv(self.file_bounds, sep=',', header=0)
        var_dict_lb = pd.Series(var_df.Lower.values, index=var_df.Var).to_dict()
        var_dict_ub = pd.Series(var_df.Upper.values, index=var_df.Var).to_dict()

        var_bounds = dict()
        type_dict  = dict()
        for var_name in var_names:

            if self.model == 'PETPT':
                key = self.model + "::@global::" + self.model.lower() + "::0::" + var_name + "::-1"
            else:
                key = self.model + "_simple::@global::" + self.model.lower() + "::0::" + var_name + "::-1"

            val = [var_dict_lb[var_name], var_dict_ub[var_name]]

            var_bounds.update({key:val})

            if var_name != 'meevp' and var_name != 'doy':
                type_val = (float, [0.0])
            elif var_name == 'meevp':
                type_val = (str, ["A", "G"])
            elif  var_name == 'doy':
                type_val = (int, list(range(1, 366)))

            type_dict.update({key:type_val})



        problem = {
                'num_vars': len(args),
                'names': args,
                'bounds': [var_bounds[arg] for arg in args]
                }

        Ns = self.sample_size

        sobol_indices_lst = []

        for i in range(len(Ns)):

            Si, sample_time, analysis_time = tG.sobol_analysis(Ns[i], problem, var_types=type_dict)        
            S1_dict = dict(zip(var_names,Si["S1"].tolist()))

            for k in range(Si["S2"].shape[0]):
                for l in range(k,Si["S2"].shape[1]):
                    if k != l:
                        Si["S2"][l][k] = Si["S2"][k][l]

            Si["S2"] = np.nan_to_num(Si["S2"]).tolist()

            S2_dataframe = pd.DataFrame(data = Si["S2"], columns = var_names).to_json()

            sobol_dict = {"sample size": np.log10(Ns[i]), "First Order": S1_dict, "Second Order (DataFrame)": S2_dataframe, "Sample Time": sample_time, "Analysis Time": analysis_time}

            sobol_indices_lst.append(sobol_dict)

        return sobol_indices_lst



    def generate_json(self):

        """ 
            Returns:
                    JSON file with sample sizes and sobol indices 
        """

        with open("sobol_indices_" + self.model + ".json", "w") as fout:
            json.dump(self.sobol_index_from_GrFN(), fout)
