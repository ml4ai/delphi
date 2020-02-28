import inspect
import importlib
import json
import sys
from delphi.GrFN.networks import GroundedFunctionNetwork
import numpy as np
import pandas as pd
import time
import json

class InputError(Exception):
    pass

def sobol_index_from_GrFN(model, file_bounds, sample_size):
    sys.path.insert(0, "../tests/data/program_analysis")

    if min(sample_size) < 1:
        raise InputError("Sample Size too small --- Incorrect Sobol Indices Generated -- Min. sample size > 10E+4")

    if model == 'PETPT':
        fortran_code = 'PETPT.for'
    elif model == 'PETASCE':
        fortran_code = 'PETASCE_simple.for'
    else:
        raise InputError("Model Name Invalid!")

    tG = GroundedFunctionNetwork.from_fortran_file("../tests/data/program_analysis/" + fortran_code)

    args = tG.inputs
    var_names = [args[i].split('::')[4] for i in range(len(args))]
    
    var_df = pd.read_csv(file_bounds, sep=',', header=0)
    var_dict_lb = pd.Series(var_df.Lower.values, index=var_df.Var).to_dict()
    var_dict_ub = pd.Series(var_df.Upper.values, index=var_df.Var).to_dict()
    
    var_bounds = dict()
    type_dict  = dict()
    for var_name in var_names:
        
        if model == 'PETPT':
            key = model + "::@global::" + model.lower() + "::0::" + var_name + "::-1"
        else:
            key = model + "_simple::@global::" + model.lower() + "::0::" + var_name + "::-1"
        
        val = [var_dict_lb[var_name], var_dict_ub[var_name]]
        
        var_bounds.update({key:val})
        
        if var_name != 'meevp' and var_name != 'doy':
            type_val = (float, [0.0])
        elif var_name == 'meevp':
            type_val = (str, ["A", "G"])
        elif  var_name == 'doy':
            type_val = (int, list(range(1, 366)))

        type_dict.update({key:type_val})


    # print(var_bounds)
    # print(type_dict)
    # print(args)

    problem = {
            'num_vars': len(args),
            'names': args,
            'bounds': [var_bounds[arg] for arg in args]
            }

    Ns = sample_size

    sobol_indices_lst = []
    clocktime_Sobol = []

    for i in range(len(Ns)):

        start = time.clock()
        Si = tG.sobol_analysis(Ns[i], problem, var_types=type_dict)        
        end = time.clock()
        S1_dict = dict(zip(var_names,Si["S1"].tolist()))
    
        for k in range(Si["S2"].shape[0]):
            for l in range(k,Si["S2"].shape[1]):
                if k != l:
                    Si["S2"][l][k] = Si["S2"][k][l]
        
        Si["S2"] = np.nan_to_num(Si["S2"]).tolist()

        S2_dataframe = pd.DataFrame(data = Si["S2"], columns = var_names).to_json()
        
        sobol_dict = {"sample size": np.log10(Ns[i]), "First Order": S1_dict, "Second Order (DataFrame)": S2_dataframe, "Clocktime": end-start}
        
        sobol_indices_lst.append(sobol_dict)
    
    with open("sobol_indices_" + model + ".json", "w") as fout:
        json.dump(sobol_indices_lst, fout)


# sobol_index_from_GrFN('PETPT', 'petpt_var_bounds.csv', [10, 100, 1000, 10000])
sobol_index_from_GrFN('PETASCE', 'petasce_var_bounds.csv', [10, 100, 1000, 10000])
# sobol_index_from_GrFN('PETASCE', 'petasce_var_bounds.csv', 10**2)

