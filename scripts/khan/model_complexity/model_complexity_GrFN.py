import json
import sys
import interactive
from delphi.GrFN.networks import GroundedFunctionNetwork
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm, trange
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

def model_complexity(num, shared_var_bounds, non_shared_var_bounds, model1, model2):
    sys.path.insert(0, "../../../tests/data/program_analysis")
    G1 = GroundedFunctionNetwork.from_fortran_file(f"../../../tests/data/program_analysis/{model1}.for")
    var1_args = G1.inputs
    
    var1_names = [var1_args[i].split('::')[4] for i in range(len(var1_args))]

    if model2 == 'PETASCE':
        model2 = model2 + '_simple'

    G2 = GroundedFunctionNetwork.from_fortran_file(f"../../../tests/data/program_analysis/{model2}.for")
    var2_args = G2.inputs
    
    var2_names = [var2_args[i].split('::')[4] for i in range(len(var2_args))]

    non_shared_var2_names = [i for i in var2_names if i not in var1_names]

    shared_df = pd.read_csv(shared_var_bounds, sep=',')
    shared_dict_lb = pd.Series(shared_df.Lower.values, index=shared_df.Var).to_dict()
    shared_dict_ub = pd.Series(shared_df.Upper.values, index=shared_df.Var).to_dict()
    
    non_shared_df = pd.read_csv(non_shared_var_bounds, sep=',')
    non_shared_dict_lb = pd.Series(non_shared_df.Lower.values, index=non_shared_df.Var).to_dict()
    non_shared_dict_ub = pd.Series(non_shared_df.Upper.values, index=non_shared_df.Var).to_dict()


    var1_bounds = dict()
    var2_bounds = dict()
    for var_name in var1_names:
        key1 = var_name 
        key2 = var_name 
        if shared_dict_lb[var_name] == shared_dict_ub[var_name]:
            val = np.full(size, float(shared_dict_lb[var_name]), dtype =float)
        else:
            val = np.linspace(float(shared_dict_lb[var_name]), float(shared_dict_ub[var_name]), size)
        var1_bounds.update({key1:val})
        var2_bounds.update({key2:val})
   
    

    for var_name in non_shared_var2_names:
        key = var_name
        if var_name != 'meevp':
            if non_shared_dict_lb[var_name] == non_shared_dict_ub[var_name]:
                val = np.full(size, float(non_shared_dict_lb[var_name]), dtype=float)
            else:
                val = np.linspace(float(non_shared_dict_lb[var_name]), float(non_shared_dict_ub[var_name]), size)
        else:
            val = np.full(size, non_shared_dict_lb[var_name], dtype=str)

        var2_bounds.update({key:val})

    var1_vals = G1.run(var1_bounds)[0]
    var2_vals = G2.run(var2_bounds)[0]
    x = list(range(1, len(var1_vals)+1))

    names = np.array(var2_names)
    
    stacked_arr = np.column_stack(tuple(var2_bounds[names[i]] for i in range(0,len(var2_names)) ))

    annot_dict = list(map(dict, np.dstack((np.repeat(names[None, :], size, axis=0), stacked_arr ))))

    interact_plot = interactive.Interactive_Plot(x, var1_vals, var2_vals, model1, model2, annot_dict)
    interact_plot.fig_plot()

if __name__ == '__main__':

    file1 = 'shared_vars_bounds_petpno_petpen.csv'
    file2 = 'non_shared_vars_bounds_petpno_petpen.csv'
    size = 25
    model1 = 'PETPNO'
    model2 = 'PETPEN'

    model_complexity(size, file1, file2, model1, model2)

    file1 = 'shared_vars_bounds_petpt_petpno.csv'
    file2 = 'non_shared_vars_bounds_petpt_petpno.csv'
    size = 25
    model1 = 'PETPT'
    model2 = 'PETPNO'

    model_complexity(size, file1, file2, model1, model2)

    file1 = 'shared_vars_bounds_petpt_petpen.csv'
    file2 = 'non_shared_vars_bounds_petpt_petpen.csv'
    size = 25
    model1 = 'PETPT'
    model2 = 'PETPEN'

    model_complexity(size, file1, file2, model1, model2)
