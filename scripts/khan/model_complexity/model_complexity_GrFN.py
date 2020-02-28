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
    G1 = GroundedFunctionNetwork.from_fortran_file("../../../tests/data/program_analysis/PETPT.for")
    petpt_args = G1.inputs
    
    petpt_var_names = [petpt_args[i].split('::')[4] for i in range(len(petpt_args))]

    G2 = GroundedFunctionNetwork.from_fortran_file("../../../tests/data/program_analysis/PETASCE_simple.for")
    petasce_args = G2.inputs
    
    petasce_var_names = [petasce_args[i].split('::')[4] for i in
            range(len(petasce_args))]

    petasce_non_shared_var_names = [i for i in petasce_var_names if i not in petpt_var_names]

    # print(petpt_var_names)
    # print(petasce_non_shared_var_names)
    
    # shared_df = pd.read_csv(shared_vars, sep='\s+', header=0)
    # shared_dict = pd.Series(shared_df.Vals.values, index=shared_df.Var).to_dict()
    shared_df = pd.read_csv(shared_var_bounds, sep=',')
    shared_dict_lb = pd.Series(shared_df.Lower.values, index=shared_df.Var).to_dict()
    shared_dict_ub = pd.Series(shared_df.Upper.values, index=shared_df.Var).to_dict()
    
    # print(shared_dict)

    # non_shared_df = pd.read_csv(non_shared_vars, sep='\s+', header=0)
    # non_shared_dict = pd.Series(non_shared_df.Vals.values, index=non_shared_df.Var).to_dict()
    non_shared_df = pd.read_csv(non_shared_var_bounds, sep=',')
    non_shared_dict_lb = pd.Series(non_shared_df.Lower.values, index=non_shared_df.Var).to_dict()
    non_shared_dict_ub = pd.Series(non_shared_df.Upper.values, index=non_shared_df.Var).to_dict()
    # print(non_shared_df)
    # print(non_shared_dict)

    petpt_bounds = dict()
    petasce_bounds = dict()
    for var_name in petpt_var_names:
        key1 = var_name 
        key2 = var_name 
        if shared_dict_lb[var_name] == shared_dict_ub[var_name]:
            val = np.full(size, float(shared_dict_lb[var_name]), dtype =float)
        else:
            val = np.linspace(float(shared_dict_lb[var_name]), float(shared_dict_ub[var_name]), size)
        petpt_bounds.update({key1:val})
        petasce_bounds.update({key2:val})
   
    print(petpt_bounds)

    for var_name in petasce_non_shared_var_names:
        key = var_name
        if var_name != 'meevp':
            if non_shared_dict_lb[var_name] == non_shared_dict_ub[var_name]:
                val = np.full(size, float(non_shared_dict_lb[var_name]), dtype=float)
            else:
                val = np.linspace(float(non_shared_dict_lb[var_name]), float(non_shared_dict_ub[var_name]), size)
        else:
            val = np.full(size, 'A', dtype=str)

        petasce_bounds.update({key:val})
    
    print(petasce_bounds)

    # bounds = {
    # "PETPT::@global::petpt::0::msalb::-1": 0.18,
    # "PETPT::@global::petpt::0::srad::-1": 10.0,
    # "PETPT::@global::petpt::0::tmax::-1": 40.0,
    # "PETPT::@global::petpt::0::tmin::-1": -20.0,
    # "PETPT::@global::petpt::0::xhlai::-1": 5,
    # } 

    # bounds = {
    # "PETPT::@global::petpt::0::msalb::-1": 0.18,
    # "PETPT::@global::petpt::0::srad::-1": 10.0,
    # "PETPT::@global::petpt::0::tmax::-1": 40.0,
    # "PETPT::@global::petpt::0::tmin::-1": -20.0,
    # "PETPT::@global::petpt::0::xhlai::-1": 5,
    # } 

    # petpt_vals = G1.run(bounds)
    petpt_vals = G1.run(petpt_bounds)[0]
    petasce_vals = G2.run(petasce_bounds)[0]
    x = list(range(1, len(petpt_vals)+1))

    print(petpt_vals)
    print(petasce_vals)
    print(x)
    names = petasce_var_names

    # print(names)
    # print(petasce_bounds[names[0]].round(2))
    # print(petasce_bounds[names[1]].round(2))
    # print(petasce_bounds[names[2]].round(2))
    # print(petasce_bounds[names[2]].round(2))
    # print(petasce_bounds[names[3]].round(2))
    # print(petasce_bounds[names[4]].round(2))
    # print(petasce_bounds[names[5]].round(2))
    # print(petasce_bounds[names[6]].round(2))
    # print(petasce_bounds[names[7]].round(2))
    # print(petasce_bounds[names[8]].round(2))
    # print(petasce_bounds[names[9]].round(2))
    # print(petasce_bounds[names[10]].round(2))
    # print(petasce_bounds[names[11]])
    # print(petasce_bounds[names[12]].round(2))
    stacked_arr = np.column_stack((petasce_bounds[names[0]].round(2),
        petasce_bounds[names[1]].round(2),
        petasce_bounds[names[2]].round(2), petasce_bounds[names[3]].round(2),
        petasce_bounds[names[4]].round(2), petasce_bounds[names[5]].round(2),
        petasce_bounds[names[6]].round(2), petasce_bounds[names[7]].round(2),
        petasce_bounds[names[8]].round(2), petasce_bounds[names[9]].round(2),
        petasce_bounds[names[10]].round(2), petasce_bounds[names[11]],
        petasce_bounds[names[12]].round(2)
        ))

    # stacked_arr = stacked_arr.round(2)
    # print(stacked_arr)
    
    names = np.array(petasce_var_names)
    # print(names)

    annot_dict = list(map(dict, np.dstack((np.repeat(names[None, :], size, axis=0), stacked_arr ))))
    # print(annot_dict)

    interact_plot = interactive.Interactive_Plot(x, petpt_vals, petasce_vals, model1, model2, annot_dict)
    interact_plot.fig_plot()

if __name__ == '__main__':

    file1 = 'shared_vars_bounds_petpt_petasce.csv'
    file2 = 'non_shared_vars_bounds_petpt_petasce.csv'
    size = 3
    model1 = 'PETPT'
    model2 = 'PETASCE'

    model_complexity(size, file1, file2, model1, model2)


