import json
import sys
from delphi.GrFN.networks import GroundedFunctionNetwork
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm, trange
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

def model_complexity(num, shared_vars, non_shared_vars):
    sys.path.insert(0, "../tests/data/program_analysis")
    G1 = GroundedFunctionNetwork.from_fortran_file("../tests/data/program_analysis/PETPT.for")
    petpt_args = G1.inputs
    
    petpt_var_names = [petpt_args[i].split('::')[4] for i in range(len(petpt_args))]

    G2 = GroundedFunctionNetwork.from_fortran_file("../tests/data/program_analysis/PETASCE_simple.for")
    petasce_args = G2.inputs
    
    petasce_var_names = [petasce_args[i].split('::')[4] for i in
            range(len(petasce_args))]

    petasce_non_shared_var_names = [i for i in petasce_var_names if i not in petpt_var_names]

    # print(petpt_var_names)
    # print(petasce_non_shared_var_names)
    
    shared_df = pd.read_csv(shared_vars, sep='\s+', header=0)
    shared_dict = pd.Series(shared_df.Vals.values, index=shared_df.Var).to_dict()
    
    # print(shared_dict)

    non_shared_df = pd.read_csv(non_shared_vars, sep='\s+', header=0)
    non_shared_dict = pd.Series(non_shared_df.Vals.values, index=non_shared_df.Var).to_dict()
    
    # print(non_shared_df)
    # print(non_shared_dict)

    petpt_bounds = dict()
    petasce_bounds = dict()
    for var_name in petpt_var_names:
        key1 = "PETPT::@global::petpt::0::" + var_name + "::-1"
        key2 = "PETASCE::@global::petasce::0::" + var_name + "::-1"
        if shared_dict[var_name + '_lb'] == shared_dict[var_name + '_ub']:
            val =  float(shared_dict[var_name + '_lb'])
        else:
            val = np.linspace(float(shared_dict[var_name + '_lb']), float(shared_dict[var_name + '_ub']), num)
        petpt_bounds.update({key1:val})
        petasce_bounds.update({key2:val})
   
    print(petpt_bounds)

    for var_name in petasce_non_shared_var_names:
        key = "PETASCE::@global::petasce::0::" + var_name + "::-1"
        if var_name != 'meevp':
            if non_shared_dict[var_name + '_lb'] == non_shared_dict[var_name + '_ub']:
                val =  float(non_shared_dict[var_name + '_lb'])
            else:
                val = np.linspace(float(non_shared_dict[var_name + '_lb']), float(non_shared_dict[var_name + '_ub']), num)
        else:
            # print(non_shared_dict[var_name + '_lb'])
            val = 'A'
            # val = non_shared_dict[var_name + '_lb']
        petasce_bounds.update({key:val})
    
    print(petasce_bounds)

    # bounds = {
    # "PETPT::@global::petpt::0::msalb::-1": 0.18,
    # "PETPT::@global::petpt::0::srad::-1": 10.0,
    # "PETPT::@global::petpt::0::tmax::-1": 40.0,
    # "PETPT::@global::petpt::0::tmin::-1": -20.0,
    # "PETPT::@global::petpt::0::xhlai::-1": 5,
    # } 


    # petpt_vals = G1.run(bounds)

file1 = 'shared_vars_bounds.txt'
file2 = 'non_shared_vars_bounds.txt'
size = 2

model_complexity(size, file1, file2)


