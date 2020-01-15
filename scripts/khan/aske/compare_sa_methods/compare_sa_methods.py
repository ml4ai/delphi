import inspect
import importlib
import json
import sys
from delphi.GrFN.networks import GroundedFunctionNetwork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import seaborn as sns
sns.set_style('whitegrid')
import time




def sobol_index_from_GrFN(model, file_bounds):
    sys.path.insert(0, "../tests/data/program_analysis")
    if model == 'PETPT':
        fortran_code = 'PETPT.for'
    elif model == 'PETASCE':
        fortran_code = 'PETASCE_simple.for'

    tG = GroundedFunctionNetwork.from_fortran_file("../tests/data/program_analysis/" + fortran_code)

    args = tG.inputs
    var_names = [args[i].split('::')[4] for i in range(len(args))]
    
    var_df = pd.read_csv(file_bounds, sep='\s+', header=0)
    var_dict = pd.Series(var_df.Vals.values, index=var_df.Var).to_dict()
    
    var_bounds = dict()
    type_dict  = dict()
    for var_name in var_names:
        
        if model == 'PETPT':
            key = model + "::@global::" + model.lower() + "::0::" + var_name + "::-1"
        else:
            key = model + "_simple::@global::" + model.lower() + "::0::" + var_name + "::-1"
        
        val = [var_dict[var_name+ '_lb'], var_dict[var_name + '_ub']]
        
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

    Ns = [10**x for x in range(1,5)]

    S1_Sobol, S2_Sobol = [], []
    clocktime_Sobol = []

    for i in range(len(Ns)):
        start = time.clock()
        Si = tG.sobol_analysis(Ns[i], problem, var_types=type_dict)        
        end = time.clock()
        clocktime_Sobol.append(end - start)
        S1_Sobol.append(Si["S1"]) 
        S2_Sobol.append(Si["S2"])


    var_names = [args[i].split('::')[4] for i in range(len(args))]

    plt.figure(figsize=(12,8))
    for i in trange(len(S1_Sobol[0]), desc="Plotting for different sample sizes"):
        val = [pt[i] for pt in S1_Sobol]
        plt.scatter(np.log10(Ns), val, color = 'r', s=50)
        if i < 5:
            plt.plot(np.log10(Ns), val, label=var_names[i])
        else:
            plt.plot(np.log10(Ns), val, label=var_names[i], linestyle='--')
        plt.legend(loc='best', fontsize=20)
        plt.xlabel("Number of samples (log scale)", fontsize=30)
        plt.ylabel("Indices in Sobol method", fontsize=30)
        plt.title(r"First Order Sobol Index S$_i$ in PETASCE Model for different sample size (log10 scale)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.show()



    S2_dataframe = pd.DataFrame(np.concatenate(S2_Sobol), columns = args).fillna(0)

    for i in trange(len(Ns)):
        corr = S2_dataframe[i*len(args):(i+1)*len(args)]
        np.fill_diagonal(corr.values,0)
        arr = np.triu(corr) + np.triu(corr,1).T
        if len(var_names) < 10:
            plt.figure(figsize=(12, 12))
        else:
            plt.figure(figsize=(15,15))
        g = sns.heatmap(arr, annot=True, cmap ='Blues', xticklabels = var_names, yticklabels = var_names, annot_kws={"fontsize":10})
        plt.title("Second Order index for sample size {0}".format(Ns[i]), fontsize=30)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    
    plt.show()

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)

    ax.scatter(Ns, clocktime_Sobol, label = 'Sobol', color ='r', s = 50)
    ax.plot(Ns, clocktime_Sobol, color ='black')
    plt.legend()
    plt.xlabel('Number of Samples', fontsize=30)
    plt.ylabel('Clocktime (in seconds)', fontsize=30)
    plt.title('Time taken for computation of Sobol Indices ('  + model + ') as a function of sample size', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# sobol_index_from_GrFN('PETPT', 'petpt_var_bounds.txt')
# sobol_index_from_GrFN('PETASCE', 'petasce_var_bounds.txt')

