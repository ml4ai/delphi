import inspect
import importlib
import json
import sys

# import delphi.analysis.sensitivity.variance_methods as methods
# from delphi.analysis.sensitivity.reporter import Reporter
# from delphi.translators.for2py.data.PETASCE import PETASCE
# from delphi.translators.for2py.data.Plant_pgm import MAIN
from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.sensitivity import sobol_analysis, FAST_analysis, RBD_FAST_analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import time




def test_PETASCE_GrFN():
    #sys.path.insert(0, "tests/data/GrFN/")
    sys.path.insert(0, "../tests/data/GrFN")
    lambdas = importlib.__import__("PETASCE_simple_torch_lambdas")
    #pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
    pgm = json.load(open("../tests/data/GrFN/PETASCE_simple_torch.json", "r"))
    tG = GroundedFunctionNetwork.from_dict(pgm, lambdas)
    bounds = {
            "petasce::msalb_0": [0, 1],      # TODO: Khan set proper values for x1, x2
            "petasce::srad_0": [1, 30],       # TODO: Khan set proper values for x1, x2
            "petasce::tmax_0": [-30, 60],       # TODO: Khan set proper values for x1, x2
            "petasce::tmin_0": [-30, 60],       # TODO: Khan set proper values for x1, x2
            "petasce::xhlai_0": [0, 20],       # TODO: Khan set proper values for x1, x2
            "petasce::canht_0": [0, 3],      # TODO: Khan set proper values for x1, x2
            "petasce::doy_0": [1, 365],      # TODO: Khan set proper values for x1, x2
            "petasce::meevp_0": [0, 1],       # TODO: Khan set proper values for x1, x2
            "petasce::tdew_0": [-30, 60],       # TODO: Khan set proper values for x1, x2
            "petasce::windht_0": [0.1, 10],       # TODO: Khan set proper values for x1, x2
            "petasce::windrun_0": [0, 900],      # TODO: Khan set proper values for x1, x2
            "petasce::xlat_0": [-90, 90],       # TODO: Khan set proper values for x1, x2
            "petasce::xelev_0": [0, 6000],      # TODO: Khan set proper values for x1, x2
            }
    type_info = {
        "petasce::doy_0": (int, list(range(1, 366))),
        "petasce::meevp_0": (str, ["A", "G"]),
        "petasce::msalb_0": (float, [0.0]),
        "petasce::srad_0": (float, [0.0]),
        "petasce::tmax_0": (float, [0.0]),
        "petasce::tmin_0": (float, [0.0]),
        "petasce::xhlai_0": (float, [0.0]),
        "petasce::tdew_0": (float, [0.0]),
        "petasce::windht_0": (float, [0.0]),
        "petasce::windrun_0": (float, [0.0]),
        "petasce::xlat_0": (float, [0.0]),
        "petasce::xelev_0": (float, [0.0]),
        "petasce::canht_0": (float, [0.0]),
    }

    args = tG.inputs

    #print(args)
    problem = {
            'num_vars': len(args),
            'names': args,
            'bounds': [bounds[arg] for arg in args]
            }

    num = 3
    Ns = [1000]
    for i in range(num):
        Ns.append(Ns[i]*10)

    S1_Sobol, S2_Sobol = [], []
    S1_FAST = []
    S1_RBD_FAST = []
    clocktime_Sobol, clocktime_FAST, clocktime_RBD_FAST = [], [], []

    for i in range(len(Ns)):
#        start = time.clock()
        #Si = sobol_analysis(G, Ns[i], problem)
        Si = tG.sobol_analysis(Ns[i], problem, var_types=type_info, use_torch=True)        
#        end = time.clock()
        #print(" Time elapsed : ", end - start)
#        clocktime_Sobol.append(end - start)
        S1_Sobol.append(Si["S1"]) 
        S2_Sobol.append(Si["S2"])



    #print("S1 indices are :", S1_Sobol)
    #print("Total number of elements in S1:", len(S1[0]))

    #print("S2 indices are :", S2)

    S2_dataframe = pd.DataFrame(np.concatenate(S2_Sobol), columns = args).fillna(0)
    #print("S2 dataframe :", S2_dataframe)

    for i in range(len(S1_Sobol[0])):
        val = [pt[i] for pt in S1_Sobol]
        plt.scatter(Ns, val, color = 'r', s = 50)
        #plt.plot(Ns, val, color = 'b')
        if i < 5:
            plt.plot(Ns, val, label = args[i])
        else:
            plt.plot(Ns, val, label = args[i], linestyle = '--')
        plt.legend(loc='best')
        plt.xlabel("Number of samples")
        plt.ylabel("Indices in Sobol method")
        plt.ylim(-0.0005, 0.025)
        
    plt.show()

    for i in range(len(Ns)):
        corr = S2_dataframe[i*len(args):(i+1)*len(args)]
        np.fill_diagonal(corr.values,1)
        #print(corr)
        arr = np.triu(corr) + np.triu(corr,1).T
        #print(arr)
        plt.figure()
        sns.heatmap(arr, annot=True, cmap ='Blues', xticklabels = True,
                yticklabels = True)
        plt.title("Second Order index for sample size {0}".format(Ns[i]))
        

    plt.show()

        

test_PETASCE_GrFN()

