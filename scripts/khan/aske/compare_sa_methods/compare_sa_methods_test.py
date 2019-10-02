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




def test_PETPT_GrFN():
    #   sys.path.insert(0, "tests/data/GrFN/")
    sys.path.insert(0, "../tests/data/GrFN")
    lambdas = importlib.__import__("PETPT_torch_lambdas")
#   pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
    pgm = json.load(open("../tests/data/GrFN/PETPT_numpy.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    args = G.inputs
    bounds = {
            "petpt::msalb_0": [0, 1],      # TODO: Khan set proper values for x1, x2
            "petpt::srad_0": [1, 30],       # TODO: Khan set proper values for x1, x2
            "petpt::tmax_0": [-30, 60],       # TODO: Khan set proper values for x1, x2
            "petpt::tmin_0": [-30, 60],       # TODO: Khan set proper values for x1, x2
            "petpt::xhlai_0": [0, 20],      # TODO: Khan set proper values for x1, x2
            }

    problem = {
            'num_vars': len(args),
            'names': args,
            'bounds': [bounds[arg] for arg in args]
            }

    Ns = [1000]
    for i in range(1):
        Ns.append(Ns[i]*10)
        #print("Time values :", timeit.Timer('sobol_analysis(G, Ns[i],problem)').timeit())
        #plotTC(sobol_analysis(G, Ns[i],problem), 10, 1000, 10, 10)

    print(" Number of samples:", Ns)
    S1_Sobol, S2_Sobol = [], []
    S1_FAST = []
    S1_RBD_FAST = []
    clocktime_Sobol, clocktime_FAST, clocktime_RBD_FAST = [], [], []

    for i in range(len(Ns)):
        start = time.clock()
        Si = sobol_analysis(G, Ns[i], problem)
        end = time.clock()
        #print(" Time elapsed : ", end - start)
        clocktime_Sobol.append(end - start)
        S1_Sobol.append(Si["S1"]) 
        S2_Sobol.append(Si["S2"])



    #print("S1 indices are :", S1)
    #print("Total number of elements in S1:", len(S1[0]))

    #print("S2 indices are :", S2)

    S2_dataframe = pd.DataFrame(np.concatenate(S2_Sobol), columns = args).fillna(0)
    #print("S2 dataframe :", S2_dataframe)

    for i in range(len(S1_Sobol[0])):
        val = [pt[i] for pt in S1_Sobol]
        plt.scatter(Ns, val, label = args[i], color = 'r', s = 50)
        plt.plot(Ns, val, color = 'b')
        plt.legend()
        plt.xlabel("Number of samples")
        plt.ylabel("Indices for {0} in Sobol method".format(args[i]))
        plt.show()

        #if i < len(S1_Sobol[0])-1:
    for i in range(len(Ns)):
        corr = S2_dataframe[i*len(args):(i+1)*len(args)]
        np.fill_diagonal(corr.values,1)
        print(corr)
        arr = np.triu(corr) + np.triu(corr,1).T
        print(arr)
        sns.heatmap(arr, annot=True, cmap ='Blues')
        plt.show()




test_PETPT_GrFN()

