from SALib.sample import saltelli, fast_sampler, latin
from SALib.analyze import sobol, fast, rbd_fast
import numpy as np
import torch

from delphi.GrFN.utils import timeit


@timeit
def sobol_analysis(network, num_samples, prob_def, use_torch=False, var_types=None):
    def create_input_tensor(name, samples):
        type_info = var_types[name]
        if type_info[0] == str:
            (val1, val2) = type_info[1]
            return np.where(samples >= 0.5, val1, val2)
        else:
            return torch.tensor(samples)

    print("Sampling via Saltelli...")
    samples = saltelli.sample(prob_def, num_samples, calc_second_order=True)
    print("Running GrFN...")
    if use_torch:
        samples = np.split(samples, samples.shape[1], axis=1)
        if var_types is None:
            values = {n: torch.tensor(s)
                      for n, s in zip(prob_def["names"], samples)}
        else:
            values = {n: create_input_tensor(n, s)
                      for n, s in zip(prob_def["names"], samples)}
        print(samples[0])
        Y = network.run(values, torch_size=len(samples[0])).numpy()
    else:
        Y = np.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            values = {n: val for n, val in zip(prob_def["names"], sample)}
            Y[i] = network.run(values)

    print("Analyzing via Sobol...")
    return sobol.analyze(prob_def, Y, print_to_console=True)


@timeit
def FAST_analysis(network, num_samples, prob_def):
    print("Sampling via FAST sampler...")
    samples = fast_sampler.sample(prob_def, num_samples)
    samples = np.split(samples, samples.shape[1], axis=1)
    values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
    print("Running GrFN...")
    Y = network.run(values).numpy()
    print("Analyzing via FAST...")
    return fast.analyze(prob_def, Y, print_to_console=True)


@timeit
def RBD_FAST_analysis(network, num_samples, prob_def):
    print("Sampling via RBD-FAST...")
    samples = latin.sample(prob_def, num_samples)
    X = samples
    samples = np.split(samples, samples.shape[1], axis=1)
    values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
    print("Running GrFN..")
    Y = network.run(values).numpy()
    print("Analyzing via RBD ...")
    return rbd_fast.analyze(prob_def, Y, X, print_to_console=True)
