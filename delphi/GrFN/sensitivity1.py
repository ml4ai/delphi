from SALib.sample import saltelli, fast_sampler, latin
from SALib.analyze import sobol, fast, rbd_fast
import numpy as np
import torch

from delphi.GrFN.utils import timeit


@timeit
def FAST_analysis(network, num_samples, prob_def):
    print("Sampling via FAST sampler...")
    samples = fast_sampler.sample(prob_def, num_samples)
    samples = np.split(samples, samples.shape[1], axis=1)
    samples = [s.squeeze() for s in samples]
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
    samples = [s.squeeze() for s in samples]
    values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
    print("Running GrFN..")
    Y = network.run(values).numpy()
    print("Analyzing via RBD ...")
    return rbd_fast.analyze(prob_def, Y, X, print_to_console=True)
