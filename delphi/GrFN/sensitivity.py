from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast
import numpy as np
import torch


def sobol_analysis(network, num_samples, prob_def):
    print("Sampling via Saltelli...")
    samples = saltelli.sample(prob_def, num_samples, calc_second_order=True)
    samples = np.split(samples, samples.shape[1], axis=1)
    values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
    print("Running GrFN...")
    Y = network.run(values).numpy()
    print("Analyzing via Sobol...")
    return sobol.analyze(prob_def, Y, print_to_console=True)


def FAST_analysis(network, num_samples, prob_def):
    print("Sampling...")
    samples = fast_sampler.sample(prob_def, num_samples)
    samples = np.split(samples, samples.shape[1], axis=1)
    values = {n: torch.tensor(s) for n, s in zip(prob_def["names"], samples)}
    print("Running GrFN...")
    Y = network.run(values).numpy()
    print("Getting sensitivity indicies...")
    return fast.analyze(prob_def, Y, print_to_console=True)
