import numpy as np
from tqdm import tqdm
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from delphi.GrFN.sensitivity import sobol_analysis


def get_forward_influence_blankets(GrFN1, GrFN2):
    FIB1 = GrFN1.to_FIB(GrFN2)
    FIB2 = GrFN2.to_FIB(GrFN1)
    return FIB1, FIB2


def get_S2_ranks(S2_mat):
    return [(val, r, c) for r, row in enumerate(S2_mat)
            for c, val in enumerate(row) if c > r]


def get_min_s2_sensitivity(S2_mat):
    """
    Returns a tuple of the form:
        (S2-value, variable index 1, variable index 2)
    where S2-value is the minimum of the set of all S2 indices
    """
    return min(get_S2_ranks(S2_mat), key=lambda tup: abs(tup[0]))


def get_max_s2_sensitivity(S2_mat):
    """
    Returns a tuple of the form:
        (S2-value, variable index 1, variable index 2)
    where S2-value is the maximum of the set of all S2 indices
    """
    return max(get_S2_ranks(S2_mat), key=lambda tup: abs(tup[0]))


def max_S2_sensitivity_surface(G, num_samples, sizes, bounds, presets):
    """Calculates the sensitivity surface of a GrFN for the two variables with the highest S2 index.

    Args:
        G: A GrFN.
        num_samples: Number of samples for sensitivity analysis.
        sizes: Tuple of (number of x inputs, number of y inputs).
        bounds: Set of bounds for GrFN inputs.
        presets: Set of standard values for GrFN inputs.

    Returns:
        Tuple:
            Tuple: The names of the two variables that were selected
            Tuple: The X, Y vectors of eval values
            Z: The numpy matrix of output evaluations

    """
    args = G.inputs
    Si = sobol_analysis(G, num_samples, {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    })
    S2 = Si["S2"]
    (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

    x_var = args[v1]
    x_bounds = bounds[x_var]
    X = np.linspace(*x_bounds, sizes[0])

    y_var = args[v2]
    y_bounds = bounds[y_var]
    Y = np.linspace(*y_bounds, sizes[1])

    Xm, Ym = torch.meshgrid(torch.tensor(X), torch.tensor(Y))
    inputs = {
        x_var: Xm,
        y_var: Ym
    }

    for i, arg in enumerate(args):
        if i != v1 and i != v2:
            inputs[arg] = torch.full_like(Xm, presets[arg])

    Z = G.run(inputs)

    return (x_var, y_var), (X, Y), Z.numpy()
