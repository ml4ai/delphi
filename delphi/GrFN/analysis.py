import numpy as np
from tqdm import tqdm
import torch


from delphi.GrFN.utils import timeit


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


@timeit
def S2_surface(G, sizes, bounds, presets, use_torch=False):
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
    X = np.linspace(*bounds[0][1], sizes[0])
    Y = np.linspace(*bounds[1][1], sizes[1])

    if use_torch:
        print("Using Torch for surface calculation")
        Xm, Ym = torch.meshgrid(torch.tensor(X), torch.tensor(Y))
        inputs = {n: torch.full_like(Xm, v) for n, v in presets.items()}
        print(inputs)
        inputs.update({
            bounds[0][0]: Xm,
            bounds[1][0]: Ym
        })
        Z = G.run(inputs).numpy()
    else:
        print("Performing surface calculation")
        Xm, Ym = np.meshgrid(X, Y)
        Z = np.zeros((len(X), len(Y)))
        for x in tqdm(range(len(X)), desc="Eval samples"):
            for y in range(len(Y)):
                inputs = {n: v for n, v in presets.items()}
                inputs.update({
                    bounds[0][0]: x,
                    bounds[1][0]: y
                })
                Z[x][y] = G.run(inputs)
    return X, Y, Z
