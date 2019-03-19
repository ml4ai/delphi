import numpy as np
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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


def get_sensitivity_surface(var_indices, bounds, dims, presets, GrFN):
    (var1_idx, var2_idx) = var_indices
    ((x_low, x_high), (y_low, y_high)) = bounds
    (x_dim, y_dim) = dims

    X = np.arange(x_low, x_high, (x_high - x_low) / x_dim)
    Y = np.arange(y_low, y_high, (y_high - y_low) / y_dim)
    X, Y = np.meshgrid(X, Y)

    args = GrFN.get_model_args()

    surface = np.zeros((len(X), len(Y)))
    for r in tqdm(range(len(X)), desc="Eval PLANT"):
        for c in range(len(X[0])):
            eval_args = list()
            for idx, arg in enumerate(args):
                if idx == var1_idx:
                    eval_args.append([X[r][c]])
                elif idx == var2_idx:
                    eval_args.append([Y[r][c]])
                else:
                    eval_args.append(presets[idx])

            surface[r][c] = self.analyzer.eval_sample(tuple(eval_args))
    print("finished evaluating meshgrid")
    return (X, Y, surface)


def visualize_surface(var_indices, X, Y, surface):
    (var1_idx, var2_idx) = var_indices
    args = self.analyzer.get_model_args()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("S2 surface visualization")
    ax.set_xlabel("{}".format(args[var1_idx]))
    ax.set_ylabel("{}".format(args[var2_idx]))
    ax.plot_surface(X, Y, surface, cmap=cm.viridis_r,
                    linewidth=0, antialiased=False)
    # ax.plot_wireframe(X, Y, Z)
    # plt.show()
    return plt
