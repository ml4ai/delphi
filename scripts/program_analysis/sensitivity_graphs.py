import numpy as np
import inspect
import pickle
import sys
from tqdm import tqdm

import delphi.program_analysis.data.Plant_pgm as plant

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def PLANT_wrapper(DOY, TMAX, TMIN, SWFAC1, PT, di, N, dN):
    SWFAC2 = [1.000]
    PD = [5.000]
    EMP1 = [0.104]
    EMP2 = [0.640]
    sla = [0.028]
    nb = [5.300]
    p1 = [0.030]

    return plant.MAIN(DOY, TMAX, TMIN, SWFAC1, SWFAC2, PD, EMP1, EMP2, PT, sla,
                      di, N, nb, dN, p1)


def evaluate_PLANT(X, Y, x_idx, y_idx, presets):
    sig = inspect.signature(PLANT_wrapper)
    args = list(sig.parameters)

    Z = np.zeros((len(X), len(X)))
    for r in tqdm(range(len(X)), desc="Eval PLANT"):
        for c in range(len(X[0])):
            eval_args = list()
            for idx, arg in enumerate(args):
                if idx == x_idx:
                    eval_args.append([X[r][c]])
                elif idx == y_idx:
                    eval_args.append([Y[r][c]])
                else:
                    eval_args.append(presets[idx])

            Z[r][c] = PLANT_wrapper(*tuple(eval_args))
    return Z


sig = inspect.signature(PLANT_wrapper)
args = list(sig.parameters)

preset_vals = [[234], [33.9000015], [22.7999992], [1.0], [0.974399984], [0.0],
               [12.0639181], [0.0]]

bounds = [
    [0, 147],
    [0.000, 33.900],
    [-2.800, 21.100],
    [0.442, 1.000],
    [0.000, 1.000],
    [0.000, 15.000],
    [2.000, 12.039],
    [0.000, 0.100]
]

num_samples = 1000

data = pickle.load(open("PLANT-sensitivity.pkl", "rb"))
Z = data["outputs"]
Si = data["Si"]
S2 = Si["S2"]
S1 = Si["S1"]
# print(S1)
# sys.exit()

S2_ranks = [(val, r, c) for r, row in enumerate(S2)
            for c, val in enumerate(row) if c > r]

(s_val, r, c) = min(S2_ranks, key=lambda tup: abs(tup[0]))
print("Dim 1 name: {}".format(args[r]))
print("Dim 2 name: {}".format(args[c]))

(x_low, x_high) = bounds[r]
(y_low, y_high) = bounds[c]

X = np.arange(x_low, x_high, (x_high - x_low) / num_samples)
Y = np.arange(y_low, y_high, (y_high - y_low) / num_samples)
X, Y = np.meshgrid(X, Y)

Z = evaluate_PLANT(X, Y, r, c, preset_vals)
print("finished evaluating meshgrid")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("Low S2 index ({:.4f}) visualization".format(s_val))
ax.set_xlabel("{}".format(args[r]))
ax.set_ylabel("{}".format(args[c]))
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis_r,
                       linewidth=0, antialiased=False)
# ax.plot_wireframe(X, Y, Z)

plt.show()
