import numpy as np
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Reporter:
    def __init__(self, analyzer, num_samples=1000):
        self.analyzer = analyzer
        self.num_samples = num_samples
        self.get_sensitivity_indices(self.num_samples)

    def get_sensitivity_indices(self, num_samples):
        self.analyzer.sample(num_samples=num_samples)
        self.analyzer.evaluate()
        self.Si = self.analyzer.analyze(parallel=False,
                                        print_to_console=False,
                                        n_processors=None)

    def __get_s2_ranks__(self):
        return [(val, r, c) for r, row in enumerate(self.Si["S2"])
                for c, val in enumerate(row) if c > r]

    def get_min_s2_sensitivity(self):
        S2_ranks = self.__get_s2_ranks__()
        return min(S2_ranks, key=lambda tup: abs(tup[0]))

    def get_max_s2_sensitivity(self):
        S2_ranks = self.__get_s2_ranks__()
        return max(S2_ranks, key=lambda tup: abs(tup[0]))

    def get_sensitivity_surface(self, var1_idx, var2_idx, bounds, presets):
        ((x_low, x_high), (y_low, y_high)) = bounds

        X = np.arange(x_low, x_high, (x_high - x_low) / self.num_samples)
        Y = np.arange(y_low, y_high, (y_high - y_low) / self.num_samples)
        X, Y = np.meshgrid(X, Y)

        args = self.analyzer.get_model_args()

        surface = np.zeros((len(X), len(X)))
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

                surface[r][c] = self.analyzer.evaluate(*tuple(eval_args))
        print("finished evaluating meshgrid")
        return surface

    def visualize_surface(self, var1_idx, var2_idx, X, Y, surface):
        args = self.analyzer.get_model_args()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("S2 surface visualization")
        ax.set_xlabel("{}".format(args[var1_idx]))
        ax.set_ylabel("{}".format(args[var2_idx]))
        ax.plot_surface(X, Y, surface, cmap=cm.viridis_r,
                        linewidth=0, antialiased=False)
        # ax.plot_wireframe(X, Y, Z)
        plt.show()
