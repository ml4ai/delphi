import pickle

import numpy as np
import matplotlib.pyplot as plt


def main():
    Si_data = pickle.load(open("penman_data.pkl", "rb"))

    N_amounts = [int(n) for n in np.linspace(1000, 10000, num=10)]

    plt.subplots(nrows=4, ncols=1, sharex=True)
    plt.xticks(N_amounts)

    for model_name, results in Si_data.items():
        N_sizes = sorted(list(results.keys()))
        S1_results = {k: list() for k in results[N_size].O1_indices.keys()}
        for N_size in N_sizes:
            S1_data = results[N_size].O1_indices
            for var_name, value in S1_data.items():
                S1_results[var_name].append()



    plt.show()


if __name__ == '__main__':
    main()
