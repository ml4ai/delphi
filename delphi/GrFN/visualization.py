import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

class SensitivityVisualizer(object):

    """
    This class is responsible for generating plots of the first and second
    order sensitivity indices as well as the runtime of each computation as a
    function of the logarithm of the sample sizes.

    Attributes:
        S (list) : List of python dictionaries with the following keys - 'N',
        'S1', 'S2', 'sampling time', 'execution time', 'analysis time'.
        N -- sample size (log 10 scale)
        S1 -- numpy array of First Order Sensitivity Indices
        S2 -- pandas dataframe of Second Order Sensitivity Indices
        sampling time -- time taken to complete the sampling process after
        invoking a method from the SALib library
        execution time -- time taken for executing the GrFN
        analysis time -- time taken for the computation of Sensitivity Indices
    """

    def __init__(self, S: list):

        self.N = []
        self.S1_indices = []
        self.S2_dataframe = []
        self.sample_time = []
        self.execution_time = []
        self.analysis_time = []

        for item in S:
            self.N.append(float(np.log10(item["sample size"])))
            self.S1_indices.append(item["S1"])
            self.S2_dataframe.append(item["S2"])
            self.sample_time.append(float(item["sampling time"]))
            self.execution_time.append(float(item["execution time"]))
            self.analysis_time.append(float(item["analysis time"]))

    def create_S1_plot(self, filename = "s1_plot.pdf"):
        """ Creates a plot of S1 versus log (base 10) of sample sizes.
        Args:
            filename: Filename to save plot to.
        """

        S1_dataframe = pd.DataFrame(self.S1_indices)
        S1_dataframe.index = self.N

        cols = list(S1_dataframe.columns)

        fig, ax = plt.subplots(figsize=(12, 8))

        for col in cols:
            S1_dataframe.reset_index().plot(
                kind="scatter", x="index", y=col, ax=ax, c="r", s=50
            )
            S1_dataframe.reset_index().plot(
                kind="line", x="index", y=col, ax=ax
            )
            plt.legend(loc="right", fontsize=20)
            plt.xlim(min(self.N) - 1, max(self.N) + 1)
            plt.xlabel("$log_{10}N$", fontsize=30)
            plt.ylabel("$S_i$", fontsize=30)
            plt.title(r"$S_i$ vs $log_{10}N$", fontsize=30)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim(-0.2, 1.0)

        plt.savefig(filename)
        return plt

    def create_S2_plot(self, filename = "s2_plot.pdf"):
        """ Creates gridplot with second order Sobol index matrices for largest sample
        size on log (base 10) scale.

        Args:
            filename: Filename to save plot to.
        """

        elem = len(self.N) - 1

        S2_mat = self.S2_dataframe[elem].to_dict()
        df = pd.DataFrame(S2_mat)
        max_val = max(df.max(axis=0).values)
        var_names = df.columns

        if len(df.columns) < 10:
            plt.figure(figsize=(12, 12))
        else:
            plt.figure(figsize=(15, 15))
        cmap = sns.diverging_palette(240, 10, n=9)
        g = sns.heatmap(
            df,
            cmap=cmap,
            annot=True,
            xticklabels=var_names,
            yticklabels=var_names,
            annot_kws={"fontsize": 10},
            vmax=max_val,
            vmin=-max_val,
        )
        plt.title(
            "$S_{ij}$ ( $log_{10}N$ = " + str(self.N[elem]) + " )", fontsize=30
        )
        plt.xticks(fontsize=15, rotation=30)
        plt.yticks(fontsize=15, rotation=0)

        plt.savefig(filename)
        return plt

    def create_clocktime_plot(self, filename = "clocktime_plot.pdf"):
        """ Creates plot of runtime (Sample Time, Execution Time, and Analysis Time)
        versus log (base 10) of sample sizes """

        fig, ax = plt.subplots(figsize=(15, 8))

        ax.scatter(self.N, self.sample_time, color="r", s=50)
        ax.plot(self.N, self.sample_time, color="black", label="Sample Time")
        ax.scatter(self.N, self.execution_time, color="r", s=50)
        ax.plot(self.N, self.execution_time, color="b", label="Execution Time")
        ax.scatter(self.N, self.analysis_time, color="r", s=50)
        ax.plot(self.N, self.analysis_time, color="g", label="Analysis Time")
        plt.legend()
        plt.xlabel("$log_{10}N$", fontsize=30)
        plt.ylabel("Runtime (in seconds)", fontsize=30)
        plt.title(
            "Sampling, Execution, and Analysis Times vs $log_{10}N$",
            fontsize=30,
        )
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(filename)

        return plt
