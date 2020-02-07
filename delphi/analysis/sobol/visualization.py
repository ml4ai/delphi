import json
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
sns.set_style('whitegrid')


class SobolVisualizer(object):

    """ This class is responsible for generating plots of the first and second order
    sobol indices as well as the runtime of each computation as a function of the log of sample sizes
    """

    def __init__(self, model):

        self.model = model


    def index_from_json(self, filename):
        
        """ input <- json file 
            output -> sample sizes, S1 & S2 indices, runtime"""
        
        data = open(filename, encoding='utf-8').read()
        js = json.loads(data)

        Ns = list()
        S1_Sobol = list()
        S2_data = list()
        sample_time_sobol = list()
        analysis_time_sobol = list()

        for item in js:
            Ns.append(float(item['sample size']))
            S1_Sobol.append(item['First Order'])
            S2_data.append(item['Second Order (DataFrame)'])
            sample_time_sobol.append(float(item['Sample Time']))
            analysis_time_sobol.append(float(item['Analysis Time']))

        return Ns, S1_Sobol, S2_data, sample_time_sobol, analysis_time_sobol


    def S1_Sobol_plot(self, sobol_dict):
        
        """ Function to plot S1 versus log N plots"""

        Ns = list()
        S1_Sobol = list()

        for item in sobol_dict:
            Ns.append(float(item['sample size']))
            S1_Sobol.append(item['First Order'])

        S1_dataframe = pd.DataFrame(S1_Sobol)
        S1_dataframe.index = Ns

        cols = list(S1_dataframe.columns)

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        for col in cols:
            S1_dataframe.reset_index().plot(kind='scatter', x="index", y=col, ax=ax, c ='r', s=50)
            S1_dataframe.reset_index().plot(kind='line', x="index", y=col, ax=ax)
            plt.legend(loc='right', fontsize=20)
            plt.xlim(min(Ns)-1, max(Ns)+1)
            plt.xlabel("Number of samples (log scale)", fontsize=30)
            plt.ylabel("Indices in Sobol method", fontsize=30)
            plt.title(r"First Order Sobol Index S$_i$ in " + self.model + " Model for different sample size (log10 scale)", fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        plt.show()

    def S2_Sobol_plot(self, sobol_dict):

        """ Function to plot second order sobol index matrices for different sample sizes on log
        scale"""

        Ns = list()
        S2_dataframe = list()

        for item in sobol_dict:
            Ns.append(float(item['sample size']))
            S2_dataframe.append(item['Second Order (DataFrame)'])


        for i in range(len(Ns)):
            S2_Sobol = ast.literal_eval(S2_dataframe[i])
            df = pd.DataFrame(S2_Sobol)
            if len(df.columns) < 10:
                plt.figure(figsize=(12,12))
            else:
                plt.figure(figsize=(15,15))
            g = sns.heatmap(df, cmap="Blues", annot=True, xticklabels=df.columns, yticklabels=df.columns, annot_kws={"fontsize":10})
            plt.title("Second Order index for sample size {0} (log10 scale)".format(Ns[i]), fontsize=30)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()


    def clocktime_sobol_plot(self, sobol_dict):

        """ Function to plot Runtime versus log N plots for each computation """

        Ns = list()
        sample_time_sobol = list()
        analysis_time_sobol = list()

        for item in sobol_dict:
            Ns.append(float(item['sample size']))
            sample_time_sobol.append(float(item['Sample Time']))
            analysis_time_sobol.append(float(item['Analysis Time']))

        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111)

        ax.scatter(Ns, sample_time_sobol, color='r', s=50)
        ax.plot(Ns, sample_time_sobol, color ='black', label='Sample Time Sobol')
        ax.scatter(Ns, analysis_time_sobol, color='r', s=50)
        ax.plot(Ns, analysis_time_sobol, color ='g', label='Analysis Time Sobol')
        plt.legend()
        plt.xlabel('Number of Samples (log10 scale)', fontsize=30)
        plt.ylabel('Runtime (in seconds)', fontsize=30)
        plt.title('Time taken for computation (Sampling, Analysis) of Sobol Indices ('  + self.model + ') as a function of sample size (log10 scale)', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()


