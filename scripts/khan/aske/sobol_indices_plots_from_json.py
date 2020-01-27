import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import seaborn as sns
sns.set_style('whitegrid')

class Sobol_Index_Plots:

    def __init__(self, filename, model):

        self.filename = filename
        self.model = model

    def index_from_dict(self):

        data = open(self.filename, encoding='utf-8').read()
        js = json.loads(data)

        sample_size = list()
        S1_Sobol = list()
        S2_data = list()
        clocktime_sobol = list()

        for item in js:
            sample_size.append(float(item['sample size']))
            S1_Sobol.append(item['First Order'])
            S2_data.append(item['Second Order (DataFrame)'])
            clocktime_sobol.append(float(item['Clocktime']))
    
        return sample_size, S1_Sobol, S2_data, clocktime_sobol
        
    def S1_Sobol_plot(self):

        sample_size, S1_Sobol = self.index_from_dict()[0], self.index_from_dict()[1]

        S1_dataframe = pd.DataFrame(S1_Sobol)
        S1_dataframe.index = sample_size
    
        cols = list(S1_dataframe.columns)

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        for col in cols:
            S1_dataframe.reset_index().plot(kind='scatter', x="index", y=col, ax=ax, c ='r', s=50)
            S1_dataframe.reset_index().plot(kind='line', x="index", y=col, ax=ax)
            plt.legend(loc='right', fontsize=20)
            plt.xlim(min(sample_size)-1, max(sample_size)+1)
            plt.xlabel("Number of samples (log scale)", fontsize=30)
            plt.ylabel("Indices in Sobol method", fontsize=30)
            plt.title(r"First Order Sobol Index S$_i$ in " + self.model + " Model for different sample size (log10 scale)", fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        plt.show()

    def S2_Sobol_plot(self):
    
        sample_size, S2_dataframe = self.index_from_dict()[0], self.index_from_dict()[2]

        for i in range(len(sample_size)):
            S2_Sobol = ast.literal_eval(S2_dataframe[i])
            df = pd.DataFrame(S2_Sobol)
            if len(df.columns) < 10:
                plt.figure(figsize=(12,12))
            else:
                plt.figure(figsize=(15,15))
            g = sns.heatmap(df, cmap="Blues", annot=True, xticklabels=df.columns, yticklabels=df.columns, annot_kws={"fontsize":10})
            plt.title("Second Order index for sample size {0} (log10 scale)".format(sample_size[i]), fontsize=30)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()


    def clocktime_sobol_plot(self):
    
        sample_size, clocktime_sobol = self.index_from_dict()[0], self.index_from_dict()[3]

        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111)

        ax.scatter(sample_size, clocktime_sobol, label='Sobol', color='r', s=50)
        ax.plot(sample_size, clocktime_sobol, color ='black')
        plt.legend()
        plt.xlabel('Number of Samples (log10 scale)', fontsize=30)
        plt.ylabel('Clocktime (in seconds)', fontsize=30)
        plt.title('Time taken for computation of Sobol Indices ('  + self.model + ') as a function of sample size', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()


# if __name__ == '__main__':

    # sobol = Sobol_Index_Plots('sobol_indices_PETPT.json', 'PETPT')
    # sobol.S1_Sobol_plot()
    # sobol.S2_Sobol_plot()
    # sobol.clocktime_sobol_plot()

    # sobol = Sobol_Index_Plots('sobol_indices_PETASCE.json', 'PETASCE')
    # sobol.S1_Sobol_plot()
    # sobol.S2_Sobol_plot()
    # sobol.clocktime_sobol_plot()
