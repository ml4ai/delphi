from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')

def plot(filename):


    data = np.loadtxt(filename)

    var1_lb = list(set(data[:,0]))
    var1_ub = list(set(data[:,1]))


    varname1 = 'x values'
    varname2 = '  '

    ydata = [0]*len(var1_lb)
    # print(len(var1_lb), len(ydata))
    plt.figure(figsize=(8,8))
    plt.scatter(var1_lb, ydata, color = 'r', lw=2)
    plt.title('1D plot of allowed ' + varname1, fontsize=20)
    plt.xlabel(varname1, fontsize=20)
    plt.ylabel(varname2)
    plt.xlim(min(var1_lb)-10, max(var1_lb)+10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

# filename = '/Users/souratoshkhan/delphi/scripts/khan/Ibex/intervals.txt'
# plot(filename)
