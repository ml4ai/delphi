from glob import glob
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')

def plot(filename):


    data = np.loadtxt(filename)
    
    model = filename.split('/')[-1].split('_')[1] + ' ' + filename.split('_')[2].split('.')[0]

    var1_lb = data[:,0]
    var1_ub = data[:,1]

    var1_ub[var1_ub == inf] = 1000

    # print(var1_lb, var1_ub)

    varname1 = 'Range of Function'
    varname2 = '  '

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(len(var1_ub)):
        ax.axhline(y=0, xmin=var1_lb[i], xmax=var1_ub[i], color = 'r', linestyle='-')
    plt.title(' Range of ' + model, fontsize=20)
    plt.xlabel(varname1, fontsize=20)
    plt.ylabel(varname2, fontsize=20)
    plt.xlim(min(var1_lb)-10, max(var1_ub)+10)
    plt.ylim(-100, 100)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig(model + '.png')    
    plt.show()

# filename = '/Users/souratoshkhan/delphi/scripts/khan/Ibex/intervals_inverse_func.txt'
# plot(filename)
