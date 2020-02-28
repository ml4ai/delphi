from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')


def plot(filename):
    
    data = np.loadtxt(filename)
    
    varname1 = filename.split('_')[0].split('/')[-1]
    varname2 = filename.split('_')[1]
    model = filename.split('_')[2]
    var1_lb = data[:,0]
    var1_ub = data[:,1]

    var2_lb = data[:,2]
    var2_ub = data[:,3]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(len(var1_ub)):
        ax.add_patch(patches.Rectangle((var1_lb[i], var2_lb[i]), var1_ub[i]-var1_lb[i], var2_ub[i]-var2_lb[i], edgecolor = (0.78, 0.24, 0.52)))
    plt.title('2D plot of allowed ' + varname1 + ' and ' + varname2 + ' in ' + model, fontsize=20)
    plt.xlabel(varname1, fontsize=30)
    plt.ylabel(varname2, fontsize=30)
    plt.ylim(min(var2_lb)-5, max(var2_ub)+5)
    plt.xlim(min(var1_lb)-5, max(var1_ub)+5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

# filepath = '/Users/souratoshkhan/delphi/scripts/khan/Ibex/'
# filename = filepath + 'tmax_tmin_petpt_intervals.txt'
# plot(filename)
