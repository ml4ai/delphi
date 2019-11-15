from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')

filenames = glob('*.txt')

for filename in filenames:
    data = np.loadtxt(filename)
    
    varname1 = filename.split('_')[0]
    varname2 = filename.split('_')[1]
    model = filename.split('_')[2]
    var1_lb = data[:,0]
    var1_ub = data[:,1]

    var2_lb = data[:,2]
    var2_ub = data[:,3]

    # print(tmax_lb, tmax_ub, tmin_lb, tmin_ub)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(len(var1_ub)):
        ax.add_patch(patches.Rectangle((var1_lb[i], var2_lb[i]), var1_ub[i]-var1_lb[i], var2_ub[i]-var2_lb[i]))
    plt.title('2D plot of allowed ' + varname1 + ' and ' + varname2 + ' in ' + model)
    plt.xlabel(varname1)
    plt.ylabel(varname2)
    plt.ylim(min(var2_lb)-5, max(var2_ub)+5)
    plt.xlim(min(var1_lb)-5, max(var1_ub)+5)
    plt.savefig(varname1 + '_' + varname2 + '_' + model + '_eo5.5-10.png')    
    # plt.show()


