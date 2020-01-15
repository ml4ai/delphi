import petpt_petasce_plots as pt
import  model_compute_from_dict as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plots(size, shared_var_bounds, non_shared_var_bounds):


    tmax = tmin = srad = msalb = xhlai = np.zeros(size)


    shared_df = pd.read_csv(shared_var_bounds, sep='\s+')
    shared_dict = pd.Series(shared_df.Vals.values, index=shared_df.Var).to_dict()
    # print(shared_df)
    # print(shared_dict)

    petpt_var_names = list(set([name.split('_')[0] for name in shared_dict.keys()]))
    # print(petpt_var_names)

    petpt_bounds = dict()

    for var_name in petpt_var_names:
        if shared_dict[var_name + '_lb'] == shared_dict[var_name +  '_ub']:
            val = np.full(size, float(shared_dict[var_name + '_lb']), dtype =float)
        else:
            val = np.linspace(float(shared_dict[var_name + '_lb']), float(shared_dict[var_name + '_ub']), size)
        petpt_bounds.update({var_name:val})

    # print(petpt_bounds)



    non_shared_df = pd.read_csv(non_shared_var_bounds, sep='\s+', error_bad_lines=False, low_memory=False)
    non_shared_dict = pd.Series(non_shared_df.Vals.values, index=non_shared_df.Var).to_dict()
    # print(non_shared_df)
    # print(non_shared_dict)

    petasce_bounds = petpt_bounds

    petasce_var_names = list(set([name.split('_')[0] for name in non_shared_dict.keys()]))
    # print(petasce_var_names)

    for var_name in petasce_var_names:
        if var_name != 'meevp':
            if non_shared_dict[var_name + '_lb'] == non_shared_dict[var_name +  '_ub']:
                val = np.full(size, float(non_shared_dict[var_name + '_lb']), dtype=float)
            else:
                val = np.linspace(float(non_shared_dict[var_name + '_lb']), float(non_shared_dict[var_name + '_ub']), size)
        petasce_bounds.update({var_name:val})

    petasce_var_names.append('meevp')
    petasce_bounds.update({'meevp':np.full(size, 'G', dtype=str)})
    # print(petasce_bounds)

    vfunc1 = np.vectorize(dt.PETPT)
    vfunc2 = np.vectorize(dt.PETASCE)
    y1 = vfunc1(**petpt_bounds)
    y2 = vfunc2(**petasce_bounds)
    x = list(range(1, len(y1)+1))

    # print('y1 vals: ', y1)
    # print('y2 vals: ', y2)


    # print(petpt_var_names)

    petasce_var_names = petasce_var_names + petpt_var_names
    # print(petasce_var_names)

    stacked_arr = np.column_stack((petasce_bounds['tmax'].round(2),
        petasce_bounds['tmin'].round(2),
        petasce_bounds['srad'].round(2), petasce_bounds['msalb'].round(2),
        petasce_bounds['xhlai'].round(2), petasce_bounds['tdew'].round(2),
        petasce_bounds['windrun'].round(2), petasce_bounds['xlat'].round(2),
        petasce_bounds['canht'].round(2), petasce_bounds['windht'].round(2),
        petasce_bounds['doy'].round(2), petasce_bounds['xelev'].round(2), petasce_bounds['meevp']))

    # stacked_arr = stacked_arr.round(2)
    # print(stacked_arr)

    names = np.array(petasce_var_names)
    # print(names)

    annot_dict = list(map(dict, np.dstack((np.repeat(names[None, :], size, axis=0), stacked_arr ))))
    # print(annot_dict)

    interact_plot = pt.Interactive_Plot(x, y1, y2, annot_dict)
    interact_plot.fig_plot()

    return x, y1, y2, annot_dict 

if __name__ == '__main__':

    num = 25
    shared_vars_file = 'shared_vars_bounds.txt'
    non_shared_vars_file = 'non_shared_vars_bounds.txt'

    plots(num, shared_vars_file, non_shared_vars_file)



