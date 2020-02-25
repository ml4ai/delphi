import numpy as np
import pandas as pd
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN
from delphi.GrFN.sensitivity import SensitivityIndices, SensitivityAnalyzer
from delphi.GrFN.visualization import SensitivityVisualizer

def sensitivity(model, N, B, method):

    tG = GrFN.from_fortran_file(f"../../tests/data/program_analysis/{model}.for", save_file=True)

    if method == 'Sobol':
        (sobol_dict, timing_data) = SensitivityAnalyzer.Si_from_Sobol(N, tG, B, save_time= True)
    elif method == 'FAST':
        (sobol_dict, timing_data) = SensitivityAnalyzer.Si_from_FAST(N, tG, B, save_time= True)
    elif method == 'RBD FAST':
        (sobol_dict, timing_data) = SensitivityAnalyzer.Si_from_RBD_FAST(N, tG, B, save_time= True)
    else:
        print('Method not known!')
        exit(0)

    (sample_time, exec_time, analysis_time) = timing_data

    return sobol_dict.__dict__, sample_time, exec_time, analysis_time


def generate_indices_for_plots(model, B, sample_list, method):

    var_names = B.keys()
    sobol_indices_lst = list()

    for i in range(len(sample_list)):
        Si, sample_time, exec_time, analysis_time = sensitivity(model, sample_list[i], B, method)
        S1_dict = dict(zip(var_names, Si['O1_indices'].tolist()))

        for k in range(Si['O2_indices'].shape[0]):
            for l in range(k, Si['O2_indices'].shape[1]):
                if k != l:
                    Si['O2_indices'][l][k] =  Si['O2_indices'][k][l]

        Si['O2_indices']  = np.nan_to_num(Si['O2_indices']).tolist()

        S2_dataframe = pd.DataFrame(data=Si['O2_indices'], columns=var_names)

        sobol_dict = {'sample size':sample_list[i], 'S1': S1_dict, 'S2':
                S2_dataframe, 'sampling time':  sample_time, 'execution time':
                exec_time, 'analysis time': analysis_time}

        sobol_indices_lst.append(sobol_dict)

    return sobol_indices_lst


def sensitivity_visualization(model, B, sample_list, method):

    indices_lst = generate_indices_for_plots(model, B, sample_list, method)

    plots = SensitivityVisualizer(indices_lst)

    plots.S1_plot()

    plots.S2_plot()

    plots.clocktime_plot()


def PETPT_inputs():

    B = {
        'tmax':[-30.0, 60.0],
        'tmin':[-30.0, 60.0],
        'srad': [0.0, 30.0],
        'msalb': [0.0, 1.0],
        'xhlai': [0.0, 20.0]
    }

    return B

def PETPNO_inputs():

    B = {
        'tmax':[-30.0, 60.0],
        'tmin':[-30.0, 60.0],
        'srad': [0.0, 30.0],
        'msalb': [0.0, 1.0],
        'xhlai': [0.0, 20.0],
        'tavg': [-30, 60],
        'tdew': [-30, 60],
        'windsp': [0.0, 10.0],
        'clouds': [0.0, 1.0]
    }


    return B

def PETPEN_inputs():

    B = {
        'tmax':[-30.0, 60.0],
        'tmin':[-30.0, 60.0],
        'srad': [0.0, 30.0],
        'msalb': [0.0, 1.0],
        'xhlai': [0.0, 20.0],
        'tavg': [-30, 60],
        'tdew': [-30, 60],
        'windsp': [1.0, 10.0],
        'windht': [1.0, 25.0],
        'vapr': [0.0, 20.0],
        'clouds': [0.0, 1.0],
        'eoratio': [0.0, 2.0]
    }


    return B


def PETDYN_inputs():

    B = {
        'tmax':[-30.0, 60.0],
        'tmin':[-30.0, 60.0],
        'srad': [0.0, 30.0],
        'msalb': [0.0, 1.0],
        'xhlai': [0.0, 20.0],
        'tavg': [-30, 60],
        'tdew': [-30, 60],
        'windsp': [0.0, 10.0],
        'canht': [0.0, 5.0],
        'clouds': [0.0, 1.0]
    }

    return B


def PETASCE_inputs():

    B = {
        "doy": [1, 365],
        "meevp": [0, 1],
        "msalb": [0, 1],
        "srad": [1, 30],
        "tmax": [-30, 60],
        "tmin": [-30, 60],
        "xhlai": [0, 20],
        "tdew": [-30, 60],
        "windht": [0.1, 10],
        "windrun": [0, 900],
        "xlat": [3, 12],
        "xelev": [0, 6000],
        "canht": [0.001, 3],
    }

    return B


if __name__ == '__main__':

    # These are working -- no bugs
    sensitivity_visualization('PETPT', PETPT_inputs(), [10, 100, 1000, 10000], 'Sobol')
    sensitivity_visualization('PETPNO', PETPNO_inputs(), [10, 100, 1000, 10000], 'Sobol')
    sensitivity_visualization('PETPEN', PETPEN_inputs(), [10, 100, 1000, 10000], 'Sobol')
    sensitivity_visualization('PETDYN', PETDYN_inputs(), [10, 100, 1000, 10000], 'Sobol')

    # Not Working!
    sensitivity_visualization('PETPT', PETPT_inputs(), [100, 1000, 10000], 'FAST')

    # Not Working!
    sensitivity_visualization('PETPT', PETPT_inputs(), [100, 1000, 10000], 'RBD FAST')
