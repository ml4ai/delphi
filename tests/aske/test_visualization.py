import os
import numpy as np
import pandas as pd
import pytest
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN
from delphi.GrFN.sensitivity import SensitivityAnalyzer
from delphi.GrFN.visualization import SensitivityVisualizer


@pytest.fixture
def sensitivity_visualizer():

    N = [10, 100, 1000, 10000]
    tG = GrFN.from_fortran_file("tests/data/program_analysis/PETPT.for")
    var_bounds = {
        "tmax": [-30.0, 60.0],
        "tmin": [-30.0, 60.0],
        "srad": [0.0, 30.0],
        "msalb": [0.0, 1.0],
        "xhlai": [0.0, 20.0],
    }

    sensitivity_indices_lst = []

    var_names = var_bounds.keys()

    for i in range(len(N)):
        (Si, timing_data) = SensitivityAnalyzer.Si_from_Sobol(
            N[i], tG, var_bounds, save_time=True
        )
        (sample_time, exec_time, analysis_time) = timing_data
        sobol_dict = Si.__dict__
        S1_dict = dict(zip(var_names, sobol_dict["O1_indices"].tolist()))

        for k in range(sobol_dict["O2_indices"].shape[0]):
            for l in range(k, sobol_dict["O2_indices"].shape[1]):
                if k != l:
                    sobol_dict["O2_indices"][l][k] = sobol_dict["O2_indices"][
                        k
                    ][l]

        sobol_dict["O2_indices"] = np.nan_to_num(
            sobol_dict["O2_indices"]
        ).tolist()

        S2_dataframe = pd.DataFrame(
            data=sobol_dict["O2_indices"], columns=var_names
        )

        sobol_dict_visualizer = {
            "sample size": np.log10(N[i]),
            "S1": S1_dict,
            "S2": S2_dataframe,
            "sampling time": sample_time,
            "execution time": exec_time,
            "analysis time": analysis_time,
        }

        sensitivity_indices_lst.append(sobol_dict_visualizer)

    yield SensitivityVisualizer(sensitivity_indices_lst)
    # Cleanup actions
    os.remove("s1_plot.pdf")
    os.remove("s2_plot.pdf")
    os.remove("clocktime_plot.pdf")

def test_sensitivity_visualization(sensitivity_visualizer):
    sensitivity_visualizer.create_S1_plot()
    sensitivity_visualizer.create_S2_plot()
    sensitivity_visualizer.create_clocktime_plot()
