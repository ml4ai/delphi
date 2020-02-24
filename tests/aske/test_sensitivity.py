import pytest

import numpy as np

from delphi.GrFN.sensitivity import SensitivityIndices, SensitivityAnalyzer
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN
from test_GrFN import petpt_grfn, petasce_grfn


@pytest.fixture
def Si_Obj():
    return SensitivityIndices({
        'S1':0.5,
        'S2': np.array([[0.5, 0.2], [0.1, 0.8]]),
        'ST':1.0,
        'S1_conf':0.05,
        'S2_conf':0.05,
        'ST_conf':0.05
    })


def test_check_order_functions(Si_Obj):
    assert Si_Obj.check_first_order()
    assert Si_Obj.check_second_order()
    assert Si_Obj.check_total_order()


def test_min_max_S2(Si_Obj):
    assert Si_Obj.get_min_S2() == 0.1
    assert Si_Obj.get_max_S2() == 0.8

@pytest.mark.skip("TODO Khan: add the files for these tests")
def test_from_file(Si_Obj):
    assert isinstance(Si_Obj.from_csv('test.csv'), dict)
    assert isinstance(Si_Obj.from_json('test.json'), dict)
    assert isinstance(Si_Obj.from_pickle('test_pickle'), dict)


def test_Sobol(petpt_grfn):
    N = 1000
    B = {
        'PETPT::@global::petpt::0::tmax::-1':[0.0, 40.0],
        'PETPT::@global::petpt::0::tmin::-1':[0.0, 40.0],
        'PETPT::@global::petpt::0::srad::-1': [0.0, 30.0],
        'PETPT::@global::petpt::0::msalb::-1': [0.0, 1.0],
        'PETPT::@global::petpt::0::xhlai::-1': [0.0, 20.0]
    }

    (indices, timing_data) = SensitivityAnalyzer.Si_from_Sobol(
        N, petpt_grfn, B, save_time=True
    )

    (sample_time_sobol,
     exec_time_sobol,
     analyze_time_sobol) = timing_data

    assert isinstance(indices, SensitivityIndices)
    assert isinstance(sample_time_sobol, float)
    assert isinstance(exec_time_sobol, float)
    assert isinstance(analyze_time_sobol, float)
