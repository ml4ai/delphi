import os
import pytest

import numpy as np

from delphi.GrFN.sensitivity import SensitivityIndices, SensitivityAnalyzer
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN
from test_GrFN import petpt_grfn, petasce_grfn


@pytest.fixture
def Si_Obj():
    return SensitivityIndices({
        'S1':[0.5, 0.5],
        'S2': [[0.5, 0.2], [0.1, 0.8]],
        'ST':[0.75, 0.25],
        'S1_conf':[-0.05, 0.05],
        'S2_conf':[-0.05, 0.05],
        'ST_conf':[-0.05, 0.05]
    }, {"names": ["x1", 'x2']})


def test_check_order_functions(Si_Obj):
    assert Si_Obj.check_first_order()
    assert Si_Obj.check_second_order()
    assert Si_Obj.check_total_order()


def test_min_max_S2(Si_Obj):
    assert Si_Obj.get_min_S2() == 0.1
    assert Si_Obj.get_max_S2() == 0.8


def test_roundtrip_json_serialization(Si_Obj):
    json_filepath = "tests/data/GrFN/test_example_SI.json"
    Si_Obj.to_json_file(json_filepath)
    new_Si = SensitivityIndices.from_json_file(json_filepath)

    assert Si_Obj != new_Si
    assert Si_Obj.parameter_list == new_Si.parameter_list
    assert all([x1 == x2 for x1, x2 in zip(Si_Obj.O1_indices, new_Si.O1_indices)])
    assert all([x1 == x2 for x1, x2 in zip(Si_Obj.OT_indices, new_Si.OT_indices)])

    assert all([x1 == x2 for x1, x2 in zip(Si_Obj.O1_confidence, new_Si.O1_confidence)])
    assert all([x1 == x2 for x1, x2 in zip(Si_Obj.OT_confidence, new_Si.OT_confidence)])

    assert all([
        x1 == x2 for arr1, arr2 in zip(Si_Obj.O2_indices, new_Si.O2_indices)
        for x1, x2 in zip(arr1, arr2)
    ])
    assert all([x1 == x2 for x1, x2 in zip(Si_Obj.O2_confidence, new_Si.O2_confidence)])
    os.remove(json_filepath)



def test_Sobol(petpt_grfn):
    N = 1000
    B = {
        "tmax":[0.0, 40.0],
        "tmin":[0.0, 40.0],
        "srad": [0.0, 30.0],
        "msalb": [0.0, 1.0],
        "xhlai": [0.0, 20.0]
    }

    (indices, timing_data) = SensitivityAnalyzer.Si_from_Sobol(
        N, petpt_grfn, B, save_time=True
    )

    (sample_time_sobol,
     exec_time_sobol,
     analyze_time_sobol) = timing_data
