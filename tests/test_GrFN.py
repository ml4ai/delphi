import importlib
import pytest
import json
import sys

import numpy as np
import torch

from delphi.translators.for2py.floatNumpy import Float32
from delphi.GrFN.networks import GroundedFunctionNetwork

data_dir = "tests/data/GrFN/"
sys.path.insert(0, "tests/data/program_analysis")

@pytest.fixture
def crop_yield_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/crop_yield.f")

@pytest.fixture
def petpt_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/PETPT.for")

@pytest.fixture
def petasce_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/PETASCE_simple.for")


def test_petpt_creation_and_execution(petpt_grfn):
    assert isinstance(petpt_grfn, GroundedFunctionNetwork)
    assert len(petpt_grfn.inputs) == 5
    assert len(petpt_grfn.outputs) == 1

    values = {name: 1.0 for name in petpt_grfn.inputs}
    res = petpt_grfn.run(values)
    assert res == Float32(0.029983712)


def test_petasce_creation(petasce_grfn):
    A = petasce_grfn.to_agraph()
    CAG = petasce_grfn.to_CAG_agraph()
    CG = petasce_grfn.to_call_agraph()

    values = {
        "petasce::doy_-1": 20.0,
        "petasce::meevp_-1": "A",
        "petasce::msalb_-1": 0.5,
        "petasce::srad_-1": 15.0,
        "petasce::tmax_-1": 10.0,
        "petasce::tmin_-1": -10.0,
        "petasce::xhlai_-1": 10.0,
        "petasce::tdew_-1": 20.0,
        "petasce::windht_-1": 5.0,
        "petasce::windrun_-1": 450.0,
        "petasce::xlat_-1": 45.0,
        "petasce::xelev_-1": 3000.0,
        "petasce::canht_-1": 2.0,
    }

    res = petasce_grfn.run(values)
    assert res == Float32(0.00012496980836348878)


def test_crop_yield_creation(crop_yield_grfn):
    A = crop_yield_grfn.to_agraph()
    assert isinstance(crop_yield_grfn, GroundedFunctionNetwork)


@pytest.mark.skip
def test_petasce_torch_execution():
    lambdas = importlib.__import__("PETASCE_simple_torch_lambdas")
    pgm = json.load(open(data_dir + "PETASCE_simple_torch.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    N = 100
    samples = {
        "petasce::doy_0": np.random.randint(1, 100, N),
        "petasce::meevp_0": np.where(np.random.rand(N) >= 0.5, 'A', 'W'),
        "petasce::msalb_0": np.random.uniform(0, 1, N),
        "petasce::srad_0": np.random.uniform(1, 30, N),
        "petasce::tmax_0": np.random.uniform(-30, 60, N),
        "petasce::tmin_0": np.random.uniform(-30, 60, N),
        "petasce::xhlai_0": np.random.uniform(0, 20, N),
        "petasce::tdew_0": np.random.uniform(-30, 60, N),
        "petasce::windht_0": np.random.uniform(0, 10, N),
        "petasce::windrun_0": np.random.uniform(0, 900, N),
        "petasce::xlat_0": np.random.uniform(0, 90, N),
        "petasce::xelev_0": np.random.uniform(0, 6000, N),
        "petasce::canht_0": np.random.uniform(0.001, 3, N),
    }

    values = {
        k: torch.tensor(v, dtype=torch.double) if v.dtype != "<U1" else v
        for k, v in samples.items()
    }

    res = G.run(values, torch_size=N)
    assert res.size()[0] == N
