import importlib
import pytest
import json
import sys

import numpy as np
import torch

from delphi.GrFN.networks import GroundedFunctionNetwork, NodeType

data_dir = "tests/data/GrFN/"
sys.path.insert(0, "tests/data/GrFN/")


def test_petpt_creation_and_execution():
    filepath = "tests/data/GrFN/PETPT.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.model_inputs) == 5
    assert len(G.outputs) == 1

    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == 0.02998371219618677


def test_petasce_creation():
    filepath = "tests/data/GrFN/PETASCE_simple.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)
    A = G.to_agraph()
    CAG = G.to_CAG_agraph()
    CG = G.to_call_agraph()

    values = {
        "petasce::doy_0": 20,
        "petasce::meevp_0": "A",
        "petasce::msalb_0": 0.5,
        "petasce::srad_0": 15,
        "petasce::tmax_0": 10,
        "petasce::tmin_0": -10,
        "petasce::xhlai_0": 10,
        "petasce::tdew_0": 20,
        "petasce::windht_0": 5,
        "petasce::windrun_0": 450,
        "petasce::xlat_0": 45,
        "petasce::xelev_0": 3000,
        "petasce::canht_0": 2,
    }

    res = G.run(values)
    assert res == 0.00012496980836348878


@pytest.mark.skip
def test_petasce_torch_execution():
    lambdas = importlib.__import__("PETASCE_simple_torch_lambdas")
    with open(data_dir + "PETASCE_simple_torch.json", "r") as f:
        pgm = json.load(f)
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
