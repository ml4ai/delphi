import importlib
import pytest
import json
import sys

import numpy as np
import torch

from delphi.GrFN.networks import GroundedFunctionNetwork, NodeType

data_dir = "tests/data/GrFN/"
sys.path.insert(0, "tests/data/GrFN/")


def test_crop_yield_creation_and_execution():
    lambdas = importlib.__import__("crop_yield_lambdas")
    pgm = json.load(open(data_dir + "crop_yield.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.inputs) == 6           # TODO: update this later
    assert len(G.outputs) == 2          # TODO: update this later


def test_petpt_creation_and_execution():
    filepath = "tests/data/GrFN/PETPT.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)
    print(G)        # Shadow testing

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.model_inputs) == 5
    assert len(G.outputs) == 1

    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == 0.02998371219618677


def test_petasce_creation():
    filepath = "tests/data/GrFN/PETASCE_simple.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)
    print(G)        # Shadow testing
    A = G.to_agraph()
    # A.draw("petasce.pdf", prog='dot')
    CAG = G.to_CAG_agraph()
    # CAG.draw("petasce_CAG.pdf", prog='dot')
    CG = G.to_call_agraph()
    # CG.draw("petasce_call_graph.pdf", prog='dot')

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


def test_petasce_torch_execution():
    lambdas = importlib.__import__("PETASCE_simple_torch_lambdas")
    pgm = json.load(open(data_dir + "PETASCE_simple_torch.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    # bounds = {
    #     "petasce::msalb_0": [0, 1],
    #     "petasce::srad_0": [1, 30],
    #     "petasce::tmax_0": [-30, 60],
    #     "petasce::tmin_0": [-30, 60],
    #     "petasce::xhlai_0": [0, 20],
    #     "petasce::tdew_0": [-30, 60],
    #     "petasce::windht_0": [0, 10],
    #     "petasce::windrun_0": [0, 900],
    #     "petasce::xlat_0": [0, 90],
    #     "petasce::xelev_0": [0, 6000],
    #     "petasce::canht_0": [0.001, 3],
    # }

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
    print(res)


def test_petpt_numpy_execution():
    lambdas = importlib.__import__("PETPT_numpy_lambdas")
    pgm = json.load(open(data_dir + "PETPT_numpy.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
    values = {
        "petpt::msalb_0": np.random.rand(1000),
        "petpt::srad_0": np.random.randint(1, 100, size=1000),
        "petpt::tmax_0": np.random.randint(30, 40, size=1000),
        "petpt::tmin_0": np.random.randint(10, 15, size=1000),
        "petpt::xhlai_0": np.random.rand(1000)
    }
    result = G.run(values)
    assert result.shape == (1000,)
    assert all([G.nodes[n]["value"] is not None for n in G.nodes()
                if G.nodes[n]["type"] == NodeType.VARIABLE])
    G.clear()
    assert all([G.nodes[n]["value"] is None for n in G.nodes()
                if G.nodes[n]["type"] == NodeType.VARIABLE])


def test_ProgramAnalysisGraph_from_GrFN():
    sys.path.insert(0, "tests/data/GrFN/")
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
    A = G.to_agraph()
    CAG = G.to_CAG_agraph()
    CG = G.to_call_agraph()
