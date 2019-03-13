import importlib
import pytest
import json

import numpy as np

from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork, NodeType


def test_crop_yield_creation_and_execution():
    filepath = "../../tests/data/crop_yield.f"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.inputs) == 3
    assert len(G.outputs) == 1

    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == -1       # TODO: update this with a good value


def test_petpt_creation():
    filepath = "../../tests/data/PETPT.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.inputs) == 5
    assert len(G.outputs) == 1


def test_petpt_execution():
    lambdas = importlib.__import__("PETPT_lambdas")
    pgm = json.load(open("PETPT.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == 0.02998371219618677


def test_petpt_numpy_execution():
    lambdas = importlib.__import__("PETPT_numpy_lambdas")
    pgm = json.load(open("PETPT_numpy.json", "r"))
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
    assert all([G[n]["value"] is not None for n in G.nodes()
                if G[n]["type"]] == NodeType.VARIABLE)
    G.clear()
    assert all([G[n]["value"] is None for n in G.nodes()
                if G[n]["type"]] == NodeType.VARIABLE)
