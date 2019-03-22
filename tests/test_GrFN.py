import importlib
import pytest
import json
import sys

import numpy as np

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
    assert len(G.inputs) == 5
    assert len(G.outputs) == 1

    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == 0.02998371219618677


def test_petasce_creation_and_execution():
    filepath = "tests/data/GrFN/PETASCE_simple.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)
    print(G)        # Shadow testing

    assert isinstance(G, GroundedFunctionNetwork)
    assert len(G.inputs) == 13
    assert len(G.outputs) == 1

    values = {name: 1 for name in G.inputs}
    res = G.run(values)
    assert res == 0.02998371219618677

test_petasce_creation_and_execution()

# TODO: Figure this thing out
# def test_petasce_creation():
#     filepath = "delph/translators/for2py/data/PETASCE.py"
#     stem = Path(filepath).stem
#     lambdas_path = f"tests/data/GrFN/{stem}_lambdas.py"
#     json_filename = f"tests/data/GrFN/{stem}.json"
#     G = GroundedFunctionNetwork.from_python_src(filepath, lambdas_path, json_filename)
#
#     assert isinstance(G, GroundedFunctionNetwork)


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
