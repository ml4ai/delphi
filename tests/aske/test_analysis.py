import pytest
import inspect
import importlib
import json
import sys

from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.sensitivity import FAST_analysis, RBD_FAST_analysis
import delphi.GrFN.analysis as analysis
from test_GrFN import petpt_grfn, petasce_grfn
import delphi.translators.for2py.f2grfn as f2grfn

import numpy as np


sys.path.insert(0, "tests/data/program_analysis")


def test_regular_PETPT(petpt_grfn):
    args = petpt_grfn.inputs
    bounds = {
        "PETPT::@global::petpt::0::msalb::-1": [0, 1],
        "PETPT::@global::petpt::0::srad::-1": [1, 20],
        "PETPT::@global::petpt::0::tmax::-1": [-30, 60],
        "PETPT::@global::petpt::0::tmin::-1": [-30, 60],
        "PETPT::@global::petpt::0::xhlai::-1": [0, 20],
    }

    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    Ns = 1000
    Si = petpt_grfn[0].sobol_analysis(Ns, problem)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)


@pytest.mark.skip("Need to update to latest JSON")
def test_PETPT_with_torch():
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/program_analysis/PETPT.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    args = G.inputs
    bounds = {
        "petpt::msalb_-1": [0, 1],
        "petpt::srad_-1": [1, 20],
        "petpt::tmax_-1": [-30, 60],
        "petpt::tmin_-1": [-30, 60],
        "petpt::xhlai_-1": [0, 20],
    }

    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    Ns = 1000                      # TODO: Khan, experiment with this value
    Si = G.sobol_analysis(Ns, problem, use_torch=True)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)

    Si = FAST_analysis(G, Ns, problem)
    assert len(Si.keys()) == 3
    assert len(Si["S1"]) == len(args)

    Si = RBD_FAST_analysis(G, Ns, problem)
    assert len(Si.keys()) == 2
    assert len(Si["S1"]) == len(args)


def test_PETASCE_sobol_analysis(petasce_grfn):
    bounds = {
        "PETASCE_simple::@global::petasce::0::doy::-1": [1, 365],
        "PETASCE_simple::@global::petasce::0::meevp::-1": [0, 1],
        "PETASCE_simple::@global::petasce::0::msalb::-1": [0, 1],
        "PETASCE_simple::@global::petasce::0::srad::-1": [1, 30],
        "PETASCE_simple::@global::petasce::0::tmax::-1": [-30, 60],
        "PETASCE_simple::@global::petasce::0::tmin::-1": [-30, 60],
        "PETASCE_simple::@global::petasce::0::xhlai::-1": [0, 20],
        "PETASCE_simple::@global::petasce::0::tdew::-1": [-30, 60],
        "PETASCE_simple::@global::petasce::0::windht::-1": [0.1, 10],  # HACK: has a hole in 0 < x < 1 for petasce__assign__wind2m_1
        "PETASCE_simple::@global::petasce::0::windrun::-1": [0, 900],
        "PETASCE_simple::@global::petasce::0::xlat::-1": [3, 12],     # HACK: south sudan lats
        "PETASCE_simple::@global::petasce::0::xelev::-1": [0, 6000],
        "PETASCE_simple::@global::petasce::0::canht::-1": [0.001, 3],
    }

    type_info = {
        "PETASCE_simple::@global::petasce::0::doy::-1": (int, list(range(1, 366))),
        "PETASCE_simple::@global::petasce::0::meevp::-1": (str, ["A", "W"]),
        "PETASCE_simple::@global::petasce::0::msalb::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::srad::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::tmax::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::tmin::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::xhlai::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::tdew::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::windht::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::windrun::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::xlat::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::xelev::-1": (float, [0.0]),
        "PETASCE_simple::@global::petasce::0::canht::-1": (float, [0.0]),
    }

    args = petasce_grfn.inputs
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    Si = petasce_grfn[0].sobol_analysis(100, problem, var_types=type_info)
    assert len(Si["S1"]) == len(petasce_grfn.inputs)
    assert len(Si["S2"][0]) == len(petasce_grfn.inputs)


@pytest.mark.skip("Need to update to latest JSON")
def test_PETASCE_with_torch():
    # Torch model
    sys.path.insert(0, "tests/data/GrFN")
    lambdas = importlib.__import__("PETASCE_simple_torch_lambdas")
    pgm = json.load(open("tests/data/GrFN/PETASCE_simple_torch.json", "r"))
    tG = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    bounds = {
        "petasce::doy_0": [1, 365],
        "petasce::meevp_0": [0, 1],
        "petasce::msalb_0": [0, 1],
        "petasce::srad_0": [1, 30],
        "petasce::tmax_0": [-30, 60],
        "petasce::tmin_0": [-30, 60],
        "petasce::xhlai_0": [0, 20],
        "petasce::tdew_0": [-30, 60],
        "petasce::windht_0": [0.1, 10],  # HACK: has a hole in 0 < x < 1 for petasce__assign__wind2m_1
        "petasce::windrun_0": [0, 900],
        "petasce::xlat_0": [3, 12],     # HACK: south sudan lats
        "petasce::xelev_0": [0, 6000],
        "petasce::canht_0": [0.001, 3],
    }

    type_info = {
        "petasce::doy_0": (int, list(range(1, 366))),
        "petasce::meevp_0": (str, ["A", "W"]),
        "petasce::msalb_0": (float, [0.0]),
        "petasce::srad_0": (float, [0.0]),
        "petasce::tmax_0": (float, [0.0]),
        "petasce::tmin_0": (float, [0.0]),
        "petasce::xhlai_0": (float, [0.0]),
        "petasce::tdew_0": (float, [0.0]),
        "petasce::windht_0": (float, [0.0]),
        "petasce::windrun_0": (float, [0.0]),
        "petasce::xlat_0": (float, [0.0]),
        "petasce::xelev_0": (float, [0.0]),
        "petasce::canht_0": (float, [0.0]),
    }

    args = tG.inputs
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    tSi = tG.sobol_analysis(1000, problem, var_types=type_info, use_torch=True)
    assert len(tSi["S1"]) == len(tG.inputs)
    assert len(tSi["S2"][0]) == len(tG.inputs)


def test_PETPT_sensitivity_surface(petpt_grfn):
    bounds = {
        "PETPT::@global::petpt::0::msalb::-1": (0, 1),
        "PETPT::@global::petpt::0::srad::-1": (1, 20),
        "PETPT::@global::petpt::0::tmax::-1": (-30, 60),
        "PETPT::@global::petpt::0::tmin::-1": (-30, 60),
        "PETPT::@global::petpt::0::xhlai::-1": (0, 20),
    }
    presets = {
        "PETPT::@global::petpt::0::msalb::-1": 0.5,
        "PETPT::@global::petpt::0::srad::-1": 10,
        "PETPT::@global::petpt::0::tmax::-1": 20,
        "PETPT::@global::petpt::0::tmin::-1": 10,
        "PETPT::@global::petpt::0::xhlai::-1": 10,
    }

    (X, Y, Z, x_var, y_var) = petpt_grfn.S2_surface((80, 60), bounds, presets)

    assert X.shape == (80,)
    assert Y.shape == (60,)
    assert Z.shape == (80, 60)


@pytest.mark.skip("Need to update FIB for new GrFN schema")
def test_FIB_execution(petpt_grfn, petasce_grfn):
    petpt_fib = petpt_grfn.to_FIB(petasce_grfn)
    petasce_fib = petasce_grfn.to_FIB(petpt_grfn)

    pt_inputs = {name: 1 for name in petpt_grfn.inputs}

    asce_inputs = {
        "PETASCE_simple::@global::petasce::0::msalb::-1": 0.5,
        "PETASCE_simple::@global::petasce::0::srad::-1": 15,
        "PETASCE_simple::@global::petasce::0::tmax::-1": 10,
        "PETASCE_simple::@global::petasce::0::tmin::-1": -10,
        "PETASCE_simple::@global::petasce::0::xhlai::-1": 10,
    }

    asce_covers = {
        "PETASCE_simple::@global::petasce::0::canht::-1": 2,
        "PETASCE_simple::@global::petasce::0::meevp::-1": "A",
        "PETASCE_simple::@global::petasce::0::cht::0": 0.001,
        "PETASCE_simple::@global::petasce::0::cn::4": 1600.0,
        "PETASCE_simple::@global::petasce::0::cd::4": 0.38,
        "PETASCE_simple::@global::petasce::0::rso::0": 0.062320,
        "PETASCE_simple::@global::petasce::0::ea::0": 7007.82,
        "PETASCE_simple::@global::petasce::0::wind2m::0": 3.5,
        "PETASCE_simple::@global::petasce::0::psycon::0": 0.0665,
        "PETASCE_simple::@global::petasce::0::wnd::0": 3.5,
    }

    res = petpt_fib[0].run(pt_inputs, {})
    assert res[0] == np.float32(0.029983712)

    res = petasce_fib[0].run(asce_inputs, asce_covers)
    assert res[0] == np.float32(0.00012496980836348878)
