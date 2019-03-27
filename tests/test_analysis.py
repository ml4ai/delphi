import pytest
import inspect
import importlib
import json
import sys

from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.sensitivity import sobol_analysis, FAST_analysis, RBD_FAST_analysis
import delphi.GrFN.analysis as analysis
from test_GrFN import PETPT_GrFN, PETASCE_GrFN


sys.path.insert(0, "tests/data/program_analysis")


def test_regular_PETPT():

    args = PETPT_GrFN.model_inputs
    bounds = {
        "petpt::msalb_0": [0, 1],
        "petpt::srad_0": [1, 20],
        "petpt::tmax_0": [-30, 60],
        "petpt::tmin_0": [-30, 60],
        "petpt::xhlai_0": [0, 20],
    }

    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    Ns = 1000 # TODO: Khan, experiment with this value
    Si = sobol_analysis(PETPT_GrFN, Ns, problem)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)


@pytest.mark.skip("Need to update to latest JSON")
def test_PETPT_with_torch():
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/program_analysis/PETPT.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

    args = G.model_inputs
    bounds = {
        "petpt::msalb_0": [0, 1],
        "petpt::srad_0": [1, 20],
        "petpt::tmax_0": [-30, 60],
        "petpt::tmin_0": [-30, 60],
        "petpt::xhlai_0": [0, 20],
    }

    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    Ns = 1000                      # TODO: Khan, experiment with this value
    Si = sobol_analysis(G, Ns, problem, use_torch=True)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)

    Si = FAST_analysis(G, Ns, problem)
    assert len(Si.keys()) == 3
    assert len(Si["S1"]) == len(args)

    Si = RBD_FAST_analysis(G, Ns, problem)
    assert len(Si.keys()) == 2
    assert len(Si["S1"]) == len(args)


def test_PETASCE_sobol_analysis():
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

    args = PETASCE_GrFN.model_inputs
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    N = 1000
    Si = sobol_analysis(PETASCE_GrFN, 100, problem, var_types=type_info)
    tSi = sobol_analysis(tG, N, problem, var_types=type_info, use_torch=True)

    assert len(Si["S1"]) == len(PETASCE_GrFN.model_inputs)
    assert len(Si["S2"][0]) == len(PETASCE_GrFN.model_inputs)

    assert len(tSi["S1"]) == len(tG.model_inputs)
    assert len(tSi["S2"][0]) == len(tG.model_inputs)


def test_PETPT_sensitivity_surface():
    bounds = {
        "petpt::msalb_0": (0, 1),
        "petpt::srad_0": (1, 20),
        "petpt::tmax_0": (-30, 60),
        "petpt::tmin_0": (-30, 60),
        "petpt::xhlai_0": (0, 20),
    }
    presets = {
        "petpt::msalb_0": 0.5,
        "petpt::srad_0": 10,
        "petpt::tmax_0": 20,
        "petpt::tmin_0": 10,
        "petpt::xhlai_0": 10,
    }

    args = PETPT_GrFN.model_inputs
    Si = sobol_analysis(PETPT_GrFN, 1000, {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    })
    S2 = Si["S2"]
    (s2_max, v1, v2) = analysis.get_max_s2_sensitivity(S2)

    x_var = args[v1]
    y_var = args[v2]
    search_space = [(x_var, bounds[x_var]), (y_var, bounds[y_var])]
    preset_vals = {
        arg: presets[arg] for i, arg in enumerate(args) if i != v1 and i != v2
    }

    (X, Y, Z) = analysis.S2_surface(PETPT_GrFN, (80, 60), search_space, preset_vals)
    print(Z)

    assert X.shape == (80,)
    assert Y.shape == (60,)
    assert Z.shape == (80, 60)


def test_FIB_creation():
    # filepath = "tests/data/GrFN/PETPT.for"
    # PETPT_GrFN = GroundedFunctionNetwork.from_fortran_file(filepath)

    # A = PETPT_GrFN.to_agraph()
    # A.draw("PETPT_GrFN.pdf", prog="dot")
    # A = PETPT_GrFN.to_call_agraph()
    # A.draw("PETPT_CG.pdf", prog="dot")
    A = PETPT_GrFN.to_CAG_agraph()
    A.draw("PETPT_CAG.pdf", prog="dot")

    # filepath = "tests/data/GrFN/PETASCE_simple.for"
    # PETASCE_GrFN = GroundedFunctionNetwork.from_fortran_file(filepath)

    # A = PETASCE_GrFN.to_agraph()
    # A.draw("PETASCE_GrFN.pdf", prog="dot")
    # A = PETASCE_GrFN.to_call_agraph()
    # A.draw("PETASCE_CG.pdf", prog="dot")
    A = PETASCE_GrFN.to_CAG_agraph()
    A.draw("PETASCE_CAG.pdf", prog="dot")

    # PETPT_FIB = PETPT_GrFN.to_FIB(PETASCE_GrFN)
    PETASCE_FIB = PETASCE_GrFN.to_FIB(PETPT_GrFN)
    # A1 = PETPT_FIB.to_agraph()
    # A1.draw("PETPT_FIB.pdf", prog="dot")
    A2 = PETASCE_FIB.to_agraph()
    A2.draw("PETASCE_FIB.pdf", prog="dot")
