import inspect
import importlib
import json
import sys

from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.GrFN.sensitivity import sobol_analysis, FAST_analysis, RBD_FAST_analysis
import delphi.GrFN.analysis as analysis


sys.path.insert(0, "tests/data/GrFN/")


def test_regular_PETPT():
    filepath = "tests/data/GrFN/PETPT.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)

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
    Si = sobol_analysis(G, Ns, problem)
    assert len(Si.keys()) == 6
    assert len(Si["S1"]) == len(args)


def test_PETPT_with_torch():
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
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
    # Regular model
    filepath = "tests/data/GrFN/PETASCE_simple.for"
    G = GroundedFunctionNetwork.from_fortran_file(filepath)

    # Torch model
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

    args = G.model_inputs
    problem = {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }

    N = 1000
    Si = sobol_analysis(G, 100, problem, var_types=type_info)
    tSi = sobol_analysis(tG, N, problem, var_types=type_info, use_torch=True)

    assert len(Si["S1"]) == len(G.model_inputs)
    assert len(Si["S2"][0]) == len(G.model_inputs)

    assert len(tSi["S1"]) == len(tG.model_inputs)
    assert len(tSi["S2"][0]) == len(tG.model_inputs)


def test_PETPT_sensitivity_surface():
    # filepath = "tests/data/GrFN/PETPT.for"
    # G = GroundedFunctionNetwork.from_fortran_file(filepath)
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

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

    args = G.model_inputs
    Si = sobol_analysis(G, 1000, {
        'num_vars': len(args),
        'names': args,
        'bounds': [bounds[arg] for arg in args]
    }, use_torch=True)
    S2 = Si["S2"]
    (s2_max, v1, v2) = analysis.get_max_s2_sensitivity(S2)

    x_var = args[v1]
    y_var = args[v2]
    search_space = [(x_var, bounds[x_var]), (y_var, bounds[y_var])]
    preset_vals = {
        arg: presets[arg] for i, arg in enumerate(args) if i != v1 and i != v2
    }

    (X, Y, Z) = analysis.S2_surface(G, (1000, 1000), search_space, preset_vals, use_torch=True)
    print(Z)

    assert X.shape == (1000,)
    assert Y.shape == (1000,)
    assert Z.shape == (1000, 1000)


test_PETASCE_sobol_analysis()
