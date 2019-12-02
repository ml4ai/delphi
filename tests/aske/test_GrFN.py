import importlib
import pytest
import json
import sys

import numpy as np

from delphi.GrFN.networks import GroundedFunctionNetwork
import delphi.GrFN.linking as linking
import delphi.translators.GrFN2WiringDiagram.translate as GrFN2WD
import delphi.translators.for2py.f2grfn as f2grfn

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


@pytest.fixture
def sir_simple_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/SIR-simple.f")


@pytest.fixture
def sir_gillespie_inline_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/SIR-Gillespie-SD_inline.f")


@pytest.fixture
def sir_gillespie_ms_grfn():
    return GroundedFunctionNetwork.from_fortran_file("tests/data/program_analysis/SIR-Gillespie-MS.f")


def test_petpt_creation_and_execution(petpt_grfn):
    A = petpt_grfn.to_agraph()
    A.draw("PETPT--GrFN.pdf", prog="dot")
    CAG = petpt_grfn.to_CAG_agraph()
    CAG.draw('PETPT--CAG.pdf', prog='dot')
    assert isinstance(petpt_grfn, GroundedFunctionNetwork)
    assert len(petpt_grfn.inputs) == 5
    assert len(petpt_grfn.outputs) == 1

    values = {name: 1.0 for name in petpt_grfn.inputs}
    res = petpt_grfn.run(values)
    assert res[0] == np.float32(0.029983712)


def test_petasce_creation(petasce_grfn):
    A = petasce_grfn.to_agraph()
    CAG = petasce_grfn.to_CAG_agraph()
    CG = petasce_grfn.to_call_agraph()
    A.draw('PETASCE--GrFN.pdf', prog='dot')
    CAG.draw('PETASCE--CAG.pdf', prog='dot')

    values = {
        "PETASCE_simple::@global::petasce::0::doy::-1": 20.0,
        "PETASCE_simple::@global::petasce::0::meevp::-1": "A",
        "PETASCE_simple::@global::petasce::0::msalb::-1": 0.5,
        "PETASCE_simple::@global::petasce::0::srad::-1": 15.0,
        "PETASCE_simple::@global::petasce::0::tmax::-1": 10.0,
        "PETASCE_simple::@global::petasce::0::tmin::-1": -10.0,
        "PETASCE_simple::@global::petasce::0::xhlai::-1": 10.0,
        "PETASCE_simple::@global::petasce::0::tdew::-1": 20.0,
        "PETASCE_simple::@global::petasce::0::windht::-1": 5.0,
        "PETASCE_simple::@global::petasce::0::windrun::-1": 450.0,
        "PETASCE_simple::@global::petasce::0::xlat::-1": 45.0,
        "PETASCE_simple::@global::petasce::0::xelev::-1": 3000.0,
        "PETASCE_simple::@global::petasce::0::canht::-1": 2.0,
    }

    res = petasce_grfn.run(values)
    assert res == np.float32(0.00012496980836348878)


def test_crop_yield_creation(crop_yield_grfn):
    assert isinstance(crop_yield_grfn, GroundedFunctionNetwork)
    G = crop_yield_grfn.to_agraph()
    G.draw('crop_yield--GrFN.pdf', prog='dot')
    CAG = crop_yield_grfn.to_CAG_agraph()
    CAG.draw('crop_yield--CAG.pdf', prog='dot')


def test_sir_simple_creation(sir_simple_grfn):
    assert isinstance(sir_simple_grfn, GroundedFunctionNetwork)
    G = sir_simple_grfn.to_agraph()
    G.draw('SIR-simple--GrFN.pdf', prog='dot')
    CAG = sir_simple_grfn.to_CAG_agraph()
    CAG.draw('SIR-simple--CAG.pdf', prog='dot')
    # This importlib look up the lambdas file. Thus, the program must
    # maintain the files up to this level before clean up.
    lambdas = importlib.__import__(f"SIR-simple_lambdas")
    (D, I, S, F) = GrFN2WD.to_wiring_diagram(sir_simple_grfn, lambdas)
    assert len(D) == 3
    assert len(I) == 3
    assert len(S) == 9
    assert len(F) == 5


def test_sir_gillespie_inline_creation(sir_gillespie_inline_grfn):
    assert isinstance(sir_gillespie_inline_grfn, GroundedFunctionNetwork)
    G = sir_gillespie_inline_grfn.to_agraph()
    G.draw('SIR-Gillespie_inline--GrFN.pdf', prog='dot')
    CAG = sir_gillespie_inline_grfn.to_CAG_agraph()
    CAG.draw('SIR-Gillespie_inline--CAG.pdf', prog='dot')


def test_sir_gillespie_ms_creation(sir_gillespie_ms_grfn):
    assert isinstance(sir_gillespie_ms_grfn, GroundedFunctionNetwork)
    G = sir_gillespie_ms_grfn.to_agraph()
    G.draw('SIR-Gillespie_ms--GrFN.pdf', prog='dot')
    CAG = sir_gillespie_ms_grfn.to_CAG_agraph()
    CAG.draw('SIR-Gillespie_ms--CAG.pdf', prog='dot')


def test_linking_graph():
    grfn = json.load(open("tests/data/program_analysis/SIR-simple_with_groundings.json", "r"))
    tables = linking.make_link_tables(grfn)
    linking.print_table_data(tables)
    assert len(tables.keys()) == 11


@pytest.mark.skip("Need to update to latest JSON")
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
