import os
import json
import sys
from datetime import date
import importlib
import xml.etree.ElementTree as ET
import subprocess as sp
import ast
import pytest

from delphi.GrFN.networks import GroundedFunctionNetwork
from delphi.visualization import visualize
from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)

from pathlib import Path
from typing import Dict, Tuple


def get_python_source(original_fortran_file) -> Tuple[str, str, str]:
    stem = original_fortran_file.stem
    preprocessed_fortran_file = stem + "_preprocessed.f"
    lambdas_filename = stem + "_lambdas.py"
    json_filename = stem + ".json"

    with open(original_fortran_file, "r") as f:
        inputLines = f.readlines()

    with open(preprocessed_fortran_file, "w") as f:
        f.write(preprocessor.process(inputLines))

    xml_string = sp.run(
        [
            "java",
            "fortran.ofp.FrontEnd",
            "--class",
            "fortran.ofp.XMLPrinter",
            "--verbosity",
            "0",
            preprocessed_fortran_file,
        ],
        stdout=sp.PIPE,
    ).stdout

    trees = [ET.fromstring(xml_string)]
    comments = get_comments.get_comments(preprocessed_fortran_file)
    os.remove(preprocessed_fortran_file)
    xml_to_json_translator = translate.XMLToJSONTranslator()
    outputDict = xml_to_json_translator.analyze(trees, comments)
    pySrc = pyTranslate.create_python_string(outputDict)[0][0]
    return pySrc, lambdas_filename, json_filename


def make_grfn_dict(original_fortran_file) -> Dict:
    pySrc, lambdas_filename, json_filename = get_python_source(original_fortran_file)
    asts = [ast.parse(pySrc)]
    pgm_dict = genPGM.create_pgm_dict(lambdas_filename, asts, json_filename, save_file=False)
    return pgm_dict


@pytest.fixture
def crop_yield_grfn_dict():
    yield make_grfn_dict(Path("tests/data/crop_yield.f"))


@pytest.fixture
def petpt_grfn_dict():
    yield make_grfn_dict(Path("tests/data/PETPT.for"))
    os.remove("PETPT_lambdas.py")


@pytest.fixture
def io_grfn_dict():
    yield make_grfn_dict(Path("tests/data/io-tests/iotest_05.for"))
    os.remove("iotest_05_lambdas.py")


@pytest.fixture
def array_python_IR_test():
    yield get_python_source(Path("tests/data/arrays/arrays-basic-06.f"))[0]


@pytest.fixture
def derived_types_python_IR_test():
    yield get_python_source(
        Path("tests/data/derived-types/derived-types-03.f")
    )[0]


def test_crop_yield_grfn_generation(crop_yield_grfn_dict):
    with open("tests/data/crop_yield.json", "r") as f:
        json_dict = json.load(f)
        json_dict["dateCreated"] = str(date.today())

    assert crop_yield_grfn_dict == json_dict


def test_petpt_grfn_generation(petpt_grfn_dict):
    with open("tests/data/PETPT.json", "r") as f:
        json_dict = json.load(f)
        json_dict["dateCreated"] = str(date.today())
    assert petpt_grfn_dict == json_dict


def test_io_grfn_generation(io_grfn_dict):
    with open("tests/data/io-tests/iotest_05_grfn.json", "r") as f:
        json_dict = json.load(f)
        json_dict["dateCreated"] = str(date.today())
    assert io_grfn_dict == json_dict


def test_array_pythonIR_generation(array_python_IR_test):
    with open("tests/data/arrays-basic-06.py", "r") as f:
        python_dict = f.read()
    assert array_python_IR_test == python_dict


def test_derived_type_pythonIR_generation(derived_types_python_IR_test):
    with open("tests/data/derived-types-03.py", "r") as f:
        python_dict = f.read()
    assert derived_types_python_IR_test == python_dict


def test_ProgramAnalysisGraph_from_GrFN():
    sys.path.insert(0, "tests/data/GrFN/")
    lambdas = importlib.__import__("PETPT_torch_lambdas")
    pgm = json.load(open("tests/data/GrFN/PETPT_numpy.json", "r"))
    G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
    PAG = G.to_ProgramAnalysisGraph()
    visualize(PAG, save=True, filename="petpt_pag.pdf")
