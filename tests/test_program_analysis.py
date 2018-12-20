import os
import json
from datetime import date
from delphi.autoTranslate.scripts import (
    f2py_pp,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
)
from delphi.GrFN.scopes import Scope
from delphi.GrFN.ProgramAnalysisGraph import ProgramAnalysisGraph
import xml.etree.ElementTree as ET
import subprocess as sp
import ast
import pytest
from delphi.visualization import visualize
from pathlib import Path
from importlib import import_module
from typing import Dict


def make_grfn_dict(original_fortran_file) -> Dict:
    stem = original_fortran_file.stem
    preprocessed_fortran_file = stem + "_preprocessed.f"
    lambdas_filename = stem + "_lambdas.py"
    json_filename = stem + ".json"

    with open(original_fortran_file, "r") as f:
        inputLines = f.readlines()

    with open(preprocessed_fortran_file, "w") as f:
        f.write(f2py_pp.process(inputLines))

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
    pySrc = pyTranslate.create_python_string(outputDict)
    asts = [ast.parse(pySrc)]
    pgm_dict = genPGM.create_pgm_dict(lambdas_filename, asts, json_filename)
    return pgm_dict


@pytest.fixture
def crop_yield_grfn_dict():
    yield make_grfn_dict(Path("tests/data/crop_yield.f"))


@pytest.fixture
def petpt_grfn_dict():
    yield make_grfn_dict(Path("tests/data/PETPT.for"))
    os.remove("PETPT_lambdas.py")


def test_crop_yield_grfn_generation(crop_yield_grfn_dict):
    with open("tests/data/crop_yield_grfn.json", "r") as f:
        json_dict = json.load(f)
        json_dict["dateCreated"] = str(date.today())

    assert sorted(crop_yield_grfn_dict) == sorted(json_dict)


def test_petpt_grfn_generation(petpt_grfn_dict):
    with open("tests/data/PETPT_grfn.json", "r") as f:
        json_dict = json.load(f)
        json_dict["dateCreated"] = str(date.today())
    assert sorted(petpt_grfn_dict) == sorted(json_dict)


def test_ProgramAnalysisGraph_crop_yield(crop_yield_grfn_dict):
    import crop_yield_lambdas

    A = Scope.from_dict(crop_yield_grfn_dict).to_agraph()
    G = ProgramAnalysisGraph.from_agraph(A, crop_yield_lambdas)
    G.initialize()
    visualize(G)
    os.remove("crop_yield_lambdas.py")
