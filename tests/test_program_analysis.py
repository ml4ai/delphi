import os
import json
from datetime import date
import xml.etree.ElementTree as ET
import subprocess as sp
import ast
import pytest

from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    mod_index_generator,
    rectify,
)

from pathlib import Path
from typing import Dict, Tuple


DATA_DIR = "tests/data/program_analysis"

def get_python_source(original_fortran_file) -> Tuple[str, str, str, str, Dict]:
    stem = original_fortran_file.stem
    preprocessed_fortran_file = stem + "_preprocessed.f"
    lambdas_filename = stem + "_lambdas.py"
    json_filename = stem + ".json"
    python_filename = stem + ".py"

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

    tree = rectify.buildNewASTfromXMLString(xml_string)
    trees = [tree]
    comments = get_comments.get_comments(preprocessed_fortran_file)
    os.remove(preprocessed_fortran_file)
    xml_to_json_translator = translate.XMLToJSONTranslator()
    mode_mapper_tree = tree
    generator = mod_index_generator.moduleGenerator()
    mode_mapper_dict = generator.analyze(mode_mapper_tree)
    outputDict = xml_to_json_translator.analyze(trees, comments)
    pySrc = pyTranslate.create_python_source_list(outputDict)[0][0]
    return pySrc, lambdas_filename, json_filename, python_filename, mode_mapper_dict


def make_grfn_dict(original_fortran_file) -> Dict:
    pySrc, lambdas_filename, json_filename, python_filename, mode_mapper_dict = get_python_source(original_fortran_file)
    asts = [ast.parse(pySrc)]
    _dict = genPGM.create_pgm_dict(lambdas_filename, asts, python_filename, mode_mapper_dict, save_file=False)
    for identifier in _dict["identifiers"]:
        del identifier["gensyms"]

    os.remove(lambdas_filename)
    return _dict


def postprocess_test_data_grfn_dict(_dict):
    """ Postprocess the test data grfn dict to change the date to the date of
    execution, and also remove the randomly generated gensyms """
    _dict["dateCreated"] = "".join(str(date.today()).split("-"))
    for identifier in _dict["identifiers"]:
        if "gensyms" in identifier:
            del identifier["gensyms"]

@pytest.fixture
def crop_yield_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/crop_yield.f"))[0]

@pytest.fixture
def PETPT_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/PETPT.for"))[0]

@pytest.fixture
def io_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/io-tests/iotest_05.for"))[0]

@pytest.fixture
def array_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/arrays/arrays-basic-06.f"))[0]

@pytest.fixture
def do_while_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/do-while/do_while_04.f"))[0]

def test_crop_yield_pythonIR_generation(crop_yield_python_IR_test):
    with open(f"{DATA_DIR}/crop_yield.py", "r") as f:
        python_src = f.read()
    assert crop_yield_python_IR_test == python_src

def test_PETPT_pythonIR_generation(PETPT_python_IR_test):
    with open(f"{DATA_DIR}/PETPT.py", "r") as f:
        python_src = f.read()
    assert PETPT_python_IR_test == python_src

def test_io_test_pythonIR_generation(io_python_IR_test):
    with open(f"{DATA_DIR}/io-tests/iotest_05.py", "r") as f:
        python_src = f.read()
    assert io_python_IR_test == python_src

def test_array_pythonIR_generation(array_python_IR_test):
    with open(f"{DATA_DIR}/arrays-basic-06.py", "r") as f:
        python_src = f.read()
    assert array_python_IR_test == python_src

def test_do_while_pythonIR_generation(do_while_python_IR_test):
    with open(f"{DATA_DIR}/do-while/do_while_04.py", "r") as f:
        python_src = f.read()
    assert do_while_python_IR_test == python_src
