import os
import json
from datetime import date
import xml.etree.ElementTree as ET
import subprocess as sp
import ast
import pytest

from delphi.translators.for2py import genPGM, f2grfn

from pathlib import Path
from typing import Dict, Tuple


DATA_DIR = "tests/data/program_analysis"


def get_python_source(
    original_fortran_file
) -> Tuple[str, str, str, str, Dict]:
    return f2grfn.fortran_to_grfn(original_fortran_file, True, False, ".")


def make_grfn_dict(original_fortran_file) -> Dict:
    pySrc, lambdas_filename, json_filename, python_filename, mode_mapper_dict = get_python_source(
        original_fortran_file
    )
    _dict = f2grfn.generate_grfn(
        pySrc,
        python_filename,
        lambdas_filename,
        json_filename,
        mode_mapper_dict,
        True
    )

    return _dict


def postprocess_test_data_grfn_dict(_dict):
    """ Postprocess the test data grfn dict to change the date to the date of
    execution, and also remove the randomly generated gensyms """
    _dict["dateCreated"] = "".join(str(date.today()).split("-"))
    for identifier in _dict["identifiers"]:
        if "gensyms" in identifier:
            del identifier["gensyms"]

#########################################################
#                                                       #
#               TARGET FORTRAN TEST FILE                #
#                                                       #
#########################################################

@pytest.fixture
def crop_yield_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/crop_yield.f"))[0][0]


@pytest.fixture
def PETPT_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/PETPT.for"))[0][0]


@pytest.fixture
def io_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/io-tests/iotest_05.for"))[0][0]


@pytest.fixture
def array_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/arrays/arrays-basic-06.f"))[0][0]


@pytest.fixture
def do_while_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/do-while/do_while_04.f"))[0][0]


@pytest.fixture
def derived_type_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}/derived-types/derived-types-04.f")
    )[0][0]


@pytest.fixture
def cond_goto_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/goto/goto_02.f"))[0][0]


@pytest.fixture
def uncond_goto_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/goto/goto_08.f"))[0][0]


@pytest.fixture
def diff_level_goto_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/goto/goto_09.f"))[0][0]


@pytest.fixture
def save_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/save/simple_variables/save-02.f"))[0][0]


@pytest.fixture
def cycle_exit_python_IR_test():
    yield get_python_source(Path(f"{DATA_DIR}/cycle/cycle_03.f"))[0][0]


@pytest.fixture
def module_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/modules/test_module_08.f"))[0]


@pytest.fixture
def continuation_lines_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-01.for"))[0][0]


@pytest.fixture
def continuation_lines_f90_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-02.f90"))[0][0]

    
#########################################################
#                                                       #
#               TARGET PYTHON TEST FILE                 #
#                                                       #
#########################################################


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


def test_derived_type_pythonIR_generation(derived_type_python_IR_test):
    with open(f"{DATA_DIR}/derived-types-04.py", "r") as f:
        python_src = f.read()
    assert derived_type_python_IR_test == python_src


def test_conditional_goto_pythonIR_generation(cond_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_02.py", "r") as f:
        python_src = f.read()
    assert cond_goto_python_IR_test == python_src


def test_unconditional_goto_pythonIR_generation(uncond_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_08.py", "r") as f:
        python_src = f.read()
    assert uncond_goto_python_IR_test == python_src


def test_unconditional_goto_pythonIR_generation(diff_level_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_09.py", "r") as f:
        python_src = f.read()
    assert diff_level_goto_python_IR_test == python_src


def test_save_pythonIR_generation(save_python_IR_test):
    with open(f"{DATA_DIR}/save/simple_variables/save-02.py", "r") as f:
        python_src = f.read()
    assert save_python_IR_test == python_src


def test_module_pythonIR_generation(module_python_IR_test):
    src = module_python_IR_test
    with open(f"{DATA_DIR}/modules/test_module_08.py", "r") as f:
        python_src = f.read()
    assert src[1] == python_src

    with open(f"{DATA_DIR}/modules/m_mymod8.py", "r") as f:
        python_src = f.read()
    assert src[0] == python_src


def test_cycle_exit_pythonIR_generation(cycle_exit_python_IR_test):
    with open(f"{DATA_DIR}/cycle/cycle_03.py", "r") as f:
        python_src = f.read()
    assert cycle_exit_python_IR_test == python_src


def test_continue_line_pythonIR_generation(continuation_lines_python_IR_test):
    with open(f"{DATA_DIR}/continuation_line/continuation-lines-01.py", "r") as f:
        python_src = f.read()
    assert continuation_lines_python_IR_test == python_src


def test_continue_line_f90_pythonIR_generation(
        continuation_lines_f90_python_IR_test
):
    with open(f"{DATA_DIR}/continuation_line/continuation-lines-02.py", "r") as f:
        python_src = f.read()
    assert continuation_lines_f90_python_IR_test == python_src
