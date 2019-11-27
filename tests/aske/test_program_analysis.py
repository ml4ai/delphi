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
TEMP_DIR = "."


def get_python_source(
    original_fortran_file
):
    # Setting a root directory to absolute path of /tests directory.
    root_dir = os.path.abspath(".")
    return f2grfn.fortran_to_grfn(
                original_fortran_file, 
                tester_call=True,
                network_test=False, 
                temp_dir=TEMP_DIR,
                root_dir_path=root_dir,
                processing_modules=False,
                save_file=False
           )


def make_grfn_dict(original_fortran_file) -> Dict:
    lambda_file_suffix = "_lambdas.py"
    tester_call = True
    save_file = False
    network_test = False

    (
        pySrc, 
        json_filename, 
        python_file_paths,
        base, 
        mode_mapper_dict, 
        original_fortran, 
        module_log_file_path,
        processing_modules,
    ) = get_python_source(original_fortran_file)

    for python_file_path in python_file_paths:
        python_file_path_wo_extension = python_file_path[0:-3]
        lambdas_file_path  = python_file_path_wo_extension + lambda_file_suffix
        _dict, generated_files = f2grfn.generate_grfn(
                                        pySrc[0][0],
                                        python_file_path,
                                        lambdas_file_path,
                                        mode_mapper_dict[0],
                                        str(original_fortran_file),
                                        tester_call,
                                        network_test,
                                        module_log_file_path,
                                        processing_modules,
                                        save_file
                                 )
       
        # This blocks system.json to be fully populated.
        # Since the purpose of test_program_analysis is to compare
        # the output GrFN JSON of the main program, I will leave this
        # return as it is to return the only one translated GrFN string.
        return (
                json.dumps(_dict, sort_keys=True, indent=2),
                lambdas_file_path,
                generated_files
        )



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
        Path(f"{DATA_DIR}/derived-types/derived-types-04.f"))[0][0]


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
        Path(f"{DATA_DIR}" f"/modules/test_module_08.f"))


@pytest.fixture
def continuation_lines_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-01.for"))[0][0]


@pytest.fixture
def continuation_lines_f90_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-02.f90"))[0][0]


@pytest.fixture
def SIR_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/SIR-Gillespie-SD_inline.f"))[0][0]

    
@pytest.fixture
def array_to_func_python_IR_test():
    yield get_python_source(
        Path(f"{DATA_DIR}" f"/array_func_loop/array-to-func_06.f"))[0][0]

@pytest.fixture
def multidimensional_array_test():
    yield make_grfn_dict(Path(f"{DATA_DIR}/arrays/arrays-basic-06.f"))

@pytest.fixture
def sir_gillespie_sd_test():
    yield make_grfn_dict(Path(f"{DATA_DIR}/SIR-Gillespie-SD_multi_module.f"))

@pytest.fixture
def strings_test():
    yield get_python_source(Path(f"{DATA_DIR}/strings/str06.f"))[0][0]


@pytest.fixture
def derived_type_grfn_test():
    yield make_grfn_dict(Path(f"{DATA_DIR}/derived-types/derived-types-04.f"))


#########################################################
#                                                       #
#                   PYTHON IR TEST                      #
#                                                       #
#########################################################


def test_crop_yield_pythonIR_generation(crop_yield_python_IR_test):
    with open(f"{DATA_DIR}/crop_yield.py", "r") as f:
        python_src = f.read()
    assert crop_yield_python_IR_test[0] == python_src


def test_PETPT_pythonIR_generation(PETPT_python_IR_test):
    with open(f"{DATA_DIR}/PETPT.py", "r") as f:
        python_src = f.read()
    assert PETPT_python_IR_test[0] == python_src


def test_io_test_pythonIR_generation(io_python_IR_test):
    with open(f"{DATA_DIR}/io-tests/iotest_05.py", "r") as f:
        python_src = f.read()
    assert io_python_IR_test[0] == python_src


def test_array_pythonIR_generation(array_python_IR_test):
    with open(f"{DATA_DIR}/arrays-basic-06.py", "r") as f:
        python_src = f.read()
    assert array_python_IR_test[0] == python_src


def test_do_while_pythonIR_generation(do_while_python_IR_test):
    with open(f"{DATA_DIR}/do-while/do_while_04.py", "r") as f:
        python_src = f.read()
    assert do_while_python_IR_test[0] == python_src


def test_derived_type_pythonIR_generation(derived_type_python_IR_test):
    with open(f"{DATA_DIR}/derived-types-04.py", "r") as f:
        python_src = f.read()
    assert derived_type_python_IR_test[0] == python_src


def test_conditional_goto_pythonIR_generation(cond_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_02.py", "r") as f:
        python_src = f.read()
    assert cond_goto_python_IR_test[0] == python_src


def test_unconditional_goto_pythonIR_generation(uncond_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_08.py", "r") as f:
        python_src = f.read()
    assert uncond_goto_python_IR_test[0] == python_src


def test_unconditional_goto_pythonIR_generation(diff_level_goto_python_IR_test):
    with open(f"{DATA_DIR}/goto/goto_09.py", "r") as f:
        python_src = f.read()
    assert diff_level_goto_python_IR_test[0] == python_src


def test_save_pythonIR_generation(save_python_IR_test):
    with open(f"{DATA_DIR}/save/simple_variables/save-02.py", "r") as f:
        python_src = f.read()
    assert save_python_IR_test[0] == python_src


def test_module_pythonIR_generation(module_python_IR_test):
    src = module_python_IR_test[0]
    with open(f"{DATA_DIR}/modules/test_module_08.py", "r") as f:
        python_src = f.read()
    assert src[1][0] == python_src

    with open(f"{DATA_DIR}/modules/m_mymod8.py", "r") as f:
        python_src = f.read()
    assert src[0][0] == python_src


def test_cycle_exit_pythonIR_generation(cycle_exit_python_IR_test):
    with open(f"{DATA_DIR}/cycle/cycle_03.py", "r") as f:
        python_src = f.read()
    assert cycle_exit_python_IR_test[0] == python_src


def test_continue_line_pythonIR_generation(continuation_lines_python_IR_test):
    with open(f"{DATA_DIR}/continuation_line/continuation-lines-01.py", "r") as f:
        python_src = f.read()
    assert continuation_lines_python_IR_test[0] == python_src


def test_continue_line_f90_pythonIR_generation(
        continuation_lines_f90_python_IR_test
):
    with open(f"{DATA_DIR}/continuation_line/continuation-lines-02.py", "r") as f:
        python_src = f.read()
    assert continuation_lines_f90_python_IR_test[0] == python_src


def test_SIR_pythonIR_generation(SIR_python_IR_test):
    with open(f"{DATA_DIR}/SIR-Gillespie-SD_inline.py", "r") as f:
        python_src = f.read()
    assert SIR_python_IR_test[0] == python_src


def test_array_to_func_pythonIR_generation(array_to_func_python_IR_test):
    with open(f"{DATA_DIR}/array_func_loop/array-to-func_06.py", "r") as f:
        python_src = f.read()
    assert array_to_func_python_IR_test[0] == python_src


def test_strings_pythonIR_generation(strings_test):
    with open(f"{DATA_DIR}/strings/str06.py", "r") as f:
        python_src = f.read()
    assert strings_test[0] == python_src


############################################################################
#                                                                          #
#                               GrFN TEST                                  #
#                                                                          #
############################################################################

def test_multidimensional_array_grfn_generation(multidimensional_array_test):
    with open(f"{DATA_DIR}/arrays/arrays-basic-06_GrFN.json", "r") as f:
        grfn_dict = f.read()
    assert str(multidimensional_array_test[0]) == grfn_dict

    with open(f"{DATA_DIR}/arrays/arrays-basic-06_lambdas.py", "r") as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{multidimensional_array_test[1]}", "r") as l:
        generated_lamdba_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lamdba_functions)

    f2grfn.cleanup_files(multidimensional_array_test[2])


def test_sir_gillespie_sd_grfn_generation(sir_gillespie_sd_test):
    with open(f"{DATA_DIR}/SIR-Gillespie-SD_multi_module_GrFN.json", "r") as f:
        grfn_dict = f.read()
    assert str(sir_gillespie_sd_test[0]) == grfn_dict

    with open(f"{DATA_DIR}/SIR-Gillespie-SD_multi_module_lambdas.py", "r") as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{sir_gillespie_sd_test[1]}", "r") as l:
        generated_lamdba_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lamdba_functions)

    f2grfn.cleanup_files(sir_gillespie_sd_test[2])


def test_derived_type_grfn_generation(derived_type_grfn_test):
    with open(f"{DATA_DIR}/derived-types-04_GrFN.json", "r") as f:
        grfn_dict = f.read()
    assert str(derived_type_grfn_test[0]) == grfn_dict

    with open(f"{DATA_DIR}/derived-types-04_lambdas.py", "r") as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{derived_type_grfn_test[1]}", "r") as l:
        generated_lamdba_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lamdba_functions)

    f2grfn.cleanup_files(derived_type_grfn_test[2])
