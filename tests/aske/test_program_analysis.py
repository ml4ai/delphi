import os
import re
import json
from datetime import date
import pytest

from delphi.translators.for2py import genPGM, f2grfn

from pathlib import Path
from typing import Dict


DATA_DIR = "tests/data/program_analysis"
TEMP_DIR = "."


def cleanup(filepath):
    program_name = filepath.stem
    if os.path.exists(program_name+"_lambdas.py"):
        os.remove(program_name+"_lambdas.py")
    os.remove(f"{TEMP_DIR}/rectified_{program_name}.xml")
    os.remove(f"{TEMP_DIR}/{program_name}_variable_map.pkl")
    os.remove(f"{TEMP_DIR}/{program_name}_modFileLog.json")

def get_python_source(original_fortran_file):
    # Setting a root directory to absolute path of /tests directory.
    root_dir = os.path.abspath(".")
    result_tuple =  f2grfn.fortran_to_grfn(
        original_fortran_file,
        temp_dir=TEMP_DIR,
        root_dir_path=root_dir,
        processing_modules=False,
        module_file_name = f"{TEMP_DIR}/{Path(original_fortran_file).stem}_modFileLog.json"
    )

    return result_tuple


def make_grfn_dict(original_fortran_file) -> Dict:
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).py")
    lambda_file_suffix = "_lambdas.py"

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
        python_file_path_wo_extension = filename_regex.match(
            python_file_path
        )
        lambdas_file_path = (
            python_file_path_wo_extension["filename"] + lambda_file_suffix
        )
        _dict = f2grfn.generate_grfn(
            pySrc[0][0],
            python_file_path,
            lambdas_file_path,
            mode_mapper_dict[0],
            str(original_fortran_file),
            module_log_file_path,
            processing_modules,
        )

        # This blocks system.json to be fully populated.
        # Since the purpose of test_program_analysis is to compare
        # the output GrFN JSON of the main program, I will leave this
        # return as it is to return the only one translated GrFN string.
        return _dict, lambdas_file_path


#########################################################
#                                                       #
#               TARGET FORTRAN TEST FILE                #
#                                                       #
#########################################################


@pytest.fixture
def crop_yield_python_IR_test():
    filepath = Path(f"{DATA_DIR}/crop_yield.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def PETPT_python_IR_test():
    filepath = Path(f"{DATA_DIR}/PETPT.for")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def io_python_IR_test():
    filepath = Path(f"{DATA_DIR}/io-tests/iotest_05.for")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def array_python_IR_test():
    filepath = Path(f"{DATA_DIR}/arrays/arrays-basic-06.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def do_while_python_IR_test():
    filepath = Path(f"{DATA_DIR}/do-while/do_while_04.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def derived_type_python_IR_test():
    filepath = Path(f"{DATA_DIR}/derived-types/derived-types-04.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def cond_goto_python_IR_test():
    filepath = Path(f"{DATA_DIR}/goto/goto_02.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def uncond_goto_python_IR_test():
    filepath = Path(f"{DATA_DIR}/goto/goto_08.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def diff_level_goto_python_IR_test():
    filepath = Path(f"{DATA_DIR}/goto/goto_09.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def save_python_IR_test():
    filepath =  Path(f"{DATA_DIR}" f"/save/simple_variables/save-02.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def cycle_exit_python_IR_test():
    filepath = Path(f"{DATA_DIR}/cycle/cycle_03.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def module_python_IR_test():
    filepath = Path(f"{DATA_DIR}/modules/test_module_08.f")
    yield get_python_source(filepath)
    cleanup(filepath)


@pytest.fixture
def continuation_lines_python_IR_test():
    filepath =  Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-01.for")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def continuation_lines_f90_python_IR_test():
    filepath =  Path(f"{DATA_DIR}" f"/continuation_line/continuation-lines-02.f90")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def SIR_python_IR_test():
    filepath = Path(f"{DATA_DIR}" f"/SIR-Gillespie-SD_inline.f")
    yield get_python_source(filepath)[ 0 ][0]
    cleanup(filepath)


@pytest.fixture
def array_to_func_python_IR_test():
    filepath =  Path(f"{DATA_DIR}" f"/array_func_loop/array-to-func_06.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def multidimensional_array_test():
    filepath = Path(f"{DATA_DIR}/arrays/arrays-basic-06.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def sir_gillespie_sd_test():
    filepath = Path(f"{DATA_DIR}/SIR-Gillespie-SD.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def sir_gillespie_sd_multi_test():
    filepath = Path(f"{DATA_DIR}/SIR-Gillespie-SD_multi_module.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def strings_test():
    filepath = Path(f"{DATA_DIR}/strings/str06.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def derived_type_grfn_test():
    filepath = Path(f"{DATA_DIR}/derived-types/derived-types-04.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def derived_type_array_grfn_test():
    filepath = Path(f"{DATA_DIR}/derived-types/derived-types-02.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def select_case_python_IR_test():
    filepath = Path(f"{DATA_DIR}/select_case/select02.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def select_case_grfn_test():
    filepath = Path(f"{DATA_DIR}/select_case/select02.f")
    yield make_grfn_dict(filepath)
    cleanup(filepath)


@pytest.fixture
def interface_python_IR_test():
    filepath = Path(f"{DATA_DIR}/interface/interface_01.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def multiple_interface_python_IR_test():
    filepath = Path(f"{DATA_DIR}/interface/interface_03.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


@pytest.fixture
def derived_type_with_default():
    filepath = Path(f"{DATA_DIR}/derived-types/derived-types-07.f")
    yield get_python_source(filepath)[0][0]
    cleanup(filepath)


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
    with open(f"{DATA_DIR}/arrays/arrays-basic-06.py", "r") as f:
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


def test_unconditional_goto_pythonIR_generation(
    diff_level_goto_python_IR_test,
):
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
    with open(
        f"{DATA_DIR}/continuation_line/continuation-lines-01.py", "r"
    ) as f:
        python_src = f.read()
    assert continuation_lines_python_IR_test[0] == python_src


def test_continue_line_f90_pythonIR_generation(
    continuation_lines_f90_python_IR_test,
):
    with open(
        f"{DATA_DIR}/continuation_line/continuation-lines-02.py", "r"
    ) as f:
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


def test_select_case_pythonIR_generation(select_case_python_IR_test):
    with open(f"{DATA_DIR}/select_case/select02.py", "r") as f:
        python_src = f.read()
    assert select_case_python_IR_test[0] == python_src


def test_interface_pythonIR_generation(interface_python_IR_test):
    with open(f"{DATA_DIR}/interface/m_mymod.py", "r") as f:
        python_src = f.read()
    assert interface_python_IR_test[0] == python_src


def test_multiple_interface_pythonIR_generation(
    multiple_interface_python_IR_test,
):
    with open(f"{DATA_DIR}/interface/m_interface03_mod.py", "r") as f:
        python_src = f.read()
    assert multiple_interface_python_IR_test[0] == python_src


def test_derived_type_with_default_pythonIR_generation(
    derived_type_with_default,
):
    with open(f"{DATA_DIR}/derived-types/derived-types-07.py", "r") as f:
        python_src = f.read()
    assert derived_type_with_default[0] == python_src


############################################################################
#                                                                          #
#                               GrFN TEST                                  #
#                                                                          #
############################################################################


def test_multidimensional_array_grfn_generation(multidimensional_array_test):
    with open(f"{DATA_DIR}/arrays/arrays-basic-06_GrFN.json", "r") as f:
        grfn_dict = json.load(f)
    assert multidimensional_array_test[0] == grfn_dict

    with open(f"{DATA_DIR}/arrays/arrays-basic-06_lambdas.py", "r") as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{multidimensional_array_test[1]}", "r") as l:
        generated_lambda_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lambda_functions)


def test_sir_gillespie_sd_multi_grfn_generation(sir_gillespie_sd_multi_test):
    with open(f"{DATA_DIR}/SIR-Gillespie-SD_multi_module_GrFN.json", "r") as f:
        grfn_dict = json.load(f)
    assert sir_gillespie_sd_multi_test[0] == grfn_dict

    with open(
        f"{DATA_DIR}/SIR-Gillespie-SD_multi_module_lambdas.py", "r"
    ) as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{sir_gillespie_sd_multi_test[1]}", "r") as l:
        generated_lambda_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lambda_functions)


def test_derived_type_grfn_generation(derived_type_grfn_test):
    with open(
        f"{DATA_DIR}/derived-types/derived-types-04_GrFN.json", "r"
    ) as f:
        grfn_dict = json.load(f)
    assert derived_type_grfn_test[0] == grfn_dict

    with open(
        f"{DATA_DIR}/derived-types/derived-types-04_lambdas.py", "r"
    ) as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{derived_type_grfn_test[1]}", "r") as l:
        generated_lambda_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lambda_functions)


def test_derived_type_array_grfn_generation(derived_type_array_grfn_test):
    with open(
        f"{DATA_DIR}/derived-types/derived-types-02_GrFN.json", "r"
    ) as f:
        grfn_dict = json.load(f)
    assert derived_type_array_grfn_test[0] == grfn_dict

    with open(
        f"{DATA_DIR}/derived-types/derived-types-02_lambdas.py", "r"
    ) as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{derived_type_array_grfn_test[1]}", "r") as l:
        generated_lambda_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lambda_functions)


def test_select_case_grfn_generation(select_case_grfn_test):
    with open(f"{DATA_DIR}/select_case/select02_GrFN.json", "r") as f:
        grfn_dict = json.load(f)
    assert select_case_grfn_test[0] == grfn_dict

    with open(f"{DATA_DIR}/select_case/select02_lambdas_numpy.py", "r") as f:
        target_lambda_functions = f.read()
    with open(f"{TEMP_DIR}/{select_case_grfn_test[1]}", "r") as l:
        generated_lambda_functions = l.read()
    assert str(target_lambda_functions) == str(generated_lambda_functions)
