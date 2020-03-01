"""
This program exists to replacement the autoTranslate bash script.
Instead of creating and using each file for next operation like in the
autoTranslate bash script, it creates Python object and passes it to the
next function. Thus, it works as calling a multiple functions in a
single program. This new f2grfn.py does not invoke main functions in
each program.

In simplicity, it's a single program that integrates the
functionality of test_program_analysis.py and autoTranslate.

Example:
    This script can be executed as below:

        $ python f2grfn -f <fortran_file> -r <root_dir>

fortran_file: An original input file to a program that is to be
    translated to GrFN.
root_dir: A root directory where module log file should be created or
    found.

Author: Terrence J. Lim
"""

import os
import re
import sys
import ast
import json
import argparse
import pickle
import delphi.paths
from pathlib import Path
import subprocess as sp
import xml.etree.ElementTree as ET
from os.path import isfile
from typing import Dict, List, Tuple

from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    mod_index_generator,
    rectify,
    genModFileLog,
)

OFP_JAR_FILES = [
    "antlr-3.3-complete.jar",
    "commons-cli-1.4.jar",
    "OpenFortranParser-0.8.4-3.jar",
    "OpenFortranParserXML-0.4.1.jar",
]
"""OFP_JAR_FILES is a list of JAR files used by the Open Fortran Parser (OFP).
"""

GENERATED_FILE_PATHS = []
"""A list of all the file paths that were generated during f2grfn process.
"""


MODULE_FILE_PREFIX = "m_"
"""Module file prefix that all generated Python module files will be specified
with."""

MODULE_FILE_NAME = "modFileLog.json"
"""A file that holds log of all files with module(s)
"""


def generate_ofp_xml(
    preprocessed_fortran_file, save_intermediate_files=False
) -> str:
    """ This function runs Open Fortran Parser to generate initial AST XML from
    the preprocessed Fortran file.

    Args:
        preprocessed_fortran_file (str): A preprocessed fortran file name.

    Returns:
        str: OFP-generated XML as a string
    """

    # Execute Java command to generate XML string from fortran file
    ofp_xml_string = sp.run(
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

    if save_intermediate_files:
        tree = ET.ElementTree(ET.fromstring(ofp_xml_string))
        tree.write(
            f"{Path(preprocessed_fortran_file).stem}.xml"
        )

    return ofp_xml_string


def generate_rectified_xml(
    ofp_xml_string, original_fortran_file, module_log_file_path,
):
    """This function rectifies XML that was generated by OFP. Then, it will
    generate an output file, but also returns rectified element tree object
    back to the caller.

    Args:
        ofp_xml (str): A string of XML that was generated by OFP.
        rectified_file (str): A file name that rectified XML
        will be written to.
        original_fortran_file (str): Original fortran file path.
        module_log_file_path (str): Path to module log file.

    Returns:
        A two-tuple with the following elements:
        Element Tree (ET) Object: An object of generated rectified XML.
        module_files_to_process: a list of module files to process.
    """

    rectified_xml, module_files_to_process = rectify.buildNewASTfromXMLString(
        ofp_xml_string, original_fortran_file, module_log_file_path
    )

    rectified_tree = ET.ElementTree(rectified_xml)
    rectified_tree.write(
        "rectified_" + str(Path(original_fortran_file).stem) + ".xml"
    )
    return rectified_xml, module_files_to_process


def generate_outputdict(rectified_tree, preprocessed_fortran_file) -> Dict:
    """This function generates a dictionary of ast and generates a dict
    with XML generated by translate.py and comments obtained with
    get_comments.py.

    Args:
        rectified_tree (:obj: 'ET'): An object of rectified XML.
        preprocessed_fortran_file (str): Path to preprocessed fortran file

    Returns:
        dict: A dictionary of XML generated by translate.py
    """

    output_dictionary = translate.xml_to_py([rectified_tree])
    output_dictionary["comments"] = get_comments.get_comments(
        preprocessed_fortran_file
    )

    return output_dictionary


def generate_python_sources(
    output_dictionary,
    python_files,
    main_python_file,
    temp_dir,
) -> List[Tuple]:
    """This function generates Python source file from generated Python source
    list. This function will return this list back to the caller for GrFN
    generation.

    Args:
        output_dictionary (dict): A dictionary of XML generated
        by translate.py.
        python_files: A list of python file names.
        python_file_name (str): A file name where translated python strings
        will be written to.
        temp_dir (str): Temporary directory to store the translated files.

    Returns:
        str: A string of generated Python code.
    """

    (
        python_sources,
        variable_map,
    ) = pyTranslate.get_python_sources_and_variable_map(output_dictionary)

    with open(main_python_file.replace(".py", "_variable_map.pkl"), "wb") as f:
        pickle.dump(variable_map, f)

    for python_src_tuple in python_sources:
        file_path = (
            f"{temp_dir}/{MODULE_FILE_PREFIX}{python_src_tuple[1].lower()}.py"
            if python_src_tuple[2] == "module"
            else main_python_file
        )

        with open(file_path, "w") as f:
            f.write(python_src_tuple[0])

        python_files.append(file_path)

    return python_sources


def generate_grfn(
    python_source_string,
    python_file_path,
    lambdas_file_path,
    mode_mapper_dictionary,
    original_fortran_file,
    mod_log_file_path,
    processing_modules,
) -> Dict:
    """This function generates GrFN dictionary object and file.

    Args:
        python_source_string (str): A string of Python code.
        python_file_path (str): A generated Python file path.
        lambdas_file_path (str): A lambdas file path.
        mode_mapper_dictionary (list): A mapper of file info (i.e. filename,
        module, and exports, etc).
        original_fortran_file (str): The path to the original
        Fortran file being analyzed.
        mod_log_file_path (str): A path to module log file.
        processing_modules (bool): A boolean condition marker to indicate
        whether current GrFN generation is for a module.

    Returns:
        dict: A dictionary of generated GrFN.
    """

    grfn_filepath_list = []

    # Regular expression to identify the path and name of all Python files
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).py")

    # First, find the main Python file in order to populate the module
    # mapper
    file_match = re.match(filename_regex, python_file_path)
    assert file_match, "Invalid filename."

    path = file_match.group("path")
    filename = file_match.group("filename")

    module_file_exists = is_module_file(filename)

    # Ignore all Python files of modules created by `pyTranslate.py`
    # since these module files do not contain a corresponding XML file.
    if module_file_exists:
        file_name = genPGM.get_original_file_name(original_fortran_file)
        xml_file = f"{path}rectified_{file_name}.xml"
    else:
        xml_file = f"{path}rectified_{filename}.xml"

    # Mapping all variables and objects in the various files

    module_mapper = mod_index_generator.get_index(xml_file, mod_log_file_path)
    module_import_paths = {}

    # Build GrFN and lambdas
    asts = [ast.parse(python_source_string)]

    grfn_dict = genPGM.create_grfn_dict(
        lambdas_file_path,
        asts,
        python_file_path,
        module_mapper,
        original_fortran_file,
        mod_log_file_path,
        module_file_exists,
        module_import_paths,
    )

    grfn_file = python_file_path.replace(".py", "_GrFN.json")
    if module_file_exists:
        python_file_path = path + file_name + ".py"
    grfn_filepath_list.append(grfn_file)

    # Cleanup GrFN.
    del grfn_dict["date_created"]
    for item in grfn_dict["variables"]:
        if "gensym" in item:
            del item["gensym"]
    for item in grfn_dict["containers"]:
        if "gensym" in item:
            del item["gensym"]

    # Load logs from the module log file.
    with open(mod_log_file_path) as json_f:
        module_logs = json.load(json_f)

    # Generate systems.json linking file.
    system_def = genPGM.generate_system_def(
        [python_file_path],
        grfn_filepath_list,
        module_import_paths,
        module_logs,
        original_fortran_file,
    )

    grfn_dict["system"] = system_def

    return grfn_dict


def is_module_file(filename):
    """This function is to check whether the handling
    file is a module file or not.
    Args:
        filename (str): Name of a file.
    Returns:
        (bool) True if it is a module file.
        (bool) False, if it is not a module file.
    """
    if filename.startswith(MODULE_FILE_PREFIX):
        return True
    else:
        return False


def check_classpath():
    """check_classpath() checks whether the files in OFP_JAR_FILES can all be
    found in via the environment variable CLASSPATH."""

    not_found = []
    classpath = os.environ["CLASSPATH"].split(":")
    for jar_file in OFP_JAR_FILES:
        found = False
        for path in classpath:
            dir_path = os.path.dirname(path)
            if path.endswith(jar_file) or (
                path.endswith("*") and jar_file in os.listdir(dir_path)
            ):
                found = True
                break
            if not found:
                not_found.append(jar_file)

        if not_found:
            sys.stderr.write("ERROR: JAR files not found via CLASSPATH:\n")
            sys.stderr.write(f" {','.join(not_found)}\n")
            sys.exit(1)


def indent(elem, level=0):
    """ This function indents each level of XML.

    Source: https://stackoverflow.com/questions/3095434/inserting-newlines
    -in-xml-file-generated-via-xml-etree-elementstree-in-python

    Args:
        elem (:obj: 'ET'): An element tree XML object.
        level (int): A root level of XML.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def fortran_to_grfn(
    original_fortran=None,
    temp_dir=None,
    root_dir_path=".",
    module_file_name=MODULE_FILE_NAME,
    processing_modules=False,
    save_intermediate_files=False,
):
    """This function invokes other appropriate functions
    to process and generate objects to translate fortran
    to python IR. This function will either be invoked by
    local main function or the outer tester functions,
    such as test_program_analysis.py or network.py.

    Args:
        original_fortran (str): A file name of original fortran script.
        temp_dir (str): A default temporary directory where output
        files will be stored.
        root_dir_path (str): A root directory of the program.
        module_file_name (str): A module log file name.
        processing_modules (bool): A boolean condition marker to indicate
        whether current fortran_to_grfn execution is for processing a module.

    Returns:
        {
            'python_src': A string of Python code,
            'python_file': A file name of generated python script,
            'lambdas_file': A file name where lambdas will be,
            'mode_mapper_dict': mapper of file info (i.e. filename, module, and exports, etc).
        }
    """
    current_dir = "."
    check_classpath()

    # Else, for2py function gets invoked by the test
    # programs, it will be passed with an argument
    # of original Fortran file path.
    original_fortran_file = original_fortran
    root_dir = root_dir_path
    module_file = module_file_name

    # Generate or update module log file.
    module_log_file_path = genModFileLog.mod_file_log_generator(
        root_dir, module_file
    )

    base = str(Path(original_fortran_file).stem)

    # temp_dir is None means that the output file was
    # not set by the program that calls this function.
    # Thus, generate the output temporary file based
    # on the user input or the default path "tmp".
    temp_out_dir = "tmp"
    if temp_dir is None:
        temp_dir = current_dir + "/" + temp_out_dir

    # If "tmp" directory does not exist already, simply create one.
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    else:
        assert os.access(
            temp_dir, os.W_OK
        ), f"Directory {temp_dir} is not writable.\n\
            Please, provide the directory name to hold files."

    print(f"*** ALL OUTPUT FILES LIVE IN [{temp_dir}]")

    # Output files
    python_file = temp_dir + "/" + base + ".py"

    # TODO Add some code using Pathlib to check the file extension and make
    # sure it's either .f or .for.
    preprocessed_fortran_file = str(original_fortran_file).replace(
        ".f", "_preprocessed.f"
    )
    preprocessor.create_preprocessed_file(
        original_fortran_file, preprocessed_fortran_file
    )

    # Generate OFP XML from preprocessed fortran
    ofp_xml_string = generate_ofp_xml(
        preprocessed_fortran_file,
        save_intermediate_files=save_intermediate_files,
    )

    # Rectify and generate a new XML from OFP XML
    rectified_tree, module_files_to_process = generate_rectified_xml(
        ofp_xml_string, original_fortran_file, module_log_file_path,
    )

    # If a program uses module(s) that does not reside within the same file,
    # we need to find out where they live and process those files first. Thus,
    # the program should know the list of modules that require advance
    # processing that was collected by rectify.py, below code will recursively
    # call fortran_to_grfn function to process module files end-to-end
    # (Fortran-to-GrFN).
    if module_files_to_process:
        processing_modules = True
        for target_module_file in module_files_to_process:
            fortran_to_grfn(
                target_module_file,
                temp_dir,
                root_dir,
                module_file,
                processing_modules,
                save_intermediate_files=save_intermediate_files,
            )
        processing_modules = False

    # Generate separate list of modules file.
    generator = mod_index_generator.ModuleGenerator()
    mode_mapper_dict = generator.analyze(rectified_tree, module_log_file_path)

    # This will update the log file with more information about the module,
    # such as the declared symbols and types that is needed for generating GrFN.
    if processing_modules:
        genModFileLog.update_mod_info_json(
            module_log_file_path, mode_mapper_dict[0]
        )

    output_dict = generate_outputdict(
        rectified_tree, preprocessed_fortran_file
    )

    translated_python_files = []
    # Create a list of tuples with information about the Python source files.
    python_sources = generate_python_sources(
        output_dict,
        translated_python_files,
        python_file,
        temp_dir,
        save_intermediate_files=save_intermediate_files,
    )

    if not save_intermediate_files:
        os.remove(preprocessed_fortran_file)
        for translated_python_file in translated_python_files:
            os.remove(translated_python_file)

    return (
        python_sources,
        translated_python_files,
        mode_mapper_dict,
        original_fortran_file,
        module_log_file_path,
        processing_modules,
    )
