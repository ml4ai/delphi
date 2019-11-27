"""
This program is for replacement of autoTranslate bash script.
Instead of creating and using each file for next operation
like in autoTranslate bash script, it creates python object
and passes it to the next function. Thus, it works as calling
a multiple functions in a single program. This new f2grfn.py
does not invoke main functions in each program.

In simplicity, it's a single program that integrated the
functionality of test_program_analysis.py and autoTranslate.

Example:
    This script can be executed as below:

        $ python f2grfn -f <fortran_file> -r <root_dir>

fortran_file: An original input file to a program that is
to be translated to GrFN.
root_dir: A root directory where module log file should be
created or found.

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
import ntpath as np
import subprocess as sp
import xml.etree.ElementTree as ET
from delphi.translators.for2py.mod_index_generator import get_index
from os.path import isfile

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
"""Module file prefix that all generated python module files will be specified with.
"""

MODULE_FILE_NAME = "modFileLog.json"
"""A file that holds log of all files with module(s)
"""


def generate_ofp_xml(preprocessed_fortran_file, ofp_file, tester_call):
    """ This function executes Java command to run open
    fortran parser to generate initial AST XML from
    the preprocessed fortran file.

    Args:
        preprocessed_fortran_file (str): A preprocessed fortran file name.
        ofp_file (str): A file name that the OFP XML will be written to.
        tester_call (bool): A boolean condition that will indicate whether
        the program was invoked standalone (False) or by tester scripts (True).

    Returns:
        str: OFP generate XML in a sequence of strings.
    """

    if not tester_call:
        print(
            f"+$java fortran.ofp.FrontEnd --class fortran.ofp.XMLPrinter "
            f"--verbosity 0 {preprocessed_fortran_file}"
        )

    # Execute Java command to generate XML
    # string from fortran file
    ofp_xml = sp.run(
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

    if not tester_call:
        # Indent and construct XML for file output
        xml_ast = ET.XML(ofp_xml)
        indent(xml_ast)
        tree = ET.ElementTree(xml_ast)

        try:
            ofp_file_handle = open(ofp_file, "w")
            ofp_file_handle.close()
        except IOError:
            assert False, f"Failed to write to {ofp_file}."

        tree.write(ofp_file)
        log_generated_files([ofp_file])

    return ofp_xml


def generate_rectified_xml(
        ofp_xml,
        rectified_file, 
        tester_call,
        original_fortran_file,
        module_log_file_path
):
    """This function rectifies XMl that was generated by
    OFP. Then, it will generate an output file, but
    also returns rectified element tree object back
    to the caller.

    Args:
        ofp_xml (str): A string of XML that was generated by OFP.
        rectified_file (str): A file name that rectified XML
        will be written to.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).
        original_fortran_file (str): Original fortran file path.
        module_log_file_path (str): Path to module log file.

    Returns:
        Element Tree (ET) Object: An object of generated rectified
        XML.
    """

    if not tester_call:
        print(
            "+Generating rectified XML: Func: <buildNewASTfromXMLString>, "
            "Script: <rectify.py>"
        )

    rectified_xml, module_files_to_process = rectify.buildNewASTfromXMLString(
                                                ofp_xml, original_fortran_file,
                                                module_log_file_path
                                             )

    rectified_tree = ET.ElementTree(rectified_xml)
    try:
        rectified_file_handle = open(rectified_file, 'w')
        rectified_file_handle.close()
    except IOError:
        assert False, f"Failed to write to {rectified_file}."

    rectified_tree.write(rectified_file)
    log_generated_files([rectified_file])

    return rectified_xml, module_files_to_process


def generate_outputdict(
    rectified_tree,
    preprocessed_fortran_file,
    pickle_file,
    tester_call
):
    """This function generates a dictionary of ast and
    generates a pickle file.

    Args:
        rectified_tree (:obj: 'ET'): An object of rectified XML.
        preprocessed_fortran_file (str): A file name of preprocessed
        fortran file
        pickle_file (str): A file name where binary pickle will be
        written to.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).

    Returns:
        dict: A dictionary of XML generated by translate.py
    """

    output_dictionary = translate.xml_to_py(
        [rectified_tree], preprocessed_fortran_file
    )

    if not tester_call:
        print("+Generating pickle file: Func: <xml_to_py>, Script: "
              "<translate.py>")
        try:
            with open(pickle_file, "wb") as f:
                pickle.dump(output_dictionary, f)
        except IOError:
            assert False, f"Failed to write to {pickle_file}."
    return output_dictionary


def generate_python_src(
    output_dictionary,
    python_files,
    main_python_file,
    output_file,
    variable_map_file,
    temp_dir,
    tester_call,
):
    """This function generates python source file from
    generated python source list. This function will
    return this list back to the caller for GrFN
    generation.

    Args:
        output_dictionary (dict): A dictionary of XML generated
        by translate.py.
        python_file_name (str): A file name where translated python strings
        will be written to.
        output_file (str): A file name where list of output file names
        will be written to.
        temp_dir (str): Temporary directory to store the translated files.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).

    Returns:
        str: A string of generated python code.
    """
    
    (python_source, variable_map) = pyTranslate.create_python_source_list(
     output_dictionary)

    with open(variable_map_file, "wb") as f:
        pickle.dump(variable_map, f)

    if not tester_call:
        print(
            "+Generating python source file:\
                Func: <create_python_source_list>,\
                Script: <pyTranslate.py>"
        )

    output_list = []
    for item in python_source:
        if item[2] == "module":
            module_file_generator(item, temp_dir, output_list, python_files)            
        else:
            try:
                with open(main_python_file, "w") as f:
                    output_list.append(main_python_file)
                    f.write(item[0])
                python_files.append(main_python_file)
            except IOError:
                assert False, f"Unable to write to {main_python_file}."

    try:
        with open(output_file, "w") as f:
            for fileName in output_list:
                f.write(fileName + " ")
    except IOError:
        assert False, f"Unable to write to {output_file}."

    for f in python_files:
        if not isfile(f):
            python_files.remove(f)

    return python_source


def module_file_generator(item, temp_dir, output_list, python_files):
    """This function extracts a translated module from
    the python source and generates a new separate python file.

    Args:
        item (list): A list that each element holds a translated
        python source in string.
        temp_dir (str): A path to the temporary directory that will
        hold generated files.
        output_list (list): A list that holds list of output files.
        python_files (list): A list that holds the generated python
        python file paths.
    ReturnL
        None
    """
    module_file_name = f"{MODULE_FILE_PREFIX}{item[1].lower()}.py"
    module_file_path = f"{temp_dir}/{module_file_name}"
    try:
        with open(module_file_path, "w") as f:
            output_list.append(module_file_path)
            f.write(item[0])
    except IOError:
        assert False, f"Unable to write to {module_file}."
    python_files.append(module_file_path)

def generate_grfn(
    python_source_string,
    python_file_path,
    lambdas_file_path,
    mode_mapper_dictionary,
    original_fortran_file,
    tester_call,
    network_test,
    mod_log_file_path,
    processing_modules,
    save_file=True
):
    """This function generates GrFN dictionary object and file.

    Args:
        python_source_string (str): A string of python code.
        python_file_path (str): A generated python file path.
        lambdas_file_path (str): A lambdas file path.
        mode_mapper_dictionary (list): A mapper of file info (i.e. filename,
        module, and exports, etc).
        original_fortran_file (str): The path to the original
        Fortran file being analyzed.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).
        network_test (bool): A boolean condition to mark whether current
        execution is part of network test or not.
        mod_log_file_path (str): A path to module log file.
        processing_modules (bool): A boolean condition marker to indicate
        whether current GrFN generation is for a module.
        save_file (bool): A boolean condition to mark whether to save generated
        file or not.

    Returns:
        dict: A dictionary of generated GrFN.
    """
    if not tester_call:
        print(
            "+Generating GrFN files: Func: <create_grfn_dict>, Script: "
            "<genPGM.py>"
        )

    module_file_exist = False
    module_paths = []
    grfn_filepath_list = []

    # Regular expression to identify the path and name of all python files
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).py")

    # First, find the main python file in order to populate the module
    # mapper
    file_match = re.match(filename_regex, python_file_path)
    assert file_match, "Invalid filename."

    path = file_match.group("path")
    filename = file_match.group("filename")

    module_file_exist = is_module_file(filename)

    # Ignore all python files of modules created by `pyTranslate.py`
    # since these module files do not contain a corresponding XML file.
    if module_file_exist:
        file_name = genPGM.get_original_file_name(original_fortran_file)
        xml_file = f"{path}rectified_{file_name}.xml"
    else:
        xml_file = f"{path}rectified_{filename}.xml"

    # Calling the `get_index` function in `mod_index_generator.py` to
    # map all variables and objects in the various files
    module_mapper = get_index(xml_file, mod_log_file_path)
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
                                        module_file_exist,
                                        module_import_paths
                )

    grfn_file = python_file_path[:-3] + "_GrFN.json"
    if module_file_exist:
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
    genPGM.generate_system_def(
            [python_file_path],
            grfn_filepath_list,
            module_import_paths,
            module_logs,
            original_fortran_file
    )

    log_generated_files([grfn_file, lambdas_file_path])
    if tester_call:
        return grfn_dict, GENERATED_FILE_PATHS
    else:
        # Write GrFN JSON into a file.
        with open(grfn_file, "w") as file_handle:
            file_handle.write(json.dumps(grfn_dict, sort_keys=True, indent=2))


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


def parse_args():
    """This function is for a safe command line
    input. It should receive the fortran file
    name and returns it back to the caller.

    Returns:
        str: A file name of original fortran script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        help="An input fortran file."
    )

    parser.add_argument(
        "-d",
        "--directory",
        nargs="*",
        help="A temporary directory for generated files to be stored."
    )

    parser.add_argument(
        "-r",
        "--root",
        nargs="*",
        help="A root dirctory to begin file scanning."
    )

    parser.add_argument(
        "-m",
        "--moduleLog",
        nargs="*",
        help="Module log file name."
    )

    cmd_lines = sys.argv[1:]
    args = parser.parse_args(cmd_lines)

    fortran_file = args.file[0]

    # User may or may not provide the output path.
    # If provided, return the directory path name.
    # Else, the default "tmp" name.
    if args.directory:
        out_directory = args.directory[0]
    else:
        out_directory = "tmp"

    if args.root:
        root_dir  = args.root[0]
    else:
        root_dir = "."

    if args.moduleLog:
        module_log_file = args.moduleLog[0]
    else:
        module_log_file = MODULE_FILE_NAME

    return fortran_file, out_directory, root_dir, module_log_file


def check_classpath():
    """check_classpath() checks whether the files in OFP_JAR_FILES can all
       be found in via the environment variable CLASSPATH.
    """
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

def cleanup_files(generated_file_paths):
    """This function cleans up all generated files.
    Args:
        generated_file_paths (list): List of all files that were generated.

    Returns:
        None
    """
    for filepath in generated_file_paths:
        if os.path.isfile(filepath):
            os.remove(filepath)


def log_generated_files(file_paths):
    """This function will add generated files' paths
    into the global list GENERATED_FILE_PATHS.
    Args:
        file_path (list): List of file paths.
    Returns:
        None
    """
    GENERATED_FILE_PATHS.extend(file_paths)


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


def get_file_base(original_fortran_filepath):
    """This function splits and extracts only the basename
    from the full path, then returns the tuple of
    (original_fortran_file, base) to the caller.

    Args:
        original_fortran_filepath (str): A string of path to
        the original fortran file.

    Returns:
        str: 'original_fortran_file'. A fortran file name with extension,
        str: 'base'. A fortran file without extension.
    """
    original_fortran_file = np.basename(original_fortran_filepath)
    base = os.path.splitext(original_fortran_file)[0]

    return base


def fortran_to_grfn(
    original_fortran=None,
    tester_call=False,
    network_test=False,
    temp_dir=None,
    root_dir_path=".",
    module_file_name=MODULE_FILE_NAME,
    processing_modules=False,
    save_file=True
):
    """This function invokes other appropriate functions
    to process and generate objects to translate fortran
    to python IR. This function will either be invoked by
    local main function or the outer tester functions,
    such as test_program_analysis.py or network.py.

    Args:
        original_fortran (str): A file name of original fortran script.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).
        network_test (bool): A boolean condition that will indicate
        whether the script was invoked by network.py or not.
        temp_dir (str): A default temporary directory where output
        files will be stored.
        root_dir_path (str): A root directory of the program.
        module_file_name (str): A module log file name.
        processing_modules (bool): A boolean condition marker to indicate
        whether current fortran_to_grfn execution of for processing a module.
        save_file (bool): A boolean condition to mark whether to save the generated
        files or not.

    Returns:
        str {
            'python_src': A string of python code,
            'python_file': A file name of generated python script,
            'lambdas_file': A file name where lambdas will be,
            'json_file': A file name where JSON will be written to,

        }
        dict: mode_mapper_dict, mapper of file info (i.e. filename,
        module, and exports, etc).
    """
    current_dir = "."
    check_classpath()

    # If, for2py runs manually by the user, which receives
    # the path to the file via command line argument.
    if not tester_call and not processing_modules:
        (
            original_fortran_file,
            temp_out_dir,
            root_dir,
            module_file
        ) = parse_args()
    # Else, for2py function gets invoked by the test
    # programs, it will be passed with an argument
    # of original fortran file path.
    else:
        original_fortran_file = original_fortran
        temp_out_dir = "tmp"
        root_dir = root_dir_path
        module_file = module_file_name

    # Generate or update module log file.
    module_log_file_path = genModFileLog.mod_file_log_generator(root_dir, module_file)

    base = get_file_base(original_fortran_file)

    # temp_dir is None means that the output file was
    # not set by the program that calls this function.
    # Thus, generate the output temporary file based
    # on the user input or the default path "tmp".
    if temp_dir is None:
        temp_dir = current_dir + "/" + temp_out_dir

    # If "tmp" directory does not exist already,
    # simply create one.
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    else:
        assert (
            os.access(temp_dir, os.W_OK)
        ), f"Directory {temp_dir} is not writable.\n\
            Please, provide the directory name to hold files."

    print(f"*** ALL OUTPUT FILES LIVE IN [{temp_dir}]")

    # To avoid the system.json confliction between each
    # f2grfn execution, we need to remove the system.json
    # from the directory first.
    system_json_path = temp_dir + "/" + "system.json"
    if os.path.isfile(system_json_path) and not processing_modules:
        rm_system_json = "rm " + system_json_path
        os.system(rm_system_json)

    # Output files
    preprocessed_fortran_file = temp_dir + "/" + base + "_preprocessed.f"
    ofp_file = temp_dir + "/" + base + ".xml"
    rectified_xml_file = temp_dir + "/" + "rectified_" + base + ".xml"
    pickle_file = temp_dir + "/" + base + "_pickle"
    variable_map_file = temp_dir + "/" + base + "_variables_pickle"
    python_file = temp_dir + "/" + base + ".py"
    output_file = temp_dir + "/" + base + "_outputList.txt"
    json_file = temp_dir + "/" + base + ".json"
    lambdas_file_suffix = "_lambdas.py"

    # Open and read original fortran file
    try:
        with open(original_fortran_file, "r", encoding="utf-8") as f:
            input_lines = f.readlines()
    except IOError:
        assert False, f"Fortran file: {original_fortran_file} Not Found"

    # Pre-process the read in fortran file
    if not tester_call:
        print(
            "+Generating preprocessed fortran file:\
                Func: <process>, Script: <preprocessor.py>"
        )
    try:
        with open(preprocessed_fortran_file, "w") as f:
            f.write(preprocessor.process(input_lines))
    except IOError:
        assert False, "Unable to write to file: {preprocessed_fortran_file}"

    log_generated_files([system_json_path, preprocessed_fortran_file])
    
    # Generate OFP XML from preprocessed fortran
    ofp_xml = generate_ofp_xml(
        preprocessed_fortran_file, ofp_file, tester_call
    )

    # Recify and generate a new xml from OFP XML
    rectified_tree, module_files_to_process = generate_rectified_xml(
        ofp_xml, rectified_xml_file, tester_call, original_fortran_file,
        module_log_file_path
    )

    # If a program uses module(s) that does not reside within the same file,
    # we need to find out where they live and process those files first. Thus,
    # the program should know the list of modules that require advance process
    # that was collected by rectify.py, below code will recursively call fortran_to_grfn
    # function to process module files end-to-end (Fortran-to-GrFN).
    if module_files_to_process:
        processing_modules = True
        for target_module_file in module_files_to_process:
            fortran_to_grfn(
                    target_module_file, tester_call, 
                    network_test, temp_dir, root_dir, 
                    module_file, processing_modules
            )
        processing_modules = False

    # Generate separate list of modules file.
    mode_mapper_tree = rectified_tree
    generator = mod_index_generator.ModuleGenerator()
    mode_mapper_dict = generator.analyze(mode_mapper_tree, module_log_file_path)

    # This will update the log file with more information about the module,
    # such as the declared symbols and types that is needed for generating GrFN.
    if processing_modules:
        genModFileLog.update_mod_info_json(module_log_file_path, mode_mapper_dict[0])

    # Creates a pickle file.
    output_dict = generate_outputdict(
        rectified_tree, preprocessed_fortran_file, pickle_file, tester_call
    )
    
    translated_python_files = []
    # Create a python source file.
    python_source = generate_python_src(
                        output_dict, 
                        translated_python_files,
                        python_file,
                        output_file,
                        variable_map_file,
                        temp_dir,
                        tester_call
                    )
    # Add generated list of files and log it for later removal, if needed.
    file_list = [output_file, pickle_file, variable_map_file]
    file_list.extend(translated_python_files)
    log_generated_files(file_list)

    if tester_call:
        return (
            python_source,
            json_file,
            translated_python_files,
            base,
            mode_mapper_dict,
            original_fortran_file,
            module_log_file_path,
            processing_modules,
        )

    python_file_num = 0
    # Generate GrFN file
    for python_file in translated_python_files:
        lambdas_file_path  = python_file[0:-3] + lambdas_file_suffix
        grfn_dict = generate_grfn(
            python_source[python_file_num][0],
            python_file,
            lambdas_file_path,
            mode_mapper_dict[0],
            original_fortran_file,
            tester_call,
            network_test,
            module_log_file_path,
            processing_modules
        )
        python_file_num += 1


if __name__ == "__main__":
    fortran_to_grfn()
