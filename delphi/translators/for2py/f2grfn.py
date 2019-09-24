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

        $ python f2grfn -f <fortran_file>

fortran_file: An original input file to a program that is
to be translated to GrFN.

Author: Terrence J. Lim
"""

import os
import sys
import ast
import argparse
import pickle
import delphi.paths
import ntpath as np
import subprocess as sp
import xml.etree.ElementTree as ET

from delphi.translators.for2py import (
    preprocessor,
    translate,
    get_comments,
    pyTranslate,
    genPGM,
    mod_index_generator,
    rectify,
)

OFP_JAR_FILES = [
    "antlr-3.3-complete.jar",
    "commons-cli-1.4.jar",
    "OpenFortranParser-0.8.4-3.jar",
    "OpenFortranParserXML-0.4.1.jar",
]
"""OFP_JAR_FILES is a list of JAR files used by the Open Fortran Parser (OFP).
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

    return ofp_xml


def generate_rectified_xml(ofp_xml: str, rectified_file, tester_call):
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

    Returns:
        Element Tree (ET) Object: An object of generated rectified
        XML.
    """

    if not tester_call:
        print(
            "+Generating rectified XML: Func: <buildNewASTfromXMLString>, "
            "Script: <rectify.py>"
        )

    rectified_xml = rectify.buildNewASTfromXMLString(ofp_xml)

    if not tester_call:
        rectified_tree = ET.ElementTree(rectified_xml)
        try:
            rectified_file_handle = open(rectified_file, 'w')
            rectified_file_handle.close()
        except IOError:
            assert False, f"Failed to write to {rectified_file}."

        rectified_tree.write(rectified_file)

    return rectified_xml


def generate_outputdict(
    rectified_tree, preprocessed_fortran_file, pickle_file, tester_call
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
    output_dictionary, python_file_name,
    output_file, temp_dir, tester_call
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

    python_source = pyTranslate.create_python_source_list(output_dictionary)

    if not tester_call:
        print(
            "+Generating python source file:\
                Func: <create_python_source_list>,\
                Script: <pyTranslate.py>"
        )

        output_list = []
        for item in python_source:
            if item[2] == "module":
                module_file = f"{temp_dir}/m_{item[1].lower()}.py"
                try:
                    with open(module_file, "w") as f:
                        output_list.append("m_" + item[1].lower() + ".py")
                        f.write(item[0])
                except IOError:
                    assert False, f"Unable to write to {module_file}."
            else:
                try:
                    with open(python_file_name, "w") as f:
                        output_list.append(python_file_name)
                        f.write(item[0])
                except IOError:
                    assert False, f"Unable to write to {python_file_name}."

        try:
            with open(output_file, "w") as f:
                for fileName in output_list:
                    f.write(fileName + " ")
        except IOError:
            assert False, f"Unable to write to {output_file}."

    return python_source


def generate_grfn(
    python_source_string, python_filename, lambdas_file_suffix,
    mode_mapper_dictionary, original_fortran_file, tester_call
):
    """This function generates GrFN dictionary object and file.

    Args:
        python_source_string (str): A string of python code.
        python_filename (str): A file name of generated python script.
        lambdas_file_suffix (str): The suffix of the file name where
        lambdas will be written to.
        mode_mapper_dictionary (list): A mapper of file info (i.e. filename,
        module, and exports, etc).
        original_fortran_file (str): The path to the original
        Fortran file being analyzed.
        tester_call (bool): A boolean condition that will indicate
        whether the program was invoked standalone (False) or
        by tester scripts (True).

    Returns:
        dict: A dictionary of generated GrFN.
    """

    if not tester_call:
        print(
            "+Generating GrFN files: Func: <create_grfn_dict>, Script: "
            "<genPGM.py>"
        )
        # Since process_files function invokes create_grfn_dict
        # function, we only have to call process_files in case
        # of non-test mode.
        genPGM.process_files(
                [python_filename], "GrFN.json",
                "lambdas.py", original_fortran_file,
                False
        )
    else:
        asts = [ast.parse(python_source_string)]
        grfn_dictionary = genPGM.create_grfn_dict(
            lambdas_file_suffix, asts, python_filename, mode_mapper_dictionary,
            original_fortran_file, save_file=True
        )
        del grfn_dictionary["date_created"]
        for item in grfn_dictionary["variables"]:
            if "gensym" in item:
                del item["gensym"]
        for item in grfn_dictionary["containers"]:
            if "gensym" in item:
                del item["gensym"]

        return grfn_dictionary

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
        help="An input fortran file"
    )

    parser.add_argument(
        "-d",
        "--directory",
        nargs="*",
        help="A temporary directory for generated files to be stored"
    )

    args = parser.parse_args(sys.argv[1:])

    fortran_file = args.file[0]

    # User may or may not provide the output path.
    # If provided, return the directory path name.
    # Else, the default "tmp" name.
    if args.directory is not None:
        out_directory = args.directory[0]
    else:
        out_directory = "tmp"

    return fortran_file, out_directory


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
    # the path to the file via command line argument
    if not tester_call:
        (
            original_fortran_file,
            temp_out_dir
        ) = parse_args()
    # Else, for2py function gets invoked by the test
    # programs, it will be passed with an argument
    # of original fortran file path
    else:
        original_fortran_file = original_fortran
        temp_out_dir = "tmp"

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

    # Output files
    preprocessed_fortran_file = temp_dir + "/" + base + "_preprocessed.f"
    ofp_file = temp_dir + "/" + base + ".xml"
    rectified_xml_file = temp_dir + "/" + "rectified_" + base + ".xml"
    pickle_file = temp_dir + "/" + base + "_pickle"
    translated_python_file = temp_dir + "/" + base + ".py"
    output_file = temp_dir + "/" + base + "_outputList.txt"
    json_suffix = temp_dir + "/" + base + ".json"
    lambdas_suffix = temp_dir + "/" + base + "_lambdas.py"

    # Open and read original fortran file
    try:
        with open(original_fortran_file, "r") as f:
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
        assert False, "Unable to write tofile: {preprocessed_fortran_file}"

    # Generate OFP XML from preprocessed fortran
    ofp_xml = generate_ofp_xml(
        preprocessed_fortran_file, ofp_file, tester_call
    )

    # Rectify and generate a new xml from OFP XML
    rectified_tree = generate_rectified_xml(
        ofp_xml, rectified_xml_file, tester_call
    )

    # Generate separate list of modules file
    mode_mapper_tree = rectified_tree
    generator = mod_index_generator.ModuleGenerator()
    mode_mapper_dict = generator.analyze(mode_mapper_tree)

    # Creates a pickle file
    output_dict = generate_outputdict(
        rectified_tree, preprocessed_fortran_file, pickle_file, tester_call
    )

    # Create a python source file
    python_source = generate_python_src(
        output_dict, translated_python_file, output_file, temp_dir, tester_call
    )

    if tester_call:
        os.remove(preprocessed_fortran_file)

    if not network_test:
        return (
            python_source,
            lambdas_suffix,
            json_suffix,
            translated_python_file,
            mode_mapper_dict,
            original_fortran_file,
        )
    else:
        #  TODO: This is related to networks.py and subsequent GrFN
        #  generation. Change the python_src index from [0][0] to incorporate
        #  all modules after all GrFN features have been added
        return (python_source[0][0], lambdas_suffix, json_suffix, base, original_fortran_file, mode_mapper_dict[0])


if __name__ == "__main__":
    (
        python_src,
        lambdas_file,
        json_file,
        python_file,
        mode_mapper_dict,
        original_fortran_file
    ) = fortran_to_grfn()

    # Generate GrFN file
    grfn_dict = generate_grfn(
        python_src[0][0], python_file, lambdas_file, mode_mapper_dict[0],
        original_fortran_file, False
    )
