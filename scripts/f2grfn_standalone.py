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

        $ python -f <fortran_file>

fortran_file: An original input file to a program that is to be
    translated to GrFN.
root_dir: A root directory where module log file should be created or
    found.

Author: Terrence J. Lim
"""

import json
import argparse
from delphi.translators.for2py import f2grfn

if __name__ == "__main__":
    """This function is for a safe command line
    input. It should receive the fortran file
    name and returns it back to the caller.

    Returns:
        str: A file name of original fortran script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--file", type=str, help="An input fortran file."
    )

    parser.add_argument(
        "-d",
        "--directory",
        nargs="*",
        help="A temporary directory for generated files to be stored.",
        default=".",
    )

    parser.add_argument(
        "-r",
        "--root",
        nargs="*",
        help="A root directory to begin file scanning.",
        default=".",
    )

    parser.add_argument(
        "-m",
        "--moduleLog",
        nargs="*",
        help="Module log file name.",
        default="modFileLog.json",
    )

    args = parser.parse_args()
    (
        python_sources,
        translated_python_files,
        mode_mapper_dict,
        original_fortran_file,
        module_log_file_path,
        processing_modules,
    ) = f2grfn.fortran_to_grfn(
        args.file, args.directory, args.root, args.moduleLog
    )
    python_file_num = 0
    for python_file in translated_python_files:
        lambdas_file_path = python_file.replace(".py", "_lambdas.py")
        grfn_dict = f2grfn.generate_grfn(
            python_sources[python_file_num][0],
            python_file,
            lambdas_file_path,
            mode_mapper_dict[0],
            original_fortran_file,
            module_log_file_path,
            processing_modules,
        )
        with open(python_file.replace(".py","_GrFN.json"), "w") as f:
            json.dump(grfn_dict, f, indent=2)
        python_file_num += 1
