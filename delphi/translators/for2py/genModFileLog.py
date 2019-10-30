"""
This program will scann all Fortran files in the given path
searching for files that hold modules. Then, it will create
a log file in JSON format.

Example:
        This script can be executed as below:
        $ python genModFileLog.py -f <fortran_file_path>
        
fortran_file_path: Original input file that uses module file.

Currently, this program assumes that module files reside in
the same directory as use program file.

Author: Terrence J. Lim
"""

import os
from os.path import isfile, join
import re
import sys
import json
import argparse

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

    args = parser.parse_args(sys.argv[1:])

    file_path = args.file[0]

    return file_path

def get_directory_path(file_path):
    """This function extracts the directory path from
    the input file path.
    Args:
        file_path (str): File path.
    Returns:
        str: Directory path.
    """
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).f")
    file_match = re.match(filename_regex, file_path)

    dir_path = file_match.group("path")
    return dir_path

def get_file_list_in_directory(dir_path):
    """This function lists all Fortran files (excluding directories)
    in the specified directory.
    Args:
        dir_path (str): Directory path.

    Returns:
        List: List of Fortran files.
    """
    files = []
    for f in os.listdir(dir_path):
        if (
                isfile(join(dir_path, f))
                and (f.endswith('.f')
                    or f.endswith('.for'))
        ):
            files.append(f)

    return files

def modules_from_file(file_path, file_to_mod_mapper):
    """This function checks and extracts module names, if exist,
    from the file, and map it to the file name.
    Args:
        file_path (str): File path that is guaranteed to exist in
        the directory.
        file_to_mod_mapper (dict): Dictionary of lists that will
        hold file-to-module_name mappings.
    Returns:
        None.
    """
    with open(file_path) as f:
        file_content = f.read()
    module_names = []
    # Checks if file contains "end module" or "end module",
    # which only appears in case of module declaration.
    # If not, there is no need to look into the file any further,
    # so ignore it.
    if (
            "end module" in file_content.lower()
            or "endmodule" in file_content.lower()
    ):
        # Extract the module name that follows 'end module' or 'endmodule'
        # These two lines will extract all module names in the file.
        module_names.extend(re.findall(r'(?i)(?<=end module )[^-. \n]*', file_content))
        module_names.extend(re.findall(r'(?i)(?<=endmodule )[^-. \n]*', file_content))

    file_to_mod_mapper[file_path] = module_names

def get_file_last_modified_time(file_name, file_to_mod_mapper):
    file_stat = os.stat(file_name)

def mod_file_log_generator():
    file_path = parse_args()
    dir_path = get_directory_path(file_path)

    files = get_file_list_in_directory(dir_path)

    # This will hold the file-to-module and file last modified date info.
    # One thing to notice is that the last index will be the place for
    # last modified time for file.
    # Structure:
    #   {
    #       "__file_name__" : ["__module_name__",...,"last_modified_time"],
    #       ...
    #   }
    file_to_mod_mapper = {}
    for f in files:
        file_path = dir_path + f
        modules_from_file(file_path, file_to_mod_mapper)

    print (file_to_mod_mapper)

if __name__ == "__main__":
    mod_file_log_generator()
