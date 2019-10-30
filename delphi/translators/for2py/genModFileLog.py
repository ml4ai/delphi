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

def get_files_in_directory(dir_path):
    """This function lists all Fortran files (excluding directories)
    in the specified directory.
    Args:
        dir_path (str): Directory path.

    Returns:
        List: List of Fortran files.
    """
    files = []
    for f in os.listdir(dir_path):
        if isfile(join(dir_path, f)):
            files.append(f)

    return files

def mod_file_log_generator():
    file_path = parse_args()
    dir_path = get_directory_path(file_path)

    files = get_files_in_directory(dir_path)

    # DEBUG
    print (files)

if __name__ == "__main__":
    mod_file_log_generator()
