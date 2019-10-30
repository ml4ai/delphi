import os
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
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).f")
    file_match = re.match(filename_regex, file_path)

    dir_path = file_match.group("path")
    return dir_path

def mod_file_log_generator():
    file_path = parse_args()
    dir_path = get_directory_path(file_path)
    print (dir_path)

if __name__ == "__main__":
    mod_file_log_generator()
