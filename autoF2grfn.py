"""This script automatically runs all fortran files in the user provided
directory path with f2grfn_standalone.py.

For example:
    $python3 autoF2grfn.py -d <directory_path>

directory_path: A path that the user specify where fortran files are stored.

Output:
    errors.log: Collects all produced errors during running f2grfn_standalone.py

Author: Terrence Lim
"""

import os
import sys
import argparse
import subprocess as sp

FORTRAN_FILE_EXTENSIONS = ["f90", "for", "f"]


def get_file_list_in_directory(target_dir_path):
    """This function lists all Fortran fi
    les (excluding directories)
    in the specified directory.
    Args:
        target_dir_path (str): Directory path.

    Returns:
        List: List of Fortran files.
    """
    files = []
    for (dir_path, dir_names, file_names) in os.walk(target_dir_path):
        for f in file_names:
            fext = f.split('.')[-1]
            if fext in FORTRAN_FILE_EXTENSIONS:
                files += [os.path.join(dir_path, f)]
    return files


def parse_args():
    """This function is for a safe command line
    input. It should receive the fortran file
    name and returns it back to the caller.

    Returns:
        str: A file name of original fortran script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        nargs="+",
        help="Root directory to begin the module scan from."
    )

    args = parser.parse_args(sys.argv[1:])

    # Return directory path
    return args.directory[0]


def main():
    dir_path = parse_args()
    list_of_files = get_file_list_in_directory(dir_path)

    total = num_of_files = len(list_of_files)
    logf = open("errors.log", "w")
    for f in list_of_files:
        num_of_files -= 1
        print(f"File: {f}")
        print(f"Number of files left: {num_of_files}")
        output = sp.run(
            [
                "python",
                "scripts/f2grfn_standalone.py",
                "-f",
                f,
                "-d",
                "tmp/",
            ],
            stderr=sp.PIPE,
        ).stderr
        output_str = str(output).split('\\n')
        logf.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        logf.write(f"@@ CURRENTLY HANDLING: {f}\n")
        for ost in output_str:
            logf.write(f"{ost}\n")
        logf.write("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
    print(f"{total} files processed.")


main()
