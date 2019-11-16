"""
This program will scann all Fortran files in the given path
searching for files that hold modules. Then, it will create
a log file in JSON format.

Example:
        This script can be executed as below:
        $ python genModFileLog.py -d <root_directory> -f <log_file_name>
        
fortran_file_path: Original input file that uses module file.
log_file_name: User does not have to provide the name as it is default
to "modFileLog.json", but (s)he can specify it with -f option follow by
the file name in string.

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
        "-d",
        "--directory",
        nargs="+",
        help="Root directory to begin the module scan from."
    )

    parser.add_argument(
        "-f",
        "--file",
        nargs="*",
        help="A user specified module log file name."
    )

    args = parser.parse_args(sys.argv[1:])

    root_dir_path = args.directory[0]
    if args.file is not None:
        user_specified_log_file_name = args.file[0]
        return root_dir_path, user_specified_log_file_name
    else:
        default_module_file_name = "modFileLog.json"
        return root_dir_path, default_module_file_name

def get_file_list_in_directory(root_dir_path):
    """This function lists all Fortran files (excluding directories)
    in the specified directory.
    Args:
        dir_path (str): Directory path.

    Returns:
        List: List of Fortran files.
    """
    files = []
    for (dir_path, dir_names, file_names) in os.walk(root_dir_path):
        for f in file_names:
            if (
                    "_preprocessed" not in f
                    and (f.endswith('.f') or f.endswith('.for'))
            ):
                files += [os.path.join(dir_path, f)]
    return files

def modules_from_file(file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict):
    """This function checks either the module and file path already exist
    int the log file. If it does, then it compares the last_modified_time
    in the log file with the last modified time of file in disk. Then, it
    will call 'populate_mapper' function if file was not already looked
    before or the file was modified since looked last time.
    Args:
        file_path (str): File path that is guaranteed to exist in
        the directory.
        file_to_mod_mapper (dict): Dictionary of lists that will
        hold file-to-module_name mappings.
        mod_to_file_mapper (dict): Dictionary that holds a module
        to its residing file path.
    Returns:
        None.
    """
    last_modified_time = get_file_last_modified_time(file_path)

    if file_path in file_to_mod_mapper:
        last_modified_time_in_log = file_to_mod_mapper[file_path][-1]
        if last_modified_time  == last_modified_time_in_log:
            return
        else:
            assert (
                    last_modified_time > last_modified_time_in_log
            ), "Last modified time in the log file cannot be later than on disk file's time."
            populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict)
    else:
        populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict)

def populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict):
    """This function populates two mappers by checking and extracting 
    module names, if exist, from the file, and map it to the file name.
    Args:
        file_path (str): File of a path that will be scanned.
        file_to_mod_mapper (dict): Dictionary of lists that will
        hold file-to-module_name mappings.
        mod_to_file_mapper (dict): Dictionary that holds a module
        to its residing file path.
    Returns:
        None.
    """
    with open(file_path, encoding = "ISO-8859-1") as f:
        file_content = f.read()
    module_names = []
    module_names_lowered = []
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
        module_names_lowered = [mod.lower() for mod in module_names]
        file_to_mod_mapper[file_path] = module_names_lowered.copy()
        file_to_mod_mapper[file_path].append(get_file_last_modified_time(file_path))

    for mod in module_names_lowered:
        mod_to_file_mapper[mod] = [file_path]
        mod_info_dict[mod] = {"exports":{}, "symbol_types":{}, "imports":{}}

def get_file_last_modified_time(file_path):
    """This function retrieves the file status and assigns the last modified
    time of a file at the end of the file_to_mod_mapper[file_path] list.

    Params:
        file_path (str): File path that is guaranteed to exist in
        the directory.
    Returns:
        int: Last modified time represented in integer.
    """
    file_stat = os.stat(file_path)
    return file_stat[8]

def update_mod_info_json(module_log_file_path, mode_mapper_dict):
    """This function updates each module's information, such as
    the declared variables and their types, so that genPGM.py
    can simply reference this dictionary rather than processing
    the file again.
    
    Params:
        module_log_file_path (str): A path of module log file.
        mode_mapper_dict (dict): A dictionary that holds all information
        of a module(s).
    """
    mod_info = {"exports": mode_mapper_dict["exports"]}
    symbol_types = {}
    for mod_name, mod_symbols in mode_mapper_dict["exports"].items():
        sym_type = {}
        for sym in mod_symbols:
            if sym in mode_mapper_dict["symbol_types"]:
                m_type = mode_mapper_dict["symbol_types"][sym]
                sym_type[sym] = m_type[1]
            elif (
                    mod_name in mode_mapper_dict["subprograms"] and
                    sym in mode_mapper_dict["subprograms"][mod_name]
            ):
                sym_type[sym] = "func"
        symbol_types[mod_name] = sym_type
    mod_info["symbol_types"] = symbol_types

    with open(module_log_file_path) as json_f:
        module_logs = json.load(json_f)

    for module in mode_mapper_dict["modules"]:
        mod = module_logs["mod_info"][module]
        mod["exports"] = mod_info["exports"][module]
        mod["symbol_types"] = mod_info["symbol_types"][module]
        if module in mode_mapper_dict["imports"]:
            imports = mode_mapper_dict["imports"][module]
        else:
            imports = []
        mod["imports"] = imports 
        module_logs["mod_info"][module] = mod

    with open(module_log_file_path, 'w+') as json_f:
        json_f.write(json.dumps(module_logs, indent=2))


def mod_file_log_generator(
        root_dir_path=None,
        module_log_file_name=None,
):
    """This function is like a main function to invoke other functions
    to perform all checks and population of mappers. Though, loading of
    and writing to JSON file will happen in this function.

    Args: 
        temp_dir (str): Temporary directory that log JSON file resides.

    Returns:
        None.
    """
    if root_dir_path == None:
        root_dir_path = "."
    module_log_file_path = root_dir_path + "/" + module_log_file_name

    # If module log file already exists, simply load data.
    if isfile(module_log_file_path):
        with open(module_log_file_path) as json_f:
            module_logs = json.load(json_f)
        # This will hold the file-to-module and file last modified date info.
        # One thing to notice is that the last index will be a place for
        # last modified time for file.
        # Structure (One-to-Many):
        #   {
        #       "__file_name__" : ["__module_name__",...,"last_modified_time"],
        #       ...,
        #   }
        file_to_mod_mapper = module_logs["file_to_mod"]
        # This will hold the module-to-file mapping, so any program that accesses
        # module log JSON file can directly access the file path with the module
        # name specified with "USE" without looping through the file_to_mod mapper.
        # Structure (One-to-One):
        #   {
        #       "__module_name__" : "__file_path__",
        #       ...,
        #   }
        mod_to_file_mapper = module_logs["mod_to_file"]
        mod_info_dict = module_logs["mod_info"]
    else:
        file_to_mod_mapper = {}
        mod_to_file_mapper = {}
        mod_info_dict = {}

    files = get_file_list_in_directory(root_dir_path)
   
    for file_path in files:
        modules_from_file(file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict)
    module_log = {"file_to_mod": file_to_mod_mapper, "mod_to_file": mod_to_file_mapper, "mod_info": mod_info_dict}

    with open(module_log_file_path, 'w+') as json_f:
        json_f.write(json.dumps(module_log, indent=2))

    return module_log_file_path

if __name__ == "__main__":
    root_dir_path, module_log_file = parse_args()
    log_file_path = mod_file_log_generator(root_dir_path, module_log_file)
