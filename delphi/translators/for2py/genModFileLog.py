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
from os.path import isfile
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
    """This function lists all Fortran fi
    les (excluding directories)
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


def modules_from_file(file_path, file_to_mod_mapper, mod_to_file_mapper,
                      mod_info_dict):
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
        if last_modified_time == last_modified_time_in_log:
            return
        else:
            assert (
                    last_modified_time > last_modified_time_in_log
            ), "Last modified time in the log file cannot be later than on " \
               "disk file's time."
            populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper,
                             mod_info_dict)
    else:
        populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper,
                         mod_info_dict)


def populate_mappers(file_path, file_to_mod_mapper, mod_to_file_mapper,
                     mod_info_dict):
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
    f = open(file_path, encoding="ISO-8859-1")
    f_pos = f.tell()
    file_content = f.read()

    module_names = []
    module_names_lowered = []
    module_summary = {}
    # Checks if file contains "end module" or "endmodule",
    # which only appears in case of module declaration.
    # If not, there is no need to look into the file any further,
    # so ignore it.
    if (
            "end module" in file_content.lower()
            or "endmodule" in file_content.lower()
    ):
        # Extract the module name that follows 'end module' or 'endmodule'
        # These two lines will extract all module names in the file.
        module_names.extend(re.findall(r'(?i)(?<=end module )[^-. \n]*',
                                       file_content))
        module_names.extend(re.findall(r'(?i)(?<=endmodule )[^-. \n]*',
                                       file_content))
        module_names_lowered = [mod.lower() for mod in module_names]
        file_to_mod_mapper[file_path] = module_names_lowered.copy()
        file_to_mod_mapper[file_path].append(get_file_last_modified_time(
            file_path))
        if (
            ("end subroutine" in file_content.lower()
                or "endsubroutine" in file_content.lower())
            or ("end function" in file_content.lower()
                    or "endfunction" in file_content.lower())
        ):
            # Bring the pointer back to the first character position of a file
            f.seek(f_pos)
            # regex for module, subroutine, and function
            # Because Fortran allows two different syntax for end (i.e. end module and
            # endmodule), we need to check both cases.
            modu_regex = re.compile('\s*module (?P<module>(?P<name>(\w*)\n))')
            end_modu_regex = re.compile('\s*end module (?P<module>(?P<name>(\w*)\n))')
            endmodu_regex = re.compile('\s*endmodule (?P<module>(?P<name>(\w*)\n))')
            
            interface_regex  = re.compile('\s*interface (?P<interface>(?P<name>(\w*)\n))')
            end_intr_regex = re.compile('\s*end interface (?P<interface>(?P<name>(\w*)\n))')
            endintr_regex = re.compile('\s*endinterface (?P<interface>(?P<name>(\w*)\n))')

            subr_regex = re.compile('\s*subroutine (?P<subroutine>(?P<name>.*?)\((?P<args>.*)\))')
            end_subr_regex = re.compile('\s*end subroutine (?P<subroutine>(?P<name>(\w*)\n))')
            endsubr_regex = re.compile('\s*endsubroutine (?P<subroutine>(?P<name>(\w*)\n))')
            
            func_regex = re.compile('\s*(?P<type>(double precision|double|float|int|integer|logical|real|str|string))\sfunction (?P<function>(?P<name>.*?)\((?P<args>.*)\))')
            end_func_regex = re.compile('\s*end function (?P<function>(?P<name>(\w*)\n))')
            endfunc_regex = re.compile('\s*endfunction (?P<function>(?P<name>(\w*)\n))')

            var_dec_regex = re.compile('\s*(?P<type>(double precision|double|float|int|integer|logical|real|str|string))\s(::)*(?P<variables>(.*))\n')

            current_modu = None
            current_subr = None
            current_func = None

            line = f.readline().lower()
            while (line):
                #  Removing any inline comments
                if  '!' in line:
                    line = line.partition('!')[0].strip()
                
                # Check enter and exit of module
                modu = modu_regex.match(line)
                end_modu = end_modu_regex.match(line)
                endmodu = endmodu_regex.match(line)

                # Check enter and exit of subroutine
                subr = subr_regex.match(line)
                # Check enter and exit of function
                func = func_regex.match(line)

                if modu:
                    current_modu = modu['name'].strip()
                    module_summary[current_modu] = {}
                elif end_modu or endmodu:
                    current_modu = None
                else:
                    pass

                if current_modu:
                    if subr:
                        current_subr = subr["name"].strip()
                        subr_args = subr["args"].replace(' ','').split(',')
                        module_summary[current_modu][current_subr] = {}
                        for arg in subr_args:
                            if arg:
                                module_summary[current_modu][current_subr][arg] = None
                    elif current_subr:
                        variable_dec = var_dec_regex.match(line)
                        if variable_dec and not func:
                            # DEBUG
                            # print ("    @ line: ", line)
                            # print ("        current_modu: ", current_modu)
                            # print ("        current_subr: ", current_subr)
                            # print ("        type: ", variable_dec['type'])
                            # print ("        variables: ", variable_dec['variables'])
                            var_type = variable_dec['type']
                            variables = variable_dec['variables']
                            if "precision" not in variables and "dimension" not in variables:
                                var_list = variables.split(',')
                                # DEBUG
                                # print ("       module_summary[current_modu]: ", module_summary[current_modu])
                                for var in var_list:
                                    if (
                                            current_subr in module_summary[current_modu]
                                            and var.strip() in module_summary[current_modu][current_subr]
                                    ):
                                        module_summary[current_modu][current_subr][var.strip()] = var_type
                    elif end_subr_regex or endsubr_regex:
                        current_subr = None
                    elif func:
                        func_name = func["name"]
                        func_args = func["args"]
                line  = f.readline().lower()

    for mod in module_names_lowered:
        mod_to_file_mapper[mod] = [file_path]
        mod_info_dict[mod] = {
            "exports": {},
            "symbol_types": {},
            "imports": {},
            "function_summary": {},
            "interface_functions": {}
        }
        if mod in module_summary:
            mod_info_dict[mod]["function_summary"] = module_summary[mod]

    f.close()


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
    if not root_dir_path:
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
        # This will hold the module-to-file mapping, so any program that
        # accesses module log JSON file can directly access the file path
        # with the module name specified with "USE" without looping through
        # the file_to_mod mapper.
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
        modules_from_file(file_path, file_to_mod_mapper, mod_to_file_mapper,
                          mod_info_dict)
    module_log = {
        "file_to_mod": file_to_mod_mapper,
        "mod_to_file": mod_to_file_mapper,
        "mod_info": mod_info_dict
    }

    with open(module_log_file_path, 'w+') as json_f:
        json_f.write(json.dumps(module_log, indent=2))

    return module_log_file_path


if __name__ == "__main__":
    root_dir_path, module_log_file = parse_args()
    log_file_path = mod_file_log_generator(root_dir_path, module_log_file)
