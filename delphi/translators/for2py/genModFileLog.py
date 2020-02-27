"""
This program will scan all Fortran files in the given path searching for files
that hold modules. Then, it will create a log file in JSON format.

Example:
        This script can be executed as below:
        $ python genModFileLog.py -d <root_directory> -f <log_file_name>

fortran_file_path: Original input file that uses module file.
log_file_name: User does not have to provide the name as it is default
to "modFileLog.json", but (s)he can specify it with -f option follow by
the file name in string.

Currently, this program assumes that module files reside in the same directory
as use program file.

Author: Terrence J. Lim
"""

import os
import re
import sys
import json
import argparse
from os.path import isfile
from delphi.translators.for2py import syntax, preprocessor


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
        help="Root directory to begin the module scan from.",
    )

    parser.add_argument(
        "-f",
        "--file",
        nargs="*",
        help="A user specified module log file name.",
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
    """This function lists all Fortran files (excluding directories) in the
    specified directory.

    Args:
        dir_path (str): Directory path.

    Returns:
        List: List of Fortran files.
    """
    files = []
    for (dir_path, dir_names, file_names) in os.walk(root_dir_path):
        for f in file_names:
            if "_preprocessed" not in f and (
                f.endswith(".f") or f.endswith(".for")
            ):
                path = os.path.join(dir_path, f)
                files.append(os.path.abspath(path))
    return files


def modules_from_file(
    file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict
):
    """This function checks whether the module and file path already exist
    int the log file. If they do, then it compares the last_modified_time
    in the log file with the last modified time of file in disk. Then, it
    will call 'populate_mapper' function if file was not already looked
    before or the file was modified since last analyzed.
    Args:
        file_path (str): File path that is guaranteed to exist in the
            directory.
        file_to_mod_mapper (dict): Dictionary of lists that will hold
            file-to-module_name mappings.
        mod_to_file_mapper (dict): Dictionary that holds a module to its
            residing file path.
    Returns:
        None.
    """

    if file_path in file_to_mod_mapper:
        last_modified_time = get_file_last_modified_time(file_path)
        last_modified_time_in_log = file_to_mod_mapper[file_path][-1]
        if last_modified_time != last_modified_time_in_log:
            assert last_modified_time > last_modified_time_in_log, (
                "Last modified time in the log file cannot be later than on "
                "disk file's time."
            )
            populate_mappers(
                file_path,
                file_to_mod_mapper,
                mod_to_file_mapper,
                mod_info_dict,
            )
    else:
        populate_mappers(
            file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict
        )


def populate_mappers(
    file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict
):
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
    procedure_functions = {}
    derived_types = {}

    # Checks if file contains "end module" or "endmodule",
    # which only appears in case of module declaration.
    # If not, there is no need to look into the file any further,
    # so ignore it.
    if syntax.has_module(file_content.lower()):
        # Extract the module names by inspecting each line in the file.
        f.seek(f_pos)
        org_lines = f.readlines()
        preprocessed_lines = preprocessor.process(
            org_lines, file_path, True
        ).split("\n")
        for line in preprocessed_lines:
            line = line.lower()
            match = syntax.line_starts_pgm(line)
            if match[0] and match[1] == "module":
                module_names.append(match[2])

        # Map current file to modules that it uses.
        module_names_lowered = [mod.lower() for mod in module_names]
        file_to_mod_mapper[file_path] = module_names_lowered.copy()
        file_to_mod_mapper[file_path].append(
            get_file_last_modified_time(file_path)
        )

        # If current file has subroutines, then extract subroutine information
        # that are declared within the scope of any module and store in the module
        # summary dictionary.
        if syntax.has_subroutine(file_content.lower()):
            populate_module_summary(
                preprocessed_lines,
                module_summary,
                procedure_functions,
                derived_types,
            )

    # Using collected function information, populate interface function information
    # by each module.
    populate_procedure_functions(procedure_functions, module_summary)

    # Populate actual module information (summary)
    # that will be written to thee JSONN file.
    for mod in module_names_lowered:
        mod_to_file_mapper[mod] = [file_path]
        mod_info_dict[mod] = {
            "exports": {},
            "symbol_types": {},
            "imports": {},
            "interface_functions": {},
            "derived_type_list": [],
        }
        if mod in module_summary:
            mod_info_dict[mod]["interface_functions"] = procedure_functions[
                mod
            ]
        if mod in derived_types:
            mod_info_dict[mod]["derived_type_list"] = derived_types[mod]
    f.close()


def populate_procedure_functions(procedure_functions, module_summary):
    """This function completes procedure_functions dictionary.

    Params:
        procedure_functions (dict): A dictionary to hold interface-to-procedure
            function mappings.
        module_summary (dict): A dictionary for holding module-to-subroutine-to-
            arguments mappings.

    Returns:
        None.
    """
    for mod in procedure_functions:
        if mod in module_summary:
            mod_functions = module_summary[mod]
            for interface in procedure_functions[mod]:
                for function in procedure_functions[mod][interface]:
                    if function in mod_functions:
                        procedure_functions[mod][interface][
                            function
                        ] = mod_functions[function]


def populate_module_summary(
    f, module_summary, procedure_functions, derived_types
):
    """This function extracts module, derived type, and interface information, and
    populates module summary, procedure functions, and derived types dictionaries.

    Params:
        f (str): File content.
        module_summary (dict): A dictionary for holding module-to-subroutine-to-
            arguments mappings.
        procedure_functions (dict): A dictionary to hold interface-to-procedure
            function mappings.
        derived_types (dict): Dictionary that will hold module-to-derived type
            mapping.
    Returns:
        None.
    """
    current_modu = None
    current_subr = None
    current_intr = None
    current_func = None

    isProcedure = False

    for line in f:
        line = line.lower()
        #  Removing any inline comments
        if "!" in line:
            line = line.partition("!")[0].strip()

        # Detects module and interface entering line.
        pgm = syntax.line_starts_pgm(line)
        # Detects subroutine entering line.
        subroutine = syntax.subroutine_definition(line)

        end_pgm = syntax.pgm_end(line)
        if (
            pgm[0]
            and pgm[1].strip() == "module"
            and pgm[2].strip() != "procedure"
        ):
            current_modu = pgm[2].strip()
            module_summary[current_modu] = {}
            procedure_functions[current_modu] = {}
        elif end_pgm[0] and end_pgm[1] == "module":
            current_modu = None
        else:
            pass

        # If currently processing line of code is within the scope of module,
        # we need to extract subroutine, interface, and derived type information.
        if current_modu:
            current_subr = extract_subroutine_info(
                pgm,
                end_pgm,
                module_summary,
                current_modu,
                subroutine,
                current_subr,
                line,
            )
            current_intr = extract_interface_info(
                pgm,
                end_pgm,
                procedure_functions,
                current_modu,
                current_intr,
                line,
            )
            extract_derived_type_info(end_pgm, current_modu, derived_types)


def extract_subroutine_info(
    pgm, end_pgm, module_summary, current_modu, subroutine, current_subr, line
):
    """This function extracts information of subroutine declared within
    the module, and stores those information to module_summary dictionary.

    Params:
        pgm (tuple): Current program information.
        end_pgm (typle): End of current program indicator.
        module_summary (dict): A dictionary for holding module-to-subroutine-to-
        arguments mappings.
        current_modu (str): Module name that current interface is located under.
        subroutine (tuple): Holds information of the subroutine.
        current_subr (str): Current subroutine name.
        line (str): A line from Fortran source code.
    Returns:
        (current_subr) Currently handling subroutine name.
    """

    # If subroutine encountered,
    if subroutine[0]:
        # extract the name,
        current_subr = subroutine[1][0]
        # extract any existing arguments,
        subroutine_args = subroutine[1][1]
        # and populate thee module summary dictionary
        module_summary[current_modu][current_subr] = {}
        for arg in subroutine_args:
            if arg:
                # Since we cannot find out about the argument types
                # just looking at the argument, initialize the types
                # to None as defualt.
                module_summary[current_modu][current_subr][arg] = None
    elif end_pgm[0] and end_pgm[1] == "subroutine":
        # Indication of end of subroutine.
        current_subr = None
    elif current_subr:
        variable_dec = syntax.variable_declaration(line)
        if variable_dec[0] and not syntax.line_is_func_start(line):
            if variable_dec[0]:
                var_type = variable_dec[1]
                variables = variable_dec[2]

            # Handle syntax like:
            #   precision, dimension(0:tmax) :: means, vars
            #   precision, parameter :: var = 1.234
            if "precision" in variables or "dimension" in variables:
                # Handle dimension (array)
                if "dimension" in variables:
                    var_type = "Array"
                # Extract only variable names follow by '::'
                variables = variables.partition("::")[-1].strip()
                if "=" in variables:
                    # Remove assignment syntax and only extract variable names
                    variables = variables.partition("=")[0].strip()

            var_list = variables.split(",")
            for var in var_list:
                # Search for an implicit array variable declaration
                arrayVar = syntax.line_has_implicit_array(var)
                if arrayVar[0]:
                    var = arrayVar[1]
                    var_type = "Array"
                # Map each subroutine argument with its type
                if (
                    current_subr in module_summary[current_modu]
                    and var.strip()
                    in module_summary[current_modu][current_subr]
                ):
                    module_summary[current_modu][current_subr][
                        var.strip()
                    ] = var_type
    else:
        pass
    return current_subr


def extract_interface_info(
    pgm, end_pgm, procedure_functions, current_modu, current_intr, line
):
    """This function extracts INTERFACE information, such as the name of
    interface and procedure function names, and populates procedure_functions
    dictionary.

    Params:
        pgm (tuple): Current program information.
        end_pgm (typle): End of current program indicator.
        procedure_functions (dict): A dictionary to hold interface-to-procedure
        function mappings.
        current_modu (str): Module name that current interface is located under.
        current_intr (str): Current interface name.
        line (str): A line from Fortran source code.
    Returns:
        (current_intr) Currently handling interface name.
    """

    if pgm[0] and pgm[1] == "interface":
        current_intr = pgm[2]
        procedure_functions[current_modu][current_intr] = {}
    elif end_pgm[0] and end_pgm[1] == "interface":
        current_intr = None
    elif current_intr:
        if "procedure" in line:
            # Partition the string, which should have one of syntaxes like:
            #   module procedure __function_name__
            #   module procedure , __function_name__
            #   module procedure __function_name__ , __function_name__ , ...
            # by keyword procedure. Then, only extract function names, which
            # always will be located at [-1] after partitioning. Finally, split
            # the string of function names by comma and store in the functions list.
            functions = line.partition("procedure")[-1].split(",")
            for func in functions:
                func = func.strip()
                procedure_functions[current_modu][current_intr][func] = None
    else:
        pass

    return current_intr


def extract_derived_type_info(end_pgm, current_modu, derived_types):
    """This function extracts derived types declared under current module.

    Params:
        end_pgm (tuple): End of current program indicator.
        current_modu (str): Current module name.
        derived_types (dict): Dictionary that will hold module-to-derived type
        mapping.
    Returns:
        None.
    """

    if end_pgm[0] and end_pgm[1] == "type":
        if current_modu not in derived_types:
            derived_types[current_modu] = [end_pgm[2]]
        else:
            derived_types[current_modu].append(end_pgm[2])


def get_file_last_modified_time(file_path):
    """This function retrieves the file status and assigns the last modified
    time of a file at the end of the file_to_mod_mapper[file_path] list.

    Params:
        file_path (str): File path that is assumed to exist in the
            directory.
    Returns:
        int: Last modified time represented as an integer.
    """
    file_stat = os.stat(file_path)
    return file_stat[8]


def update_mod_info_json(module_log_file_path, mode_mapper_dict):
    """This function updates each module's information, such as
    the declared variables and their types, so that genPGM.py
    can simply reference this dictionary rather than processing
    the file again.

    Params:
        module_log_file_path (str): Path to module log file.
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
                mod_name in mode_mapper_dict["subprograms"]
                and sym in mode_mapper_dict["subprograms"][mod_name]
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

    with open(module_log_file_path, "w+") as json_f:
        json_f.write(json.dumps(module_logs, indent=2))


def mod_file_log_generator(
    root_dir_path=None, module_log_file_name=None,
):
    """This function is like a main function to invoke other functions
    to perform all checks and population of mappers. Though, loading of
    and writing to JSON file will happen in this function.

    Args:
        root_dir_path: Directory to examine for Fortran files
        module_log_file_name: Path to module log file.

    Returns:
        None.
    """
    if root_dir_path is None:
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
        modules_from_file(
            file_path, file_to_mod_mapper, mod_to_file_mapper, mod_info_dict
        )

    module_log = {
        "file_to_mod": file_to_mod_mapper,
        "mod_to_file": mod_to_file_mapper,
        "mod_info": mod_info_dict,
    }

    with open(module_log_file_path, "w+") as f:
        f.write(json.dumps(module_log, indent=2))

    return module_log_file_path


if __name__ == "__main__":
    root_dir_path, module_log_file = parse_args()
    log_file_path = mod_file_log_generator(root_dir_path, module_log_file)
