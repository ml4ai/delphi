"""
This module contains code to generate a module index file for a set of Fortran
files that the program analysis pipeline runs over. The file describes each
module used in a program run. The information about each module is represented
as a JSON dictionary and has the following fields:

    name:              <module_name>
    file:              <file_containing_the_module>
    module:            <list_of_used_modules>
    symbol_export:     <list_of_symbols_exported_by_module>
    subprogram_list:   <procedure_mapping_for_module>

The procedure mapping for each subprogram `p` defined in module `M` is a
mapping from each possible tuple of argument types for p to the function to
invoke for that argument type tuple.

Author: Pratik Bhandari
"""

import xml.etree.ElementTree as ET
from typing import List
import re
import json


class ModuleGenerator(object):
    def __init__(self):
        # This string holds the current context of the program being parsed
        self.current_context = None
        # This string holds the name of the original Fortran code which is
        # being processed
        self.fileName = None
        # This string holds the name of the main PROGRAM module
        self.main = None
        # This string holds the path on which the XML file of the original
        # Fortran code is located
        self.path = None
        # Initialize all the dictionaries which we will be writing to our file
        # This is a list of all modules inside a single Fortran file
        self.modules = []
        # This dictionary holds the set of symbols exported by each module.
        self.exports = {}
        # This dictionary holds the modules used by each module/program.
        # Additionally, variables from each module can be selectively USEd (
        # imported). This is stored in the object below. If all variables of
        # a module are USEd, an `*` symbol is used to denote this.
        self.uses = {}
        # This dictionary holds the set of symbols imported by each module.
        # This is given by: IMPORTS(m) = U { EXPORTS(p) | p ∈ USES(m) }
        # Since a module can use the ONLY keyword to import only some of the
        # symbols exported by another module, in such cases, the set of
        # imported symbols is be limited to those explicitly mentioned.
        self.imports = {}
        # This dictionary holds all the private variables defined in each
        # module
        self.private = {}
        # This dictionary holds all public variables for each module/context
        self.public = {}
        # This dictionary holds all subprograms (subroutines and functions) for
        # each module
        self.subprograms = {}
        # This dictionary stores the set of symbols declared in each module.
        self.symbols = {}
        # This dictionary stores a variable-type mapping.
        self.variable_types = {}
        self.symbol_type = {}

    def populate_symbols(self):
        """ This function populates the dictionary `self.symbols` which stores
        the set of symbols declared in each module. This is the union of all
        public variables, private variables and subprograms for each module.
        """
        for item in self.modules:
            self.symbols[item] = self.public.get(item, []) + \
                                 self.private.get(item, []) + \
                                 self.subprograms.get(item, [])

    def populate_exports(self):
        """ This function populates the `self.exports` dictionary which holds
        the set of symbols exported by each module. The set of exported symbols
        is given by: (imports U symbols) - private """
        for item in self.modules:
            interim = self.imports.get(item, []) + self.symbols.get(item, [])
            self.exports[item] = [x for x in interim if x not in
                                  self.private.get(item, [])]

    def populate_imports(self, module_logs):
        """ This function populates the `self.imports` dictionary which holds
        all the private variables defined in each module."""
        for module in self.uses:
            for use_item in self.uses[module]:
                for key in use_item:
                    if len(use_item[key]) == 1 and use_item[key][0] == '*':
                        if key in self.exports:
                            symbols = self.exports[key]
                        else:
                            assert (
                                key.lower() in module_logs["mod_info"]
                            ), f"module name (key) {key} does not exist in " \
                               f"the log file."
                            symbols = module_logs["mod_info"][key]["exports"]
                            if module in module_logs["mod_info"]:
                                module_logs["mod_info"][module]["imports"] = \
                                    symbols
                        if key in symbols:
                            self.imports.setdefault(module, []).append(
                                {key: symbols[key]}
                            )
                        else:
                            self.imports.setdefault(module, []).append(
                                {key: symbols}
                            )
                    else:
                        self.imports.setdefault(module, []).append(
                            {key: use_item[key]}
                        )

    def populate_symbol_types(self):
        for var in self.variable_types:
            for module in self.symbols:
                if var in self.symbols[module]:
                    self.symbol_type[var] = [module, self.variable_types[var]]

    def parse_tree(self, root, module_logs) -> bool:
        """ This function parses the XML tree of a Fortran file and tracks and
        maps relevant object relationships """

        # Find name of PROGRAM module
        for item in root.iter():
            if item.tag == "program":
                self.main = item.attrib["name"].lower()

        variable_type = None

        for item in root.iter():
            # Get the name of the XML file being parsed
            if item.tag == "file":
                file_name = item.attrib["path"]
                file = file_name.split('/')[-1]
                file_regex = r'^(.*)_processed(\..*)$'
                path_regex = r'^.*(delphi/[^delphi].*)/\w+'
                match = re.match(file_regex, file)
                if match:
                    self.fileName = match.group(1) + match.group(2)
                match = re.match(path_regex, file_name)
                if match:
                    self.path = match.group(1)

            elif item.tag.lower() in ["module", "program"]:
                self.current_context = item.attrib["name"].lower()
                self.modules.append(item.attrib["name"].lower())

            elif (
                    item.tag.lower() == "type"
                    and "name" in item.attrib
            ):
                variable_type = item.attrib["name"].lower()
            elif item.tag.lower() == "variable":
                if item.attrib.get("name"):
                    if not self.current_context:
                        self.current_context = self.main
                    self.public.setdefault(self.current_context, []).append(
                        item.attrib["name"].lower())
                    self.variable_types[item.attrib["name"].lower()] = \
                        variable_type

            elif item.tag.lower() in ["subroutine", "function"]:
                if not self.current_context:
                    self.current_context = self.main
                self.subprograms.setdefault(self.current_context, []).append(
                    item.attrib["name"].lower())
                self.current_context = item.attrib["name"].lower()

            elif item.tag.lower() == "declaration":
                # This function parses the <declaration> tag of the XML and
                # checks if private variables are defined in the respective
                # module/program. Private variables tend to be inside the
                # declaration tag
                private_status = False
                for child in item.iter():
                    if child.tag.lower() == "access-spec" and child.attrib.get(
                            "keyword").lower() == "private":
                        private_status = True
                    if child.tag.lower() == "variable" and private_status:
                        if not self.current_context:
                            self.current_context = self.main
                        if child.attrib.get("name"):
                            self.private.setdefault(self.current_context,
                                                    []).append(
                                child.attrib["name"].lower())

            elif item.tag.lower() == "use":
                # If a module, program or subroutine uses (imports) another
                # module, a USE statement is used and we want to map the
                # relationship between different program scopes that occur
                # with the use of the USE statement
                only_symbols = []
                for child in item:
                    if child.tag.lower() == "only":
                        for innerChild in child:
                            if innerChild.tag.lower() == "name":
                                only_symbols.append(innerChild.attrib[
                                                        "id"].lower())
                if not self.current_context:
                    self.current_context = self.main
                self.uses.setdefault(self.current_context, []).append({
                    item.attrib["name"].lower(): only_symbols} if only_symbols
                    else {item.attrib["name"].lower(): ["*"]})

        self.populate_symbols()
        self.populate_symbol_types()
        self.populate_exports()
        self.populate_imports(module_logs)

        return True

    def analyze(self, tree: ET.ElementTree, mod_log_path: str) -> List:
        """ Parse the XML file from the root and keep track of all important
        data structures and object relationships between files. """
        with open(mod_log_path) as json_f:
            module_logs = json.load(json_f)

        status = self.parse_tree(tree, module_logs)
        output_dictionary = {}
        if status:
            output_dictionary['file_name'] = [self.fileName, self.path]
            output_dictionary['modules'] = self.modules
            output_dictionary['exports'] = self.exports
            output_dictionary['use_mapping'] = self.uses
            output_dictionary['imports'] = self.imports
            output_dictionary['private_objects'] = self.private
            output_dictionary['public_objects'] = self.public
            output_dictionary['subprograms'] = self.subprograms
            output_dictionary['symbols'] = self.symbols
            output_dictionary['symbol_types'] = self.symbol_type

        with open(mod_log_path, 'w+') as json_f:
            json_f.write(json.dumps(module_logs, indent=2))
        return [output_dictionary]


def get_index(xml_file: str, module_log_file_path: str):
    """ Get the root of the XML ast, instantiate the moduleGenerator and start
    the analysis process.
    """
    tree = ET.parse(xml_file).getroot()
    generator = ModuleGenerator()
    return generator.analyze(tree, module_log_file_path)
