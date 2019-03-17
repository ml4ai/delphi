"""
Generates a module index file. This file describes each module used in a program run.
The information about each module is represented as a json dictionary and has the
following fields:

    name:              <module_name>
    file:              <file_containing_the_module>
    module:            <list_of_used_modules>
    symbol_export:     <list_of_symbols_exported_by_module>
    subprogram_list:   <procedure_mapping_for_module>

The procedure mapping for each subprogram p defined in module M is a mapping from
each possible tuple of argument types for p to the function to invoke for that argument
type tuple.

Author: Pratik Bhandari
Date:   02/19/2019
"""

import sys
import xml.etree.ElementTree as ET
from typing import List, Dict
import re
import json

class moduleGenerator(object):
    def __init__(self):

        self.asts = []
        self.useList = []
        self.functionList = []
        self.entryPoint = []
        self.current_context = ''
        self.fileName = ''
        self.main = ''
        self.path = ''

        # Initialize all the dictionaries which we will be writing to our file
        self.modules = []
        self.exports = {}
        self.uses = {}
        self.imports = {}
        self.private = {}
        self.public = {}
        self.subpgms = {}
        self.syms = {}

    def parseTree(self, root) -> bool:

        # Find function name
        for item in root.iter():
            if item.tag == "program":
                self.main = item.attrib["name"].lower()

        for item in root.iter():
            if item.tag == "file":
                file_name = item.attrib["path"]
                file = file_name.split('/')[-1]
                file_reg = r'^(.*)_processed(\..*)$'
                path_reg = r'^.*(delphi/[^delphi].*)/\w+'
                match = re.match(file_reg, file)
                if match:
                    self.fileName = match.group(1) + match.group(2)
                match = re.match(path_reg, file_name)
                if match:
                    self.path = match.group(1)


            elif item.tag.lower() in ["module", "program"]:
                self.current_context = item.attrib["name"].lower()
                self.modules.append(item.attrib["name"].lower())

            elif item.tag.lower() == "variable":
                if item.attrib.get("name"):
                    if self.current_context == "":
                        self.current_context = self.main
                    if self.public.get(self.current_context):
                        self.public[self.current_context].append(item.attrib["name"].lower())
                    else:
                        self.public[self.current_context] = [item.attrib["name"].lower()]

            elif item.tag.lower() in ["subroutine", "function"]:
                if self.current_context == "":
                    self.current_context = self.main
                if self.subpgms.get(self.current_context):
                    self.subpgms[self.current_context].append(item.attrib["name"].lower())
                else:
                    self.subpgms[self.current_context] = [item.attrib["name"].lower()]

            elif item.tag.lower() == "declaration":
                self.possible_private(item)

            elif item.tag.lower() == "use":
                hasOnly = False
                only_symbols = []
                for child in item:
                    if child.tag.lower() == "only":
                        hasOnly = True
                        for innerChild in child:
                            if innerChild.tag.lower() == "name" and hasOnly:
                                only_symbols.append(innerChild.attrib["id"].lower())

                if self.current_context == "":
                    self.current_context = self.main
                if self.uses.get(self.current_context):
                    if len(only_symbols) != 0:
                        self.uses[self.current_context].append({item.attrib["name"].lower(): only_symbols})
                    else:
                        self.uses[self.current_context].append({item.attrib["name"].lower(): ['*']})
                else:
                    if len(only_symbols) != 0:
                        self.uses[self.current_context] = [{item.attrib["name"].lower(): only_symbols}]
                    else:
                        self.uses[self.current_context] = [{item.attrib["name"].lower(): ['*']}]

        self.populate_syms()
        self.populate_exports()
        self.populate_imports()

        return True

    def populate_syms(self):
        for item in self.modules:
            self.syms[item] = self.public.get(item, []) + self.private.get(item, []) + self.subpgms.get(item, [])

    def populate_exports(self):
        for item in self.modules:
            interim = self.imports.get(item, []) + self.syms.get(item, [])
            self.exports[item] = [x for x in interim if x not in self.private.get(item, [])]

    def populate_imports(self):
        for module in self.uses:
            for use_item in self.uses[module]:
                for key in use_item:
                    if len(use_item[key]) == 1 and use_item[key][0] == '*':
                        if self.imports.get(module):
                            self.imports[module].append({key: self.exports[key]})
                        else:
                            self.imports[module] = [{key: self.exports[key]}]
                    elif len(use_item[key]) > 1:
                        if self.imports.get(module):
                            self.imports[module].append({key: use_item[key]})
                        else:
                            self.imports[module] = [{key: use_item[key]}]

    def possible_private(self, item):
        private_status = False
        for child in item.iter():
            if child.tag.lower() == "access-spec" and child.attrib.get("keyword").lower() == "private":
                private_status = True
            if child.tag.lower() == "name" and private_status:
                if self.current_context == "":
                    self.current_context = self.main
                if self.private.get(self.current_context):
                    self.private[self.current_context].append(child.attrib["id"].lower())
                else:
                    self.private[self.current_context] = [child.attrib["id"].lower()]

    def analyze(self, tree: ET.ElementTree) -> List:
        status = self.parseTree(tree)
        outputDict = {}
        if status:
            outputDict['FileName'] = [self.fileName, self.path]
            outputDict['Modules'] = self.modules
            outputDict['Exports'] = self.exports
            outputDict['Uses'] = self.uses
            outputDict['Imports'] = self.imports
            outputDict['Private'] = self.private
            outputDict['Public'] = self.public
            outputDict['Subpgms'] = self.subpgms
            outputDict['Syms'] = self.syms

            with open('mod_gen.json','w') as fd:
                json.dump(outputDict, fd, indent=4)

# Return the root of the XML ast (of the Fortran program) generated by the OFP so that subsequent recursion can
# be applied to it
def get_tree(file: str) -> ET.ElementTree:
    return ET.parse(file).getroot()


# Get the root of the XML ast, instantiate the moduleGenerator and start the analysis process.
def get_index(xml_file: str):
    tree = get_tree(xml_file)
    generator = moduleGenerator()
    status = generator.analyze(tree)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} filename\n")
        sys.exit(1)

    get_index(sys.argv[1])
