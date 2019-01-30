#!/usr/bin/python

'''
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
'''

import sys
import xml.etree.ElementTree as ET
from typing import List, Dict
import re

class ParseState(object):
    """This class defines the state of the XML tree parsing
    at any given root. For any level of the tree, it stores
    the module under which it resides along with the
    module metadata."""

    def __init__(self, module=None):
        self.module = module if module is not None else {}
        self.args = (
            [arg["name"] for arg in self.module["args"]]
            if "args" in self.module
            else []
        )

    def copy(self, module=None):
        return ParseState(
            self.module if module == None else module
        )

class moduleGenerator(object):
    def __init__(self):
        self.asts = []
        self.useList = []
        self.functionList = []
        self.entryPoint = []
        self.fileName = str()


    def parseTree(self, root, state: ParseState) -> bool:

        if root.tag == "module":
            return self.process_module(root, state)

        elif root.tag == "use":
            return self.process_use(root, state)

        elif root.tag == "file":
            return self.process_file(root, state)

        elif root.tag in ["program", "subroutine"]:
            return self.process_subroutine_or_program(root, state)

        else:
            status = True
            for node in root:
                status = self.parseTree(node, state)
            return status

    def process_subroutine_or_program(self, root, state):
        # Initialize the list of used modules to be an empty list for each module
        self.useList = []

        for node in root:
            status = self.parseTree(node, state)

    def process_module(self, root, state):
        module = dict()
        module["name"] = root.attrib["name"]
        module["source_file"] = self.fileName

        # Initialize the list of used modules to be an empty list for each module
        self.useList = []

        for node in root:
            if node.tag == "header":
                continue
            elif node.tag == "body":
                sub_state = state.copy(module)
                status = self.parseTree(node, sub_state)

        # At this point, the module tag has been recursively iterated to find all use statements
        module["uses"] = self.useList

        print(module)
        # Append the module metadata to our final ast
        self.asts.append(module)

        return status

    def process_use(self, root, state):
        self.useList.append(root.attrib["name"])
        return True

    def process_file(self, root, state):
        file_name = root.attrib["path"]
        file_name = file_name.split('/')[-1]
        file_reg = r'^(.*)_processed(\..*)$'
        match = re.match(file_reg, file_name)
        if match:
            self.fileName = match.group(1) + match.group(2)
        for node in root:
            self.parseTree(node, state)

        return True



    def analyze(self, tree: ET.ElementTree) -> List:

        status = self.parseTree(tree, ParseState())

        return self.asts

def get_tree(file: str) -> ET.ElementTree:
    return ET.parse(file).getroot()

def get_index(xml_file: str):
    tree = get_tree(xml_file)
    generator = moduleGenerator()
    output_dict = generator.analyze(tree)
    print(output_dict)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} filename\n")
        sys.exit(1)

    get_index(sys.argv[1])



