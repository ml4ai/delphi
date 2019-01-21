#!/usr/bin/python

"""
This script converts the XML version of AST of the Fortran
file into a JSON representation of the AST along with other
non-source code information. The output is a pickled file
which contains this information in a parsable data structure.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        python translate.py -f <ast_file> -g <pickle_file>

ast_file: The XML represenatation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

pickle_file: The file which will contain the pickled version of JSON AST and
supporting information.

"""


import sys
import argparse
import pickle
from collections import *
import xml.etree.ElementTree as ET
from delphi.translators.for2py.scripts.get_comments import (
    get_comments,
)
from typing import List, Dict
from collections import OrderedDict


class ParseState(object):
    """This class defines the state of the XML tree parsing
    at any given root. For any level of the tree, it stores
    the subroutine under which it resides along with the
    subroutines arguments."""

    def __init__(self, subroutine=None):
        self.subroutine = subroutine if subroutine is not None else {}
        self.args = (
            [arg["name"] for arg in self.subroutine["args"]]
            if "args" in self.subroutine
            else []
        )

    def copy(self, subroutine=None):
        return ParseState(
            self.subroutine if subroutine == None else subroutine
        )


class XMLToJSONTranslator(object):
    def __init__(self):
        self.libRtns = ["read", "open", "close", "format", "print", "write"]
        self.libFns = [
            "MOD",
            "EXP",
            "INDEX",
            "MIN",
            "MAX",
            "cexp",
            "cmplx",
            "ATAN",
        ]
        self.inputFns = ["read"]
        self.outputFns = ["write"]
        self.summaries = {}
        self.asts = {}
        self.functionList = []
        self.subroutineList = []
        self.entryPoint = []

    def process_subroutine_or_program(self, root, state):
        subroutine = {"tag": root.tag, "name": root.attrib["name"]}
        self.summaries[root.attrib["name"]] = None
        if root.tag == "subroutine":
            self.subroutineList.append(root.attrib["name"])
        else:
            self.entryPoint.append(root.attrib["name"])
        for node in root:
            if node.tag == "header":
                subroutine["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, subState)
        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_call(self, root, state) -> List[Dict]:
        call = {"tag": "call"}
        for node in root:
            if node.tag == "name":
                call["name"] = node.attrib["id"]
                call["args"] = []
                for arg in node:
                    call["args"] += self.parseTree(arg, state)
        return [call]

    def process_argument(self, root, state) -> List[Dict]:
        return [{"tag": "arg", "name": root.attrib["name"]}]

    def process_declaration(self, root, state) -> List[Dict]:
        decVars = []
        decDims = []
        decType = {}
        count = 0
        for node in root:
            if node.tag == "type":
                decType = {"type": node.attrib["name"]}
            elif node.tag == "variables":
                decVars = self.parseTree(node, state)
            elif node.tag == "dimensions":
                decDims = self.parseTree(node, state)
                count = node.attrib["count"]
        print ("decDims: ", decDims)
        print ("count: ", count)
        prog = []
        for var in decVars:
            if (
                state.subroutine["name"] in self.functionList
                and var["name"] in state.args
            ):
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
                continue
            for i in range (0, int(count)):
                prog.append(decType.copy())
                prog[-1].update(var)
            if var["name"] in state.args:
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
        if decDims:
            counter = 0
            for dim in decDims:
                print ("dim: ", dim)
                for lit in dim["literal"]:
                    prog[0]["tag"] = "array"
                    prog[0]["count"] = count
                    prog[0]["value" + str(counter+1)] = lit["value"]
                counter = counter + 1
        if len(prog) > 1:
            for i in range (0, int(count)):
                print ("prog: ", prog[i])
        return prog

    def process_variable(self, root, state) -> List[Dict]:
        try:
            return [{"tag": "variable", "name": root.attrib["name"]}]
        except:
            return []

    def process_do_loop(self, root, state) -> List[Dict]:
        do = {"tag": "do"}
        for node in root:
            if node.tag == "header":
                do["header"] = self.parseTree(node, state)
            elif node.tag == "body":
                do["body"] = self.parseTree(node, state)
        return [do]

    def process_index_variable(self, root, state) -> List[Dict]:
        ind = {"tag": "index", "name": root.attrib["name"]}
        for bounds in root:
            if bounds.tag == "lower-bound":
                ind["low"] = self.parseTree(bounds, state)
            elif bounds.tag == "upper-bound":
                ind["high"] = self.parseTree(bounds, state)
        return [ind]

    def process_if(self, root, state) -> List[Dict]:
        ifs = []
        curIf = None
        for node in root:
            if node.tag == "header":
                if "type" not in node.attrib:
                    curIf = {"tag": "if"}
                    curIf["header"] = self.parseTree(node, state)
                    ifs.append(curIf)
                elif node.attrib["type"] == "else-if":
                    newIf = {"tag": "if"}
                    curIf["else"] = [newIf]
                    curIf = newIf
                    curIf["header"] = self.parseTree(node, state)
            elif node.tag == "body" and (
                "type" not in node.attrib or node.attrib["type"] != "else"
            ):
                curIf["body"] = self.parseTree(node, state)
            elif node.tag == "body" and node.attrib["type"] == "else":
                curIf["else"] = self.parseTree(node, state)
        return ifs

    def process_operation(self, root, state) -> List[Dict]:
        op = {"tag": "op"}
        for node in root:
            if node.tag == "operand":
                if "left" in op:
                    op["right"] = self.parseTree(node, state)
                else:
                    op["left"] = self.parseTree(node, state)
            elif node.tag == "operator":
                if "operator" in op:
                    newOp = {
                        "tag": "op",
                        "operator": node.attrib["operator"],
                        "left": [op],
                    }
                    op = newOp
                else:
                    op["operator"] = node.attrib["operator"]
        return [op]

    def process_literal(self, root, state) -> List[Dict]:
        for info in root:
            if info.tag == "pause-stmt":
                return [{"tag": "pause", "msg": root.attrib["value"]}]
        return [
            {
                "tag": "literal",
                "type": root.attrib["type"],
                "value": root.attrib["value"],
            }
        ]

    def process_stop(self, root, state) -> List[Dict]:
        return [{"tag": "stop"}]

    def process_name(self, root, state) -> List[Dict]:
        if root.attrib["id"] in self.libFns:
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        elif (
            root.attrib["id"] in self.functionList
            and state.subroutine["tag"] != "function"
        ):
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        else:
            ref = {"tag": "ref", "name": root.attrib["id"]}
            subscripts = []
            for node in root:
                subscripts += self.parseTree(node, state)
            if subscripts:
                ref["subscripts"] = subscripts
            return [ref]

    def process_assignment(self, root, state) -> List[Dict]:
        assign = {"tag": "assignment"}
        for node in root:
            if node.tag == "target":
                assign["target"] = self.parseTree(node, state)
            elif node.tag == "value":
                assign["value"] = self.parseTree(node, state)
        if (assign["target"][0]["name"] in self.functionList) and (
            assign["target"][0]["name"] == state.subroutine["name"]
        ):
            assign["value"][0]["tag"] = "ret"
            return assign["value"]
        else:
            return [assign]

    def process_function(self, root, state) -> List[Dict]:
        subroutine = {"tag": root.tag, "name": root.attrib["name"]}
        self.summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                subroutine["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, subState)
        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_exit(self, root, state) -> List[Dict]:
        return [{"tag": "exit"}]

    def process_return(self, root, state) -> List[Dict]:
        ret = {"tag": "return"}
        return [ret]

    def process_dimension(self, root, state) -> List[Dict]:
        dimension = {"tag": "dimension"}
        for node in root:
            if node.tag == "literal":
                dimension["literal"] = self.parseTree(node, state)
        return [dimension]

    def process_libRtn(self, root, state) -> List[Dict]:
        fn = {"tag": "call", "name": root.tag, "args": []}
        for node in root:
            fn["args"] += self.parseTree(node, state)
        return [fn]

    def parseTree(self, root, state: ParseState) -> List[Dict]:
        """
        Parses the XML ast tree recursively to generate a JSON AST
        which can be ingested by other scripts to generate Python
        scripts.

        Args:
            root: The current root of the tree.
            state: The current state of the tree defined by an object of the
                ParseState class.

        Returns:
                ast: A JSON ast that defines the structure of the Fortran file.
        """

        if root.tag in ("subroutine", "program"):
            return self.process_subroutine_or_program(root, state)

        elif root.tag == "call":
            return self.process_call(root, state)

        elif root.tag == "argument":
            return self.process_argument(root, state)

        elif root.tag == "declaration":
            return self.process_declaration(root, state)

        elif root.tag == "variable":
            return self.process_variable(root, state)

        elif root.tag == "loop" and root.attrib["type"] == "do":
            return self.process_do_loop(root, state)

        elif root.tag == "index-variable":
            return self.process_index_variable(root, state)

        elif root.tag == "if":
            return self.process_if(root, state)

        elif root.tag == "operation":
            return self.process_operation(root, state)

        elif root.tag == "literal":
            return self.process_literal(root, state)

        elif root.tag == "stop":
            return self.process_stop(root, state)

        elif root.tag == "name":
            return self.process_name(root, state)

        elif root.tag == "assignment":
            return self.process_assignment(root, state)

        elif root.tag == "function":
            return self.process_function(root, state)

        elif root.tag == "exit":
            return self.process_exit(root, state)

        elif root.tag == "return":
            return self.process_return(root, state)

        elif root.tag == "dimension":
            return self.process_dimension(root, state)

        elif root.tag in self.libRtns:
            return self.process_libRtn(root, state)

        else:
            prog = []
            for node in root:
                prog += self.parseTree(node, state)
            return prog

    def loadFunction(self, root):
        """
        Loads a list with all the functions in the Fortran File

        Args:
            root: The root of the XML ast tree.

        Returns:
            None

        Does not return anything but populates a list (self.functionList) that contains all
        the functions in the Fortran File.
        """
        for element in root.iter():
            if element.tag == "function":
                self.functionList.append(element.attrib["name"])

    def analyze(
        self, trees: List[ET.ElementTree], comments: OrderedDict
    ) -> Dict:
        outputDict = {}
        ast = []

        # Parse through the ast once to identify and grab all the funcstions
        # present in the Fortran file.
        for tree in trees:
            self.loadFunction(tree)

        # Parse through the ast tree a second time to convert the XML ast format to
        # a format that can be used to generate python statements.
        for tree in trees:
            ast += self.parseTree(tree, ParseState())

        """

        Find the entry point for the Fortran file.
        The entry point for a conventional Fortran file is always the PROGRAM section.
        This 'if' statement checks for the presence of a PROGRAM segment.

        If not found, the entry point can be any of the functions or subroutines
        in the file. So, all the functions and subroutines of the program are listed
        and included as the possible entry point.

        """
        if self.entryPoint:
            entry = {"program": self.entryPoint[0]}
        else:
            entry = {}
            if self.functionList:
                entry["function"] = self.functionList
            if self.subroutineList:
                entry["subroutine"] = self.subroutineList

        # Load the functions list and Fortran ast to a single data structure which
        # can be pickled and hence is portable across various scripts and usages.
        outputDict["ast"] = ast
        outputDict["functionList"] = self.functionList
        outputDict["comments"] = comments
        return outputDict


def get_trees(files: List[str]) -> List[ET.ElementTree]:
    return [ET.parse(f).getroot() for f in files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        nargs="*",
        help="Pickled version of routines for which dependency graphs should be generated",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="A list of AST files in XML format to analyze",
    )
    parser.add_argument(
        "-i", "--input", nargs="*", help="Original Fortran Source code file."
    )

    args = parser.parse_args(sys.argv[1:])
    fortranFile = args.input[0]
    pickleFile = args.gen[0]

    trees = get_trees(args.files)
    comments = get_comments(fortranFile)
    translator = XMLToJSONTranslator()
    outputDict = translator.analyze(trees, comments)

    with open(pickleFile, "wb") as f:
        pickle.dump(outputDict, f)
