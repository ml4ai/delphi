#!/usr/bin/python

"""

   File:    translate.py
   
   Purpose: This script converts the XML version of AST of the Fortran 
            file into a JSON representation of the AST along with other
            non-source code information. The output is a pickled file 
            which contains this information in a parsable data structure.

   Usage:   This script is executed by the autoTranslate script as one 
            of the steps in converted a Fortran source file to Python 
            file. For standalone execution:
              python translate.py -f <ast_file> -g <pickle_file>

            ast_file: The XML represenatation of the AST of the Fortran
                      file. This is produced by the OpenFortranParser.

            pickle_file: The file which will contain the pickled version
                          of JSON AST and supporting information.

"""


import xml.etree.ElementTree as ET
import sys
import argparse
import pickle

libRtns = ["read", "open", "close", "format", "print", "write"]
libFns = ["MOD", "EXP", "INDEX", "MIN", "MAX", "cexp", "cmplx", "ATAN"]
inputFns = ["read"]
outputFns = ["write"]
summaries = {}
asts = {}
functionList = []

cleanup = True


class ParseState:
    """This class defines the state of the XML tree parsing
    at any given root. For any level of the tree, it stores
    the subroutine under which it resides along with the
    subroutines arguments."""

    def __init__(self, subroutine=None):
        self.subroutine = subroutine if subroutine != None else {}
        self.args = (
            [arg["name"] for arg in self.subroutine["args"]]
            if "args" in self.subroutine
            else []
        )

    def copy(self, subroutine=None):
        return ParseState(
            self.subroutine if subroutine == None else subroutine
        )


def loadFunction(root):
    """
    Loads a list with all the functions in the Fortran File

    Args:
        root: The root of the XML ast tree.

    Returns:
        None

    Does not return anything but populates a list (functionList) that contains all
    the functions in the Fortran File.
    """
    for element in root.iter():
        if element.tag == "function":
            functionList.append(element.attrib["name"])


def parseTree(root, state):
    """
    Parses the XML ast tree recursively to generate a JSON ast
    which can be ingested by other scripts to generate Python
    scripts.

    Args:
        root: The current root of the tree.
        state: The current state of the tree defined by an object of the
            ParseState class.

    Returns:
            ast: A JSON ast that defines the structure of the Fortran file.
    """

    if root.tag == "subroutine" or root.tag == "program":
        subroutine = {"tag": root.tag, "name": root.attrib["name"]}
        summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                subroutine["args"] = parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = parseTree(node, subState)
        asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    elif root.tag == "call":
        call = {"tag": "call"}
        for node in root:
            if node.tag == "name":
                call["name"] = node.attrib["id"]
                call["args"] = []
                for arg in node:
                    call["args"] += parseTree(arg, state)
        return [call]

    elif root.tag == "argument":
        return [{"tag": "arg", "name": root.attrib["name"]}]

    # elif root.tag == "name":
    # return [{"tag":"arg", "name":root.attrib["id"]}]

    elif root.tag == "declaration":
        decVars = []
        decType = {}
        for node in root:
            if node.tag == "type":
                decType = {"type": node.attrib["name"]}
            elif node.tag == "variables":
                decVars = parseTree(node, state)
        prog = []
        for var in decVars:
            if (
                state.subroutine["name"] in functionList
                and var["name"] in state.args
            ):
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
                continue
            prog.append(decType.copy())
            prog[-1].update(var)
            if var["name"] in state.args:
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
        return prog

    elif root.tag == "variable":
        try:
            return [{"tag": "variable", "name": root.attrib["name"]}]
        except:
            return []

    elif root.tag == "loop" and root.attrib["type"] == "do":
        do = {"tag": "do"}
        for node in root:
            if node.tag == "header":
                do["header"] = parseTree(node, state)
            elif node.tag == "body":
                do["body"] = parseTree(node, state)
        return [do]

    elif root.tag == "index-variable":
        ind = {"tag": "index", "name": root.attrib["name"]}
        for bounds in root:
            if bounds.tag == "lower-bound":
                ind["low"] = parseTree(bounds, state)
            elif bounds.tag == "upper-bound":
                ind["high"] = parseTree(bounds, state)
        return [ind]

    elif root.tag == "if":
        ifs = []
        curIf = None
        for node in root:
            if node.tag == "header" and "type" not in node.attrib:
                curIf = {"tag": "if"}
                curIf["header"] = parseTree(node, state)
                ifs.append(curIf)
            elif node.tag == "header" and node.attrib["type"] == "else-if":
                newIf = {"tag": "if"}
                curIf["else"] = [newIf]
                curIf = newIf
                curIf["header"] = parseTree(node, state)
            elif node.tag == "body" and (
                "type" not in node.attrib or node.attrib["type"] != "else"
            ):
                curIf["body"] = parseTree(node, state)
            elif node.tag == "body" and node.attrib["type"] == "else":
                curIf["else"] = parseTree(node, state)
        return ifs

    elif root.tag == "operation":
        op = {"tag": "op"}
        for node in root:
            if node.tag == "operand":
                if "left" in op:
                    op["right"] = parseTree(node, state)
                else:
                    op["left"] = parseTree(node, state)
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

    elif root.tag == "literal":
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

    elif root.tag == "stop":
        return [{"tag": "stop"}]

    elif root.tag == "name":
        if root.attrib["id"] in libFns:
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += parseTree(node, state)
            return [fn]
        elif (
            root.attrib["id"] in functionList
            and state.subroutine["tag"] != "function"
        ):
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += parseTree(node, state)
            return [fn]
        # elif root.attrib["id"] in functionList and state.subroutine["tag"] == "function":
        #    fn = {"tag": "return", "name": root.attrib["id"]
        else:
            ref = {"tag": "ref", "name": root.attrib["id"]}
            subscripts = []
            for node in root:
                subscripts += parseTree(node, state)
            if subscripts:
                ref["subscripts"] = subscripts
            return [ref]

    elif root.tag == "assignment":
        assign = {"tag": "assignment"}
        for node in root:
            if node.tag == "target":
                assign["target"] = parseTree(node, state)
            elif node.tag == "value":
                assign["value"] = parseTree(node, state)
            # if assign["target"][0]["name"] in functionList:
            #    assign["value"][0]["tag"] = "ret"
        if (assign["target"][0]["name"] in functionList) and (
            assign["target"][0]["name"] == state.subroutine["name"]
        ):
            assign["value"][0]["tag"] = "ret"
            return assign["value"]
        else:
            return [assign]

    elif root.tag == "function":
        subroutine = {"tag": root.tag, "name": root.attrib["name"]}
        # functionList.append(root.attrib["name"])
        summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                subroutine["args"] = parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = parseTree(node, subState)
        asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    elif root.tag == "exit":
        return [{"tag": "exit"}]

    elif root.tag == "return":
        ret = {"tag": "return"}
        return [ret]

    elif root.tag in libRtns:
        fn = {"tag": "call", "name": root.tag, "args": []}
        for node in root:
            fn["args"] += parseTree(node, state)
        return [fn]

    else:
        prog = []
        for node in root:
            prog += parseTree(node, state)
        return prog


def printAstTree(astFile, tree, blockVal):
    parentVal = blockVal
    for node in tree:
        if parentVal != blockVal:
            astFile.write(
                "\tB" + str(parentVal) + " -> B" + str(blockVal) + "\n"
            )
            parentVal = blockVal
        block = "\tB" + str(blockVal) + ' [label="'
        blockVal += 1
        for key in node:
            if not isinstance(node[key], list):
                block += "'" + str(key) + "=" + str(node[key]) + "' "
        block += '"]\n'
        astFile.write(block)
        for key in node:
            if isinstance(node[key], list) and bool(node[key]):
                astFile.write(
                    "\tB"
                    + str(parentVal)
                    + " -> B"
                    + str(blockVal)
                    + ' [label="'
                    + str(key)
                    + '"]\n'
                )
                blockVal = printAstTree(astFile, node[key], blockVal)

    return blockVal


def analyze(files, pickleFile):
    global cleanup
    outputFiles = {}
    ast = []

    # Parse through the ast tree once to identify and grab all the funcstions
    # present in the Fortran file.
    for f in files:
        tree = ET.parse(f)
        loadFunction(tree)

    # Parse through the ast tree a second time to convert the XML ast format to
    # a format that can be used to generate python statements.
    for f in files:
        tree = ET.parse(f)
        ast += parseTree(tree.getroot(), ParseState())

    # Load the functions list and Fortran ast to a single data structure which
    # can be pickled and hence is portable across various scripts and usages.
    outputFiles["ast"] = ast
    outputFiles["functionList"] = functionList

    with open(pickleFile, "wb") as f:
        pickle.dump(outputFiles, f)


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
    args = parser.parse_args(sys.argv[1:])
    analyze(args.files, args.gen[0])
