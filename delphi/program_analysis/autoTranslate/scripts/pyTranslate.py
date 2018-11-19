#!/usr/bin/python

"""

   File:    pyTranslate.py

   Purpose: Convert a Fortran AST representation into a Python
            script having the same functionalities and performing
            the same operations as the original Fortran file.

   Usage:   This script is executed by the autoTranslate script as one
            of the steps in converted a Fortran source file to Python
            file. For standalone execution:
               python pyTranslate -f <pickle_file> -g <python_file>

            pickle_file: Pickled file containing the ast represenatation
                         of the Fortran file along with other non-source
                         code information.

            python_file: The Python file on which to write the resulting
                         python script.

"""

import sys
import pickle
import argparse

GETFRAME_EXPR = "sys._getframe({}).f_code.co_name"

PRINTFN = {}
LIBFNS = ["MOD", "EXP", "INDEX", "MIN", "MAX", "cexp", "cmplx", "ATAN"]
MATHFUNC = ["mod", "exp", "cexp", "cmplx"]


class PrintState:
    def __init__(
        self,
        sep=None,
        add=None,
        printFirst=True,
        definedVars=None,
        globalVars=None,
        indexRef=True,
        varTypes=None,
    ):
        self.sep = sep if sep != None else "\n"
        self.add = add if add != None else "    "
        self.printFirst = printFirst
        self.definedVars = definedVars if definedVars != None else []
        self.globalVars = globalVars if globalVars != None else []
        self.indexRef = indexRef
        self.varTypes = varTypes if varTypes != None else {}

    def copy(
        self,
        sep=None,
        add=None,
        printFirst=None,
        definedVars=None,
        globalVars=None,
        indexRef=None,
        varTypes=None,
    ):
        return PrintState(
            self.sep if sep == None else sep,
            self.add if add == None else add,
            self.printFirst if printFirst == None else printFirst,
            self.definedVars if definedVars == None else definedVars,
            self.globalVars if globalVars == None else globalVars,
            self.indexRef if indexRef == None else indexRef,
            self.varTypes if varTypes == None else varTypes,
        )


def printSubroutine(pyStrings, node, printState):
    pyStrings.append(f"\ndef {node['name']}(")
    args = []
    printAst(
        pyStrings,
        node["args"],
        printState.copy(
            sep=", ",
            add="",
            printFirst=False,
            definedVars=args,
            indexRef=False,
        ),
    )
    pyStrings.append("):")
    printAst(
        pyStrings,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add,
            printFirst=True,
            definedVars=args,
            indexRef=True,
        ),
    )


def printFunction(pyStrings, node, printState):
    pyStrings.append(f"\ndef {node['name']}(")
    args = []
    printAst(
        pyStrings,
        node["args"],
        printState.copy(
            sep=", ",
            add="",
            printFirst=False,
            definedVars=args,
            indexRef=False,
        ),
    )
    pyStrings.append("):")
    printAst(
        pyStrings,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add,
            printFirst=True,
            definedVars=args,
            indexRef=True,
        ),
    )


def printProgram(pyStrings, node, printState):
    printSubroutine(pyStrings, node, printState)
    pyStrings.append(f"\n\n{node['name']}(){printState.sep}")


def printCall(pyStrings, node, printState):
    if not printState.indexRef:
        pyStrings.append("[")

    inRef = False

    if node["name"] in LIBFNS:
        node["name"] = node["name"].lower()
        if node["name"] in MATHFUNC:
            node["name"] = "math." + node["name"]
        inRef = 1

    pyStrings.append(f"{node['name']}(")
    printAst(
        pyStrings,
        node["args"],
        printState.copy(
            sep=", ", add="", printFirst=False, definedVars=[], indexRef=inRef
        ),
    )
    pyStrings.append(")")

    if not printState.indexRef:
        pyStrings.append("]")


def printArg(pyStrings, node, printState):
    if node["type"] == "INTEGER":
        varType = "int"
    elif node["type"] in ["DOUBLE", "REAL"]:
        varType = "float"
    else:
        print(f"unrecognized type {node['type']}")
        sys.exit(1)
    pyStrings.append(f"{node['name']}: List[{varType}]")
    printState.definedVars += [node["name"]]


def printVariable(pyStrings, node, printState):
    if (
        node["name"] not in printState.definedVars
        and node["name"] not in printState.globalVars
    ):
        printState.definedVars += [node["name"]]
        if node["type"] == "INTEGER":
            initVal = 0
            varType = "int"
        elif node["type"] in ["DOUBLE", "REAL"]:
            initVal = 0.0
            varType = "float"
        else:
            print(f"unrecognized type {node['type']}")
            sys.exit(1)
        pyStrings.append(f"{node['name']}: List[{varType}] = [{initVal}]")
    else:
        printState.printFirst = False


def printDo(pyStrings, node, printState):
    pyStrings.append("for ")
    printAst(
        pyStrings,
        node["header"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyStrings.append(":")
    printAst(
        pyStrings,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add, printFirst=True, indexRef=True
        ),
    )


def printIndex(pyStrings, node, printState):
    # pyStrings.append("{0} in range({1}, {2}+1)".format(node['name'], node['low'], node['high'])) Don't use this
    # pyStrings.append(f"{node['name']}[0] in range(") Use this instead
    pyStrings.append(f"{node['name']}[0] in range(")
    printAst(
        pyStrings,
        node["low"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyStrings.append(", ")
    printAst(
        pyStrings,
        node["high"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyStrings.append("+1)")


def printIf(pyStrings, node, printState):
    pyStrings.append("if ")
    printAst(
        pyStrings,
        node["header"],
        printState.copy(sep="", add="", printFirst=True, indexRef=True),
    )
    pyStrings.append(":")
    printAst(
        pyStrings,
        node["body"],
        printState.copy(
            sep=printState.sep + printState.add, printFirst=True, indexRef=True
        ),
    )
    if "else" in node:
        pyStrings.append(printState.sep)
        pyStrings.append("else:")
        printAst(
            pyStrings,
            node["else"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                indexRef=True,
            ),
        )


def printOp(pyStrings, node, printState):
    if not printState.indexRef:
        pyStrings.append("[")
    if "right" in node:
        pyStrings.append("(")
        printAst(
            pyStrings,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )

        operator_mapping = {
            ".ne.": " != ",
            ".gt.": " > ",
            ".eq.": " == ",
            ".lt.": " < ",
            ".le.": " <= ",
        }
        pyStrings.append(
            operator_mapping.get(
                node["operator"].lower(), f" {node['operator']} "
            )
        )
        printAst(
            pyStrings,
            node["right"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyStrings.append(")")
    else:
        pyStrings.append(f"{node['operator']}")
        pyStrings.append("(")
        printAst(
            pyStrings,
            node["left"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        pyStrings.append(")")
    if not printState.indexRef:
        pyStrings.append("]")


def printLiteral(pyStrings, node, printState):
    pyStrings.append(f"{node['value']}")


def printRef(pyStrings, node, printState):
    if printState.indexRef:
        pyStrings.append(f"{node['name']}[0]")
    else:
        pyStrings.append(f"{node['name']}")


def printAssignment(pyStrings, node, printState):
    printAst(
        pyStrings,
        node["target"],
        printState.copy(sep="", add="", printFirst=False, indexRef=True),
    )
    pyStrings.append(" = ")
    printAst(
        pyStrings,
        node["value"],
        printState.copy(sep="", add="", printFirst=False, indexRef=True),
    )


def printFuncReturn(pyStrings, node, printState):
    if printState.indexRef:
        if node.get("name"):
            pyStrings.append(f"return {node['name']}[0]")
        else:
            pyStrings.append(f"return {node['value']}")
    else:
        if node.get("name"):
            pyStrings.append(f"return {node['name']}")
        else:
            if node.get("value"):
                pyStrings.append(f"return {node['value']}")
            else:
                pyStrings.append(f"return None")


def printExit(pyStrings, node, printState):
    pyStrings.append("return")


def printReturn(pyStrings, node, printState):
    #    pyStrings.append("sys.exit(0)")
    pyStrings.append("return True")


def setupPrintFns():
    PRINTFN.update(
        {
            "subroutine": printSubroutine,
            "program": printProgram,
            "call": printCall,
            "arg": printArg,
            "variable": printVariable,
            "do": printDo,
            "index": printIndex,
            "if": printIf,
            "op": printOp,
            "literal": printLiteral,
            "ref": printRef,
            "assignment": printAssignment,
            "exit": printExit,
            "return": printReturn,
            "function": printFunction,
            "ret": printFuncReturn,
            #  "read": printFileRead,
            #  "open": printFileOpen,
            #  "close": printFileClose,
        }
    )


def printAst(pyStrings, root, printState):
    for node in root:
        if printState.printFirst:
            pyStrings.append(printState.sep)
        else:
            printState.printFirst = True
        if node.get("tag"):
            PRINTFN[node["tag"]](pyStrings, node, printState)


def printPython(gen, outputFile):
    with open(outputFile[0], "rb") as f:
        outputDict = pickle.load(f)

    pySrc = create_python_string(outputDict)
    with open(gen, "w") as f:
        f.write(pySrc)


def create_python_string(outputDict):
    pyStrings = []
    pyStrings.append("from typing import List\n")
    pyStrings.append("import math")
    setupPrintFns()
    printAst(pyStrings, outputDict["ast"], PrintState())
    return "".join(pyStrings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        nargs="*",
        help="Routines for which dependency graphs should be generated",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="Pickled version of the asts together with non-source code information",
    )
    args = parser.parse_args(sys.argv[1:])
    printPython(args.gen[0], args.files)
