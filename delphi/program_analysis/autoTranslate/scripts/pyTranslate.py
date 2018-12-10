#!/usr/bin/python

"""
Purpose:
    Convert a Fortran AST representation into a Python
    script having the same functionalities and performing
    the same operations as the original Fortran file.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        python pyTranslate -f <pickle_file> -g <python_file>

pickle_file: Pickled file containing the ast represenatation of the Fortran file
along with other non-source code information.

python_file: The Python file on which to write the resulting python script.
"""

import sys
import pickle
import argparse
from typing import List, Dict


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


class PythonCodeGenerator(object):
    def __init__(self):
        self.printFn = {}
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
        self.mathFuncs = ["mod", "exp", "cexp", "cmplx"]
        self.getframe_expr = "sys._getframe({}).f_code.co_name"
        self.pyStrings = []
        self.printFn = {
            "subroutine": self.printSubroutine,
            "program": self.printProgram,
            "call": self.printCall,
            "arg": self.printArg,
            "variable": self.printVariable,
            "do": self.printDo,
            "index": self.printIndex,
            "if": self.printIf,
            "op": self.printOp,
            "literal": self.printLiteral,
            "ref": self.printRef,
            "assignment": self.printAssignment,
            "exit": self.printExit,
            "return": self.printReturn,
            "function": self.printFunction,
            "ret": self.printFuncReturn,
        }

    def printSubroutine(self, node: Dict[str, str], printState: PrintState):
        self.pyStrings.append(f"\ndef {node['name']}(")
        args = []
        self.printAst(
            node["args"],
            printState.copy(
                sep=", ",
                add="",
                printFirst=False,
                definedVars=args,
                indexRef=False,
            ),
        )
        self.pyStrings.append("):")
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                definedVars=args,
                indexRef=True,
            ),
        )

    def printFunction(self, node, printState):
        self.pyStrings.append(f"\ndef {node['name']}(")
        args = []
        self.printAst(
            node["args"],
            printState.copy(
                sep=", ",
                add="",
                printFirst=False,
                definedVars=args,
                indexRef=False,
            ),
        )
        self.pyStrings.append("):")
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                definedVars=args,
                indexRef=True,
            ),
        )

    def printProgram(self, node, printState):
        self.printSubroutine(node, printState)
        self.pyStrings.append(f"\n\n{node['name']}(){printState.sep}")

    def printCall(self, node: Dict[str, str], printState: PrintState):
        if not printState.indexRef:
            self.pyStrings.append("[")

        inRef = False

        if node["name"] in self.libFns:
            node["name"] = node["name"].lower()
            if node["name"] in self.mathFuncs:
                node["name"] = "math." + node["name"]
            inRef = 1

        self.pyStrings.append(f"{node['name']}(")
        self.printAst(
            node["args"],
            printState.copy(
                sep=", ",
                add="",
                printFirst=False,
                definedVars=[],
                indexRef=inRef,
            ),
        )
        self.pyStrings.append(")")

        if not printState.indexRef:
            self.pyStrings.append("]")

    def printAst(self, root, printState):
        for node in root:
            if printState.printFirst:
                self.pyStrings.append(printState.sep)
            else:
                printState.printFirst = True
            if node.get("tag"):
                self.printFn[node["tag"]](node, printState)

    def printArg(self, node, printState):
        if node["type"] == "INTEGER":
            varType = "int"
        elif node["type"] in ["DOUBLE", "REAL"]:
            varType = "float"
        else:
            print(f"unrecognized type {node['type']}")
            sys.exit(1)
        self.pyStrings.append(f"{node['name']}: List[{varType}]")
        printState.definedVars += [node["name"]]

    def printVariable(self, node, printState):
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
            self.pyStrings.append(
                f"{node['name']}: List[{varType}] = [{initVal}]"
            )
        else:
            printState.printFirst = False

    def printDo(self, node, printState):
        self.pyStrings.append("for ")
        self.printAst(
            node["header"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        self.pyStrings.append(":")
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                indexRef=True,
            ),
        )

    def printIndex(self, node, printState):
        self.pyStrings.append(f"{node['name']}[0] in range(")
        self.printAst(
            node["low"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        self.pyStrings.append(", ")
        self.printAst(
            node["high"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        self.pyStrings.append("+1)")

    def printIf(self, node, printState):
        self.pyStrings.append("if ")
        self.printAst(
            node["header"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        self.pyStrings.append(":")
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                indexRef=True,
            ),
        )
        if "else" in node:
            self.pyStrings.append(printState.sep)
            self.pyStrings.append("else:")
            self.printAst(
                node["else"],
                printState.copy(
                    sep=printState.sep + printState.add,
                    printFirst=True,
                    indexRef=True,
                ),
            )

    def printOp(self, node, printState):
        if not printState.indexRef:
            self.pyStrings.append("[")
        if "right" in node:
            self.pyStrings.append("(")
            self.printAst(
                node["left"],
                printState.copy(
                    sep="", add="", printFirst=True, indexRef=True
                ),
            )

            operator_mapping = {
                ".ne.": " != ",
                ".gt.": " > ",
                ".eq.": " == ",
                ".lt.": " < ",
                ".le.": " <= ",
            }

            self.pyStrings.append(
                operator_mapping.get(
                    node["operator"].lower(), f" {node['operator']} "
                )
            )
            self.printAst(
                node["right"],
                printState.copy(
                    sep="", add="", printFirst=True, indexRef=True
                ),
            )
            self.pyStrings.append(")")
        else:
            self.pyStrings.append(f"{node['operator']}")
            self.pyStrings.append("(")
            self.printAst(
                node["left"],
                printState.copy(
                    sep="", add="", printFirst=True, indexRef=True
                ),
            )
            self.pyStrings.append(")")
        if not printState.indexRef:
            self.pyStrings.append("]")

    def printLiteral(self, node, printState):
        self.pyStrings.append(f"{node['value']}")

    def printRef(self, node, printState):
        if printState.indexRef:
            self.pyStrings.append(f"{node['name']}[0]")
        else:
            self.pyStrings.append(f"{node['name']}")

    def printAssignment(self, node, printState):
        self.printAst(
            node["target"],
            printState.copy(sep="", add="", printFirst=False, indexRef=True),
        )
        self.pyStrings.append(" = ")
        self.printAst(
            node["value"],
            printState.copy(sep="", add="", printFirst=False, indexRef=True),
        )

    def printFuncReturn(self, node, printState):
        if printState.indexRef:
            if node.get("name"):
                self.pyStrings.append(f"return {node['name']}[0]")
            else:
                self.pyStrings.append(f"return {node['value']}")
        else:
            if node.get("name"):
                self.pyStrings.append(f"return {node['name']}")
            else:
                if node.get("value"):
                    self.pyStrings.append(f"return {node['value']}")
                else:
                    self.pyStrings.append(f"return None")

    def printExit(self, node, printState):
        self.pyStrings.append("return")

    def printReturn(self, node, printState):
        self.pyStrings.append("return True")

    def get_python_source(self):
        return "".join(self.pyStrings)


def create_python_string(outputDict):
    code_generator = PythonCodeGenerator()
    code_generator.pyStrings.append("from typing import List\n")
    code_generator.pyStrings.append("import math")
    code_generator.printAst(outputDict["ast"], PrintState())
    return code_generator.get_python_source()


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
    with open(args.files[0], "rb") as f:
        outputDict = pickle.load(f)
    pySrc = create_python_string(outputDict)
    with open(args.gen[0], "w") as f:
        f.write(pySrc)
