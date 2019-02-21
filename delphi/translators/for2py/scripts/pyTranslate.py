#!/usr/bin/python3.7

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
import re
from typing import List, Dict
from delphi.translators.for2py.scripts.fortran_format import *

class PrintState:
    def __init__(
            self,
            sep="\n",
            add="    ",
            printFirst=True,
            callSource=None,
            definedVars=[],
            globalVars=[],
            functionScope='',
            indexRef=True,
            varTypes={},
    ):
        self.sep = sep
        self.add = add
        self.printFirst = printFirst
        self.callSource = callSource
        self.definedVars = definedVars
        self.globalVars = globalVars
        self.functionScope = functionScope
        self.indexRef = indexRef
        self.varTypes = varTypes

    def copy(
            self,
            sep=None,
            add=None,
            printFirst=None,
            callSource=None,
            definedVars=None,
            globalVars=None,
            functionScope=None,
            indexRef=None,
            varTypes=None,
    ):
        return PrintState(
            self.sep if sep == None else sep,
            self.add if add == None else add,
            self.printFirst if printFirst == None else printFirst,
            self.callSource if callSource == None else callSource,
            self.definedVars if definedVars == None else definedVars,
            self.globalVars if globalVars == None else globalVars,
            self.functionScope if functionScope == None else functionScope,
            self.indexRef if indexRef == None else indexRef,
            self.varTypes if varTypes == None else varTypes,
        )


programName = ''


class PythonCodeGenerator(object):
    def __init__(self):
        self.printFn = {}
        self.libFns = [
            "mod",
            "exp",
            "index",
            "min",
            "max",
            "cexp",
            "cmplx",
            "atan",
            "cos",
            "sin",
            "acos",
            "asin",
            "tan",
            "atan",
            "sqrt",
            "log",
            "abs",
        ]
        self.variableMap = {}
        # This list contains the private functions
        self.privFunctions = []
        # This dictionary contains the mapping of symbol names to pythonic names
        self.nameMapper = {}
        # Dictionary to hold functions and its arguments
        self.funcArgs = {}
        self.mathFuncs = ["exp", "cexp", "cmplx", "cos", "sin", "acos", "asin", "tan", "atan", "sqrt", "log", "abs"]
        self.getframe_expr = "sys._getframe({}).f_code.co_name"
        self.pyStrings = []
        self.stateMap = {"UNKNOWN": "r", "REPLACE": "w"}
        self.format_dict = {}
        self.printFn = {
            "subroutine": self.printSubroutine,
            "program": self.printProgram,
            "call": self.printCall,
            "arg": self.printArg,
            "variable": self.printVariable,
            "do": self.printDo,
            "do-while": self.printDoWhile,
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
            "stop": self.printExit,
            "read": self.printRead,
            "write": self.printWrite,
            "open": self.printOpen,
            "format": self.printFormat,
            "module": self.printModule,
            "use": self.printUse,
            "close": self.printClose,
            "private": self.printPrivate,
            "array": self.printArray,
        }
        self.operator_mapping = {
            ".ne.": " != ",
            ".gt.": " > ",
            ".eq.": " == ",
            ".lt.": " < ",
            ".le.": " <= ",
            ".ge.": " >= ",
            ".and.": " and ",
            ".or.": " or ",
        }
        self.readFormat = []

    def printSubroutine(self, node: Dict[str, str], printState: PrintState):
        self.pyStrings.append(f"\ndef {self.nameMapper[node['name']]}(")
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
        self.pyStrings.append(f"\ndef {self.nameMapper[node['name']]}(")
        args = []
        self.funcArgs[self.nameMapper[node['name']]] = [self.nameMapper[x['name']] for x in node['args']]
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
        if printState.sep != '\n':
            printState.sep = '\n'
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                definedVars=args,
                indexRef=True,
                functionScope=self.nameMapper[node['name']],
            ),
        )

    def printModule(self, node, printState):

        self.pyStrings.append("\n")
        args = []
        self.printAst(
            node["body"],
            printState.copy(
                sep="",
                printFirst=True,
                definedVars=args,
                indexRef=True,
            ),
        )

    def printProgram(self, node, printState):
        global programName
        self.printSubroutine(node, printState)
        programName = self.nameMapper[node['name']]

    def printCall(self, node: Dict[str, str], printState: PrintState):
        if not printState.indexRef:
            self.pyStrings.append("[")

        inRef = False

        if node["name"].lower() in self.libFns:
            node["name"] = node["name"].lower()
            if node["name"] in self.mathFuncs:
                node["name"] = f"math.{node['name']}"
            inRef = 1

        if node["name"].lower() == "index":
            var = self.nameMapper[node["args"][0]["name"]]
            toFind = node["args"][1]["value"]
            self.pyStrings.append(f"{var}[0].find({toFind})")

        elif node["name"] == "mod":
            self.printAst(
                node["args"],
                printState.copy(
                    sep="%",
                    add="",
                    printFirst=False,
                    definedVars=[],
                    indexRef=inRef,
                ),
            )
        else:
            argSize = len(node["args"])
            assert argSize >= 1
            self.pyStrings.append(f"{node['name']}(")
            for arg in range (0, argSize):
                self.printAst(
                    [node["args"][arg]],
                    printState.copy(
                        sep=", ",
                        add="",
                        printFirst=False,
                        callSource="Call",
                        definedVars=[],
                        indexRef=inRef,
                    ),
                )
                if arg < argSize - 1:
                    self.pyStrings.append(", ")
            self.pyStrings.append(")")

        if not printState.indexRef:
            self.pyStrings.append("]")

    def printAst(self, root, printState):
        for node in root:
            if node.get("tag"):
                if node["tag"] == "format":
                    self.printFn["format"](node, printState)
                elif node["tag"] == "if":
                    for item in node["header"]:
                        if item["tag"] == "format":
                            self.printFn["format"](item, printState)

        for node in root:
            if node.get("tag"):
                if node["tag"] == "read":
                    self.initializeFileVars(node, printState)

        for node in root:
            if printState.printFirst:
                self.pyStrings.append(printState.sep)
            else:
                printState.printFirst = True
            if node.get("tag") and node.get("tag") != "format":
                self.printFn[node["tag"]](node, printState)

    def printPrivate(self, node, prinState):
        self.privFunctions.append(node["name"])
        self.nameMapper[node["name"]] = "_" + node["name"]

    def initializeFileVars(self, node, printState):
        label = node["args"][1]["value"]
        data_type = list_data_type(self.format_dict[label])
        index = 0
        for item in node["args"]:
            if item["tag"] == "ref":
                # if item["name"] in self.privFunctions:
                #     var = "_" + item["name"]
                # else:
                #     var = item["name"]
                self.printVariable(
                    {"name": self.nameMapper[item["name"]], "type": data_type[index]}, printState
                )
                self.pyStrings.append(printState.sep)
                index += 1

    def printArg(self, node, printState):
        if node["type"].upper() == "INTEGER":
            varType = "int"
        elif node["type"].upper() in ("DOUBLE", "REAL"):
            varType = "float"
        elif node["type"].upper() == "CHARACTER":
            varType = "str"
        else:
            print(f"unrecognized type {node['type']}")
            sys.exit(1)
        if node["arg_type"] == "arg_array":
            self.pyStrings.append(f"{self.nameMapper[node['name']]}")
        else:
            self.pyStrings.append(f"{self.nameMapper[node['name']]}: List[{varType}]")
        printState.definedVars += [self.nameMapper[node["name"]]]

    def printVariable(self, node, printState):
        initial_set = False
        if (
                self.nameMapper[node["name"]] not in printState.definedVars
                and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            printState.definedVars += [self.nameMapper[node["name"]]]
            if node.get('value'):
                init_val = node['value'][0]['value']
                initial_set = True

            if node["type"].upper() == "INTEGER":
                initVal = init_val if initial_set else 0
                varType = "int"
            elif node["type"].upper() in ("DOUBLE", "REAL"):
                initVal = init_val if initial_set else 0.0
                varType = "float"
            elif node["type"].upper() == "STRING" or node["type"].upper() == "CHARACTER":
                initVal = init_val if initial_set else ""
                varType = "str"
            else:
                print(f"unrecognized type {node['type']}")
                sys.exit(1)
            if printState.functionScope:
                if not self.nameMapper[node['name']] in self.funcArgs.get(printState.functionScope):
                    self.pyStrings.append(
                        f"{self.nameMapper[node['name']]}: List[{varType}] = [{initVal}]"
                    )
                else:
                    self.pyStrings.append(
                        f"{self.nameMapper[node['name']]}: List[{varType}]"
                    )
            else:
                self.pyStrings.append(
                    f"{self.nameMapper[node['name']]}: List[{varType}] = [{initVal}]"
                )

            # The code below might cause issues on unexpected places.
            # If weird variable declarations appear, check code below

            if not printState.sep:
                printState.sep = '\n'
            self.pyStrings.append(printState.sep)
            self.variableMap[self.nameMapper[node['name']]] = node['type']
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

    def printDoWhile(self, node, printState):
        self.pyStrings.append("while ")
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
        self.pyStrings.append(f"{self.nameMapper[node['name']]}[0] in range(")
        self.printAst(
            node["low"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        self.pyStrings.append(", ")
        self.printAst(
            node["high"],
            printState.copy(sep="", add="", printFirst=True, indexRef=True),
        )
        if node.get("step"):
            self.pyStrings.append("+1, ")
            self.printAst(
                node["step"],
                printState.copy(sep="", add="", printFirst=True, indexRef=True),
            )
            self.pyStrings.append(")")
        else:
            self.pyStrings.append("+1)")

    def printIf(self, node, printState):
        self.pyStrings.append("if ")
        newHeaders = []
        for item in node["header"]:
            if item["tag"] != "format":
                newHeaders.append(item)
        self.printAst(
            newHeaders,
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
            self.pyStrings.append(printState.sep + "else:")
            self.printAst(
                node["else"],
                printState.copy(
                    sep=printState.sep + printState.add,
                    printFirst=True,
                    indexRef=True,
                ),
            )

    def printOp(self, node, printState):
        node["left"][0]["op"] = True
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

            self.pyStrings.append(
                self.operator_mapping.get(
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
            self.pyStrings.append(f"{node['operator']}(")
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
        # if printState.callSource == "Call":
        #     self.pyStrings.append(f"[{node['value']}]")
        # else:
        #     self.pyStrings.append(node["value"])
        self.pyStrings.append(node["value"])

    def printRef(self, node, printState):
        self.pyStrings.append(self.nameMapper[node["name"]])
        if printState.indexRef and "subscripts" not in node:
            self.pyStrings.append("[0]")
        # Handles array
        if "subscripts" in node:
            if node["name"].lower() not in self.libFns:
                self.pyStrings.append(".get_((")
            self.pyStrings.append("(")
            value = ""
            subLength = len(node["subscripts"])
            for ind in node["subscripts"]:
                if ind["tag"] == "ref":
                    indName = ind["name"]
                    self.pyStrings.append(f"{indName}")
                    if "subscripts" in ind:
                        self.pyStrings.append(".get_((")
                        self.printAst(
                            ind["subscripts"],
                            printState.copy(
                                sep=", ", add="", printFirst=False, indexRef=True
                            ),
                        )
                        self.pyStrings.append("))")
                    else:
                        self.pyStrings.append("[0]")
                if ind["tag"] == "op":
                    if "right" not in ind:
                        self.pyStrings.append(f"{ind['operator']}{ind['left'][0]['value']}")
                    else:
                        self.printAst(
                            ind["left"],
                            printState.copy(
                                sep="", add="", printFirst=True, indexRef=True
                            ),
                        )
                        self.pyStrings.append(f"{ind['operator']}")
                        self.printAst(
                            ind["right"],
                            printState.copy(
                                sep="", add="", printFirst=True, indexRef=True
                            ),
                        )
                elif ind["tag"] == "literal" and "value" in ind:
                    indValue = ind["value"]
                    self.pyStrings.append(f"{indValue}")
                if (subLength > 1):
                    self.pyStrings.append(", ")
                    subLength = subLength - 1
            if node["name"].lower() not in self.libFns:
                self.pyStrings.append("))")
            self.pyStrings.append(")")

    def printAssignment(self, node, printState):
        # Writing a target variable syntax
        if "subscripts" in node["target"][0]:   # Case where the target is an array
            self.pyStrings.append(f"{node['target'][0]['name']}.set_((")
            length = len(node["target"][0]["subscripts"])
            for ind in node["target"][0]["subscripts"]:
                index = ""
                if ind["tag"] == "literal": # Case where using literal value as an array index
                    index = ind["value"]
                    self.pyStrings.append(f"{index}")
                if ind["tag"] == "ref": # Case where using variable as an array index
                    index = ind["name"]
                    self.pyStrings.append(f"{index}[0]")
                if ind["tag"] == "op":  # Case where a literal index has an operator
                    operator = ind["operator"]
                    # For left index
                    if ind["left"][0]["tag"] == "literal":
                        lIndex = ind["left"][0]["value"]
                        self.pyStrings.append(f"{lIndex}")
                    elif (ind["left"][0]["tag"] == "ref"):
                        lIndex = ind["left"][0]["name"]
                        self.pyStrings.append(f"{lIndex}[0]")

                    self.pyStrings.append(f" {operator} ")
                    # For right index
                    if ind["right"][0]["tag"] == "literal":
                        rIndex = ind["right"][0]["value"]
                        self.pyStrings.append(f"{rIndex}")
                    elif (ind["right"][0]["tag"] == "ref"):
                        rIndex = ind["right"][0]["name"]
                        self.pyStrings.append(f"{rIndex}[0]")

                if (length > 1):
                    self.pyStrings.append(", ")
                    length = length - 1
            self.pyStrings.append("), ")
        else:   # Case where the target is a single variable
            self.printAst(
                node["target"],
                printState.copy(sep="", add="", printFirst=False, indexRef=True),
            )
            self.pyStrings.append(" = ")

        # Writes a syntax for the source that is right side of the '=' operator
        if "subscripts" in node["value"][0]:
            self.pyStrings.append(f"{node['value'][0]['name']}.get_((")
            arrayLen = len(node["value"][0]["subscripts"])
            for ind in node["value"][0]["subscripts"]:
                if "name" in ind:
                    self.pyStrings.append(f"{ind['name']}[0]")
                elif "operator" in ind:
                    # For left index
                    if ind["left"][0]["tag"] == "literal":
                        lIndex = ind["left"][0]["value"]
                        self.pyStrings.append(f"{lIndex}")
                    elif (ind["left"][0]["tag"] == "ref"):
                        lIndex = ind["left"][0]["name"]
                        self.pyStrings.append(f"{lIndex}[0]")

                    self.pyStrings.append(f" {ind['operator']} ")
                    # For right index
                    if ind["right"][0]["tag"] == "literal":
                        rIndex = ind["right"][0]["value"]
                        self.pyStrings.append(f"{rIndex}")
                    elif (ind["right"][0]["tag"] == "ref"):
                        rIndex = ind["right"][0]["name"]
                        self.pyStrings.append(f"{rIndex}[0]")
                else:
                    assert ind["tag"] == "literal"
                    self.pyStrings.append(f"{ind['value']}")
                if arrayLen > 1:
                    self.pyStrings.append(", ")
                    arrayLen = arrayLen - 1
            self.pyStrings.append("))")
        else:
            self.printAst(
                node["value"],
                printState.copy(sep="", add="", printFirst=False, indexRef=True),
        )
        if "subscripts" in node["target"][0]:
            self.pyStrings.append(")")

    def printUse(self, node, printState):
        if node.get("include"):
            imports.append(f"from m_{node['arg'].lower()} import {', '.join(node['include'])}\n")
        else:
            imports.append(f"from m_{node['arg'].lower()} import *\n")

    def printFuncReturn(self, node, printState):
        if printState.indexRef:
            if node.get("args"):
                self.pyStrings.append(f"return ")
                self.printCall(node, printState)
                return
            if node.get("name") is not None:
                val = self.nameMapper[node["name"]] + "[0]"
            else:
                val = node["value"]
        else:
            if node.get("name") is not None:
                val = self.nameMapper[node["name"]]
            else:
                if node.get("value") is not None:
                    val = node["value"]
                else:
                    val = "None"
        self.pyStrings.append(f"return {val}")

    def printExit(self, node, printState):
        if node.get("value"):
            self.pyStrings.append(f"print({node['value']})")
            self.pyStrings.append(printState.sep)
        self.pyStrings.append("return")

    def printReturn(self, node, printState):
        self.pyStrings.append("")

    def printOpen(self, node, printState):
        if node["args"][0].get("arg_name") == "UNIT":
            file_handle = "file_" + str(node["args"][1]["value"])
        elif node["args"][0].get("tag") == "ref":
            file_handle = "file_" + str(self.nameMapper[node["args"][0]["name"]])
        else:
            file_handle = "file_" + str(node["args"][0]["value"])
        self.pyStrings.append(f"{file_handle} = ")
        for index, item in enumerate(node["args"]):
            if item.get("arg_name"):
                if item["arg_name"] == "FILE":
                    file_name = node["args"][index + 1]["value"][1:-1]
                    open_state = "r"
                elif item["arg_name"] == "STATUS":
                    open_state = node["args"][index + 1]["value"][1:-1]
                    open_state = self.stateMap[open_state]

        self.pyStrings.append(f'open("{file_name}", "{open_state}")')

    def printRead(self, node, printState):
        file_number = str(node["args"][0]["value"])
        if node["args"][0]["type"] == "int":
            file_handle = "file_" + file_number
        if node["args"][1]["type"] == "int":
            format_label = node["args"][1]["value"]
        
        isArray = False
        tempInd = 0
        if "subscripts" in node["args"][2]:
            array_len = len(node["args"]) - 2
            self.pyStrings.append(f"tempVar = [0] * {array_len}")
            self.pyStrings.append(printState.sep)

        ind = 0
        self.pyStrings.append("(")
        for item in node["args"]:
            if item["tag"] == "ref":
                var = self.nameMapper[item["name"]]
                if "subscripts" in item:
                    isArray = True
                    self.pyStrings.append(f"tempVar[{tempInd}]")
                    tempInd = tempInd + 1
                else:
                    self.pyStrings.append(f"{var}[0]")
                if ind < len(node["args"]) - 1:
                    self.pyStrings.append(", ")
            ind = ind + 1
        self.pyStrings.append(
            f") = format_{format_label}_obj.read_line({file_handle}.readline())"
        )
        self.pyStrings.append(printState.sep)

        if isArray == True:
            tempInd = 0 # Re-initialize to zero for array index
            for item in node["args"]:
                if item["tag"] == "ref":
                    var = self.nameMapper[item["name"]]
                    if "subscripts" in item:
                        self.pyStrings.append(f"{var}.set_((")
                        self.printAst(
                            item["subscripts"],
                            printState.copy(sep=", ", add="", printFirst=False, indexRef=True),
                        )
                        self.pyStrings.append(f"), tempVar[{tempInd}])")
                        tempInd = tempInd + 1
                        self.pyStrings.append(printState.sep)
                ind = ind + 1

    def printWrite(self, node, printState):
        write_string = ""
        # Check whether write to file or output stream
        if str(node["args"][0].get("value")) == "*":
            write_target = "outStream"
        else:
            write_target = "file"
            if node["args"][0].get("value"):
                file_id = str(node["args"][0]["value"])
            elif str(node["args"][0].get("tag")) == "ref":
                file_id = str(self.nameMapper[node["args"][0].get("name")])
            file_handle = "file_" + file_id

        # Check whether format has been specified
        if str(node["args"][1]["value"]) == "*":
            format_type = "runtime"
        else:
            format_type = "specifier"
            if node["args"][1]["type"] == "int":
                format_label = node["args"][1]["value"]

        if write_target == "file":
            self.pyStrings.append(f"write_list_{file_id} = [")
        elif write_target == "outStream":
            self.pyStrings.append(f"write_list_stream = [")

        # Check for variable arguments specified to the write statement
        for item in node["args"]:
            if item["tag"] == "ref":
                write_string += f"{self.nameMapper[item['name']]}"
                if "subscripts" in item: # Handles array
                    i = 0
                    write_string += ".get_(("
                    for ind in item["subscripts"]:
                        if "subscripts" in ind:
                            write_string += f"{ind['name']}.get_(("
                            for sub in ind["subscripts"]:
                                if sub["tag"] == "ref":
                                    write_string += f"{sub['name']}[0]"
                                elif sub["tag"] == "literal":
                                    write_string += f"{sub['value']}"
                            write_string += "))" 
                        elif "operator" in ind:
                            if "right" not in ind:
                                write_string += f"{ind['operator']}"

                            if ind["left"][0]["tag"] == "ref":
                                write_string += f"{ind['left'][0]['name']}[0] "
                            else:
                                assert ind["left"][0]["tag"] == "literal"
                                write_string += f"{ind['left'][0]['value']} "

                            if "right" in ind:
                                write_string += f"{ind['operator']} "

                                if ind["right"][0]["tag"] == "ref":
                                    write_string += f"{ind['right'][0]['name']} "
                                else:
                                    assert ind["right"][0]["tag"] == "literal"
                                    write_string += f"{ind['right'][0]['value']} "
                        else:
                            if ind["tag"] == "ref":
                                write_string += f"{ind['name']}[0]"
                            elif ind["tag"] == "op":
                                write_string += f"{ind['operator']}"
                                assert ind["left"][0]["tag"] == "literal"
                                write_string += f"{ind['left'][0]['value']}"
                            elif ind["tag"] == "literal":
                                write_string += f"{ind['value']}"
                        if i < len(item["subscripts"]) - 1:
                            write_string += ", " 
                            i = i + 1
                    write_string += "))" 
                if printState.indexRef and "subscripts" not in item:
                    write_string += "[0]"
                write_string += ", "
        self.pyStrings.append(f"{write_string[:-2]}]")
        self.pyStrings.append(printState.sep)

        # If format specified and output in a file, execute write_line on file handler
        if write_target == "file":
            if format_type == "specifier":
                self.pyStrings.append(f"write_line = format_{format_label}_obj.write_line(write_list_{file_id})")
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"{file_handle}.write(write_line)")
            elif format_type == "runtime":
                self.pyStrings.append("output_fmt = list_output_formats([")
                for var in write_string.split(','):
                    varMatch = re.match(r'^(.*?)\[\d+\]|^(.*?)[^\[]', var.strip())
                    if varMatch:
                        var = varMatch.group(1)
                        self.pyStrings.append(f"\"{self.variableMap[var.strip()]}\",")
                self.pyStrings.append("])" + printState.sep)
                self.pyStrings.append("write_stream_obj = Format(output_fmt)" + printState.sep)
                self.pyStrings.append(f"write_line = write_stream_obj.write_line(write_list_{file_id})")
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"{file_handle}.write(write_line)")

        # If printing on stdout, handle accordingly
        elif write_target == "outStream":
            if format_type == "runtime":
                self.pyStrings.append("output_fmt = list_output_formats([")
                for var in write_string.split(','):
                    varMatch = re.match(r'^(.*?)\[\d+\]|^(.*?)[^\[]', var.strip())
                    if varMatch:
                        var = varMatch.group(1)
                        self.pyStrings.append(f"\"{self.variableMap[var.strip()]}\",")
                self.pyStrings.append("])" + printState.sep)
                self.pyStrings.append("write_stream_obj = Format(output_fmt)" + printState.sep)
                self.pyStrings.append("write_line = write_stream_obj.write_line(write_list_stream)" + printState.sep)
                self.pyStrings.append("sys.stdout.write(write_line)")
            elif format_type == "specifier":
                self.pyStrings.append(f"write_line = format_{format_label}_obj.write_line(write_list_stream)")
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"sys.stdout.write(write_line)")

    def nameMapping(self, ast):
        for item in ast:
            if item.get("name"):
                self.nameMapper[item["name"]] = item["name"]
            for inner in item:
                if type(item[inner]) == list:
                    self.nameMapping(item[inner])

    def printFormat(self, node, printState):
        type_list = []
        temp_list = []
        _re_int = re.compile(r'^\d+$')
        format_list = [token["value"] for token in node["args"]]

        for token in format_list:
            if not _re_int.match(token):
                temp_list.append(token)
            else:
                type_list.append(f"{token}({','.join(temp_list)})")
                temp_list = []
        if len(type_list) == 0:
            type_list = temp_list

        # try:
        #     rep_count = int(node["args"][-1]["value"])
        # except ValueError:
        #     for item in node["args"]:
        #         type_list.append(item["value"])
        # else:
        #     values = [item["value"] for item in node["args"][:-1]]
        #     type_list.append(f"{rep_count}({','.join(values)})")

        self.pyStrings.append(printState.sep)
        self.nameMapper[f"format_{node['label']}"] = f"format_{node['label']}"
        self.printVariable(
            {"name": "format_" + node["label"], "type": "STRING"}, printState
        )
        self.format_dict[node["label"]] = type_list
        self.pyStrings.extend(
            [
                printState.sep,
                f"format_{node['label']} = {type_list}",
                printState.sep,
                f"format_{node['label']}_obj = Format(format_{node['label']})",
                printState.sep,
            ]
        )

    def printClose(self, node, printState):
        file_id = node["args"][0]["value"] if node["args"][0].get("value") else self.nameMapper[node["args"][0]["name"]]
        self.pyStrings.append(f"file_{file_id}.close()")

    def printArray(self, node, printState):
        """ Prints out the array declaration in a format of Array class
            object declaration. 'arrayName = Array(Type, [bounds])'
        """
        initial_set = False
        if (
                self.nameMapper[node["name"]] not in printState.definedVars
                and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            printState.definedVars += [self.nameMapper[node["name"]]]
            assert int(node['count']) > 0
            printState.definedVars += [node["name"]]

            varType = ""
            if node["type"].upper() == "INTEGER":
                varType = "int"
            elif node["type"].upper() in ("DOUBLE", "REAL"):
                varType = "float"
            elif node["type"].upper() == "CHARACTER":
                varType = "str"
            assert varType != ""
            
            self.pyStrings.append(f"{node['name']} = Array({varType}, [")
            for i in range (0, int(node['count'])):  
                loBound = node["low" + str(i+1)]
                upBound = node["up" + str(i+1)]
                dimensions = f"({loBound}, {upBound})"
                if i < int(node['count'])-1:
                    self.pyStrings.append(f"{dimensions}, ")
                else:
                    self.pyStrings.append(f"{dimensions}")
            self.pyStrings.append("])")

    def get_python_source(self):
        return "".join(self.pyStrings)


'''
Counts the number of modules in the fortran file including the program file.
Each module is written out into a separate python file.    
'''


def file_count(root) -> Dict:
    file_desc = {}
    for index, node in enumerate(root):
        program_type = node.get("tag")
        if program_type and program_type in ("module", "program", "subroutine"):
            file_desc[node["name"]] = (program_type, index)

    return file_desc


def create_python_string(outputDict):
    program_type = file_count(outputDict["ast"])
    py_sourcelist = []
    main_ast = []
    global imports

    for file in program_type:
        imports = []
        if 'module' in program_type[file]:
            ast = [outputDict["ast"][program_type[file][1]]]
        else:
            main_ast.append(outputDict["ast"][program_type[file][1]])
            if [program_type[file][0]] == "program":
                main_name = file
            continue
            # ast = [outputDict["ast"][program_type[file][1]]]
        code_generator = PythonCodeGenerator()
        code_generator.pyStrings.extend(
            [
                "import sys\n"
                "from typing import List\n",
                "import math\n",
                "from fortran_format import *",
            ]
        )
        # Fill the name mapper dictionary
        code_generator.nameMapping(ast)
        code_generator.printAst(ast, PrintState())
        imports = ''.join(imports)
        if len(imports) != 0:
            code_generator.pyStrings.insert(1, imports)
        if programName != '':
            code_generator.pyStrings.extend(
                [
                    f"\n\n{programName}()\n"
                ]
            )
        py_sourcelist.append((code_generator.get_python_source(), file, program_type[file][0]))

    # Writing the main program section
    code_generator = PythonCodeGenerator()
    code_generator.pyStrings.extend(
        [
            "import sys\n"
            "from typing import List\n",
            "import math\n",
            "from fortran_format import *\n",
            "from for2py_arrays import *",
        ]
    )
    code_generator.nameMapping(main_ast)
    code_generator.printAst(main_ast, PrintState())
    imports = ''.join(imports)
    if len(imports) != 0:
        code_generator.pyStrings.insert(1, imports)
    if programName != '':
        code_generator.pyStrings.extend(
            [
                f"\n\n{programName}()\n"
            ]
        )
    py_sourcelist.append((code_generator.get_python_source(), main_ast, "program"))

    return py_sourcelist


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
    for item in pySrc:
        if item[2] == "module":
            with open('m_' + item[1].lower() + '.py', "w") as f:
                f.write(item[0])
        else:
            with open(args.gen[0], "w") as f:
                f.write(item[0])
