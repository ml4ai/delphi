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
            "derived-type": self.printDerivedType,
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
                if node["isDevTypeVar"] == True:
                    initVal = init_val if initial_set else 0
                    varType = node["type"]
                else:
                    print(f"unrecognized type {node['type']}")
                    sys.exit(1)

            if "isDevTypeVar" in node and node["isDevTypeVar"]:
                self.pyStrings.append(
                    f"{self.nameMapper[node['name']]} =  {varType}()"
                )
            else:
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
            # Handles derived type variables
            if "isDevType" not in node or node["isDevType"] == False:
                self.pyStrings.append("[0]")
            if "isDevType" in node and node["isDevType"]:
                self.pyStrings.append(f".{node['field-name']}")
        # Handles array
        if "subscripts" in node:
            # Check if the node really holds an array. The is because the derive type with
            # more than 1 field access, for example var%x%y, node holds x%y also under
            # the subscripts. Thus, in order to avoid non-array derive types to be printed
            # in an array syntax, this check is necessary
            if "hasSubscripts" in node and node["hasSubscripts"]:
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
            else:
                self.pyStrings.append(".")
                self.printAst(
                    node["subscripts"],
                    printState.copy(
                        sep=", ", add="", printFirst=False, indexRef=True
                    ),
                )

    def printAssignment(self, node, printState):
        # Writing a target variable syntax
        if "subscripts" in node["target"][0]:   # Case where the target is an array
            if node["isDevType"] and "field-name" in node['target'][0]:
                if node['target'][0]['hasSubscripts']:
                    devObj = {"tag": node['target'][0]['tag'], "name": node['target'][0]['name'], "isDevType": node['target'][0]['isDevType'],
                              "hasSubscripts":node['target'][0]['isDevType'], "subscripts": [node['target'][0]['subscripts'].pop(0)]}
                    self.printAst(
                        [devObj],
                        printState.copy(
                            sep="", add="", printFirst=True, indexRef=True
                        ),
                    )
                else:
                    self.pyStrings.append(f"{node['target'][0]['name']}")
                self.pyStrings.append(f".{node['target'][0]['field-name']}")
            else:
                self.pyStrings.append(f"{node['target'][0]['name']}")
            self.pyStrings.append(".set_((")
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
                # Handles array or a variable that holds following attributes, such as var.x.y
                if "subscripts" in item: 
                    # If a variable is derived type
                    if item["isDevType"]:
                        # If hasSubscripts is true, the derived type object is also an array
                        # and has following subscripts. Therefore, in order to handle the syntax
                        # obj.get_((obj_index)).field.get_((field)), the code below handles
                        # obj.get_((obj_index)) first before the subscripts for the field variables
                        if "hasSubscripts" in item and item['hasSubscripts']:
                            devObjSub = [item["subscripts"].pop(0)]
                            write_string += ".get_(("
                            for sub in devObjSub:
                                if sub["tag"] == "ref":
                                    write_string += f"{sub['name']}[0]"
                                elif sub["tag"] == "literal":
                                    write_string += f"{sub['value']}"
                            write_string += "))"
                        # This is a case where the very first variable is not an array, but has subscripts,
                        # which holds information of following derived type field variables. i.e. var.x.y
                        # HOWEVER, the code below MUST be revisted and fixed. This is a hack of hard coding
                        # in order to print a format of var.x.y fixed number of 3 times
                        # In order to fix this, the entire printWrite should be modified that can do a recursion
                        # or break passed lists from the translate.py to have more clear groups.
                        # It cannot be handled now as it may cause a butterfly effect on all other I/O handling
                        if "hasSubscripts" in item and not item["hasSubscripts"] and "subscripts" in item and "arrayStat" not in item:
                            write_string += f".{item['subscripts'][0]['name']}.{item['subscripts'][0]['field-name']}"
                            write_string += ", "
                            write_string += f"{item['subscripts'][1]['name']}"
                            write_string += f".{item['subscripts'][1]['subscripts'][0]['name']}.{item['subscripts'][1]['subscripts'][0]['field-name']}"
                            write_string += ", "
                            write_string += f"{item['subscripts'][1]['subscripts'][1]['name']}"
                            write_string += f".{item['subscripts'][1]['subscripts'][1]['subscripts'][0]['name']}"
                            write_string += f".{item['subscripts'][1]['subscripts'][1]['subscripts'][0]['field-name']}"
                        if "field-name" in item:
                            write_string += f".{item['field-name']}"
                    # Handling array
                    if ("hasSubscripts" in item and item['hasSubscripts']) or ("arrayStat" in item and item["arrayStat"] == "isArray"):
                        i = 0
                        write_string += ".get_(("
                        for ind in item["subscripts"]:
                            # When an array uses another array's value as its index value
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
                    if "isDevType" not in item or ("isDevType" in item and not item["isDevType"]):
                        write_string += "[0]"
                    elif "isDevType" in item and item["isDevType"]:
                        write_string += f".{item['field-name']}"
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
            elif node["isDevTypeVar"] == True:
                varType = node["type"].lower() + "()"

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

            if node["isDevTypeVar"] == True:
                self.pyStrings.append(printState.sep)
                # This may require update later when we have to deal with the multi-dimensional derived type arrays
                upBound = node["up1"]
                self.pyStrings.append(f"for z in range(1, {upBound}+1):" + printState.sep)
                self.pyStrings.append(f"    obj = {node['type']}()" + printState.sep)
                self.pyStrings.append(f"    {node['name']}.set_(z, obj)" + printState.sep)

    def printDerivedType(self, node, printState):
        assert node["tag"] == "derived-type"
        self.pyStrings.append(f"class {node['name']}:")
        self.pyStrings.append(printState.sep)
        self.pyStrings.append("    def __init__(self):")
        self.pyStrings.append(printState.sep)

        curFieldType = ""
        fieldNum = 0
        for item in node:
            if f"field{fieldNum}" == item:
                if node[item][0]['type'].lower() == "integer":
                    curFieldType = "int"
                elif node[item][0]['type'].lower() in ("double", "real"):
                    curFieldType = "float"
                elif node[item][0]['type'].lower() == "character":
                    curFieldType = "str"

                fieldname = node[item][0]['field-id']
                if "array-size" in node[item][0]:
                    self.pyStrings.append(f"        self.{fieldname} =")
                    self.pyStrings.append(f" Array({curFieldType}, [")
                    self.pyStrings.append(f"(1, {node[item][0]['array-size']})])")
                else:
                    self.pyStrings.append(f"        self.{fieldname} :")
                    self.pyStrings.append(f" {curFieldType}")
                    if "value" in node[item][0]:
                        self.pyStrings.append(f" = {node[item][0]['value']}")
                    else:
                        self.pyStrings.append(" = None")
                self.pyStrings.append(printState.sep)
                fieldNum = fieldNum + 1

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
    derived_type_ast = []
    has_derived_type = False
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

    code_generator.pyStrings.extend(["\n\n"])

    # Copy the derived type ast from the main_ast into the separate list,
    # so it can be printed outside (above) the main method
    for index in list(main_ast[0]["body"]):
        if "derived-type" == index["tag"]:
            has_derived_type = True
            derived_type_ast.append(index)
            main_ast[0]["body"].remove(index)

    # Print derived type declaration(s)
    if has_derived_type == True:
        code_generator.nameMapping(derived_type_ast)
        code_generator.printAst(derived_type_ast, PrintState())

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
