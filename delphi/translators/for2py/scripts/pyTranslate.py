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
from .fortran_format import *


class PrintState:
    def __init__(
        self,
        sep="\n",
        add="    ",
        printFirst=True,
        definedVars=[],
        globalVars=[],
        indexRef=True,
        varTypes={},
    ):
        self.sep = sep
        self.add = add
        self.printFirst = printFirst
        self.definedVars = definedVars
        self.globalVars = globalVars
        self.indexRef = indexRef
        self.varTypes = varTypes

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
            "close": self.printClose,
            "array": self.printArray,
        }
        self.operator_mapping = {
            ".ne.": " != ",
            ".gt.": " > ",
            ".eq.": " == ",
            ".lt.": " < ",
            ".le.": " <= ",
        }
        self.readFormat = []

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
                node["name"] = f"math.{node['name']}"
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
            if node.get("tag"):
                if node["tag"] == "format":
                    self.printFn["format"](node, printState)
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

    def initializeFileVars(self, node, printState):
        label = node["args"][1]["value"]
        data_type = list_data_type(self.format_dict[label])
        index = 0
        for item in node["args"]:
            if item["tag"] == "ref":
                var = item["name"]
                self.printVariable(
                    {"name": var, "type": data_type[index]}, printState
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
        self.pyStrings.append(f"{node['name']}: List[{varType}]")
        printState.definedVars += [node["name"]]

    def printVariable(self, node, printState):
        if (
            node["name"] not in printState.definedVars
            and node["name"] not in printState.globalVars
        ):
            printState.definedVars += [node["name"]]
            if node["type"].upper() == "INTEGER":
                initVal = 0
                varType = "int"
            elif node["type"].upper() in ("DOUBLE", "REAL"):
                initVal = 0.0
                varType = "float"
            elif node["type"].upper() == "STRING":
                initVal = ""
                varType = "str"
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
        self.pyStrings.append(node["value"])

    def printRef(self, node, printState):
        self.pyStrings.append(node["name"])
        if printState.indexRef:
            self.pyStrings.append("[0]")

    def printAssignment(self, node, printState):
        if "subscripts" in node["target"][0]:
            self.pyStrings.append(f"{node['target'][0]['name']}.set_((")
            length = len(node["target"][0]["subscripts"])
            for ind in node["target"][0]["subscripts"]:
                index = ""
                if 'name' in ind:
                    index = ind['name']
                elif 'value' in ind:
                    index = ind['value']
                self.pyStrings.append(f"{index}[0]")
                if (length > 1):
                    self.pyStrings.append(", ")
                    length = length - 1
            self.pyStrings.append("), ")
        else:
            self.printAst(
                node["target"],
                printState.copy(sep="", add="", printFirst=False, indexRef=True),
            )
            self.pyStrings.append(" = ")
        if "subscripts" in node["value"][0]:
            self.pyStrings.append(f"{node['value'][0]['name']}.get_((")
            arrayLen = len(node["value"][0]["subscripts"])
            for ind in node["value"][0]["subscripts"]:
                if "name" in ind:
                    self.pyStrings.append(f"{ind['name']}[0]")
                else:
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

    def printFuncReturn(self, node, printState):
        if printState.indexRef:
            if node.get("name") is not None:
                val = node["name"] + "[0]"
            else:
                val = node["value"]
        else:
            if node.get("name") is not None:
                val = node["name"]
            else:
                if node.get("value") is not None:
                    val = node["value"]
                else:
                    val = "None"
        self.pyStrings.append(f"return {val}")

    def printExit(self, node, printState):
        self.pyStrings.append("return")

    def printReturn(self, node, printState):
        self.pyStrings.append("return True")

    def printOpen(self, node, printState):
        if node["args"][0].get("arg_name") == "UNIT":
            file_handle = "file_" + str(node["args"][1]["value"])
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
        self.pyStrings.append("(")
        for item in node["args"]:
            if item["tag"] == "ref":
                var = item["name"]
                self.pyStrings.append(f"{var},")
        self.pyStrings.append(
            f") = format_{format_label}_obj.read_line({file_handle}.readline())"
        )



    def printWrite(self, node, printState):
        hasArray = False
        arrayloc = 0
        for i in range(0, len(node["args"])):
            if "subscripts" in node["args"][i]:
                hasArray = True
                arrayloc = i

        if hasArray:
            var = 0
            name = node["args"][arrayloc]["name"]
            for args in node["args"]:
                if "subscripts" in args:
                    var = var + 1
                    self.pyStrings.append(f"var{var} = {name}.get_((")
                    subLength = len(args["subscripts"])
                    for ind in args["subscripts"]:
                        if ind["tag"] == "ref" and "name" in ind:
                            indName = ind["name"]
                            self.pyStrings.append(f"{indName}[0]")
                        if ind["tag"] == "op":
                            indOp = ind["operator"]
                            indValue = ind["left"][0]["value"]
                            self.pyStrings.append(f"{indOp}{indValue}")
                        elif ind["tag"] == "literal" and "value" in ind:
                            indValue = ind["value"]
                            self.pyStrings.append(f"{indValue}")
                        if (subLength > 1):
                            self.pyStrings.append(", ")
                            subLength = subLength - 1
                    self.pyStrings.append("))")
                    self.pyStrings.append(printState.sep)
        write_list = []
        write_string = ""
        file_number = str(node["args"][0]["value"])
        for i in range (0, len(node["args"])):
            if i == 0 and node["args"][i]["type"] == "int":
                file_handle = "file_" + file_number
            if i == 1 and"type" in node["args"][1]:
                if node["args"][1]["type"] == "int":
                    format_label = node["args"][1]["value"]
        else:
            format_label = node["args"][0]["value"]
        self.pyStrings.append(f"write_list_{file_number} = [")
        var_num = 0
        for item in node["args"]:
            if hasArray == False:
                if item["tag"] == "ref":
                    write_string += f"{item['name']}, "
            else:
                if "subscripts" in item:
                    var_num = var_num + 1 
                    write_string += f"var{var_num}, "
        self.pyStrings.append(f"{write_string[:-2]}]")
        self.pyStrings.append(printState.sep)
        self.pyStrings.append(
            f"write_line = format_{format_label}_obj.write_line(write_list_{file_number})"
        )
        self.pyStrings.append(printState.sep)
        self.pyStrings.append(f"{file_handle}.write(write_line)")

    def printExit(self, node, printState):
        self.pyStrings.append("return")

    def printFormat(self, node, printState):
        type_list = []
        try:
            rep_count = int(node["args"][-1]["value"])
        except ValueError:
            for item in node["args"]:
                type_list.append(item["value"])
        else:
            values = [item["value"] for item in node["args"][:-1]]
            type_list.append(f"{rep_count}({','.join(values)})")

        self.pyStrings.append(printState.sep)
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
        file_id = node["args"][0]["value"]
        self.pyStrings.append(f"file_{file_id}.close()")

    def printArray(self, node, printState):
        if int(node['count']) == 1:
            if (
                node["name"] not in printState.definedVars
                and node["name"] not in printState.globalVars
            ):
                printState.definedVars += [node["name"]]
                loBound = 1
                upBound = node["up" + node['count']]

                self.pyStrings.append(
                    f"{node['name']} = Array([({loBound}, {upBound})])"
                )
        elif int(node['count']) > 1:
            printState.definedVars += [node["name"]]

            self.pyStrings.append(f"{node['name']} = Array([")
            for i in range (0, int(node['count'])):  
                loBound = node["low" + str(i+1)]
                upBound = node["up" + str(i+1)]
                dimensions = f"({loBound}, {upBound})"
                if i < int(node['count'])-1:
                    self.pyStrings.append(f"{dimensions}, ")
                else:
                    self.pyStrings.append(f"{dimensions}")
            self.pyStrings.append("])")
        else:
            printState.printFirst = False

    def get_python_source(self):
        return "".join(self.pyStrings)


def create_python_string(outputDict):
    code_generator = PythonCodeGenerator()
    code_generator.pyStrings.extend(
        [
            "from typing import List\n",
            "import math\n",
            "from fortran_format import *\n",
            "from for2py_arrays import *",
        ]
    )
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
