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
import re
from typing import List, Dict
from fortran_format import *


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
            "mod",
            "exp",
            "index",
            "min",
            "max",
            "cexp",
            "cmplx",
            "atan",
        ]
        self.variableMap = {}
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

        if node["name"].lower() in self.libFns:
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
        if node["type"] == "INTEGER":
            varType = "int"
        elif node["type"] in ("DOUBLE", "REAL"):
            varType = "float"
        elif node["type"] == "CHARACTER":
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
            if node["type"] == "INTEGER":
                initVal = 0
                varType = "int"
            elif node["type"] in ("DOUBLE", "REAL"):
                initVal = 0.0
                varType = "float"
            elif node["type"] == "STRING" or node["type"] == "CHARACTER":
                initVal = ""
                varType = "str"
            else:
                print(f"unrecognized type {node['type']}")
                sys.exit(1)
            self.pyStrings.append(
                f"{node['name']}: List[{varType}] = [{initVal}]"
            )
            self.variableMap[node['name']] = node['type']
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
        self.pyStrings.append("")

    def printOpen(self, node, printState):
        if node["args"][0].get("arg_name") == "UNIT":
            file_handle = "file_" + str(node["args"][1]["value"])
        elif node["args"][0].get("tag") == "ref":
            file_handle = "file_" + str(node["args"][0]["name"])
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

        write_string = ""
        # Check whether write to file or output stream
        if str(node["args"][0].get("value")) == "*":
            write_target = "outStream"
        else:
            write_target = "file"
            if node["args"][0].get("value"):
                file_id = str(node["args"][0]["value"])
            elif str(node["args"][0].get("tag")) == "ref":
                file_id = str(node["args"][0].get("name"))
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

        for item in node["args"]:
            if item["tag"] == "ref":
                write_string += f"{item['name']}"
                if printState.indexRef:
                    write_string += "[0]"
                write_string += ", "
        self.pyStrings.append(f"{write_string[:-2]}]")
        self.pyStrings.append(printState.sep)

        # If format specified and output in a file, execute write_line on file handler
        if write_target == "file" and format_type == "specifier":
            self.pyStrings.append(f"write_line = format_{format_label}_obj.write_line(write_list_{file_id})")
            self.pyStrings.append(printState.sep)
            self.pyStrings.append(f"{file_handle}.write(write_line)")

        # If printing on stdout, handle accordingly
        elif write_target == "outStream" and format_type == "runtime":
            self.pyStrings.append("output_fmt = list_output_formats([")
            for var in write_string.split(','):
                varMatch = re.match(r'^(.*?)\[\d+\]|^(.*?)[^\[]',var.strip())
                if varMatch:
                    var = varMatch.group(1)
                    self.pyStrings.append(f"\"{self.variableMap[var.strip()]}\",")
            self.pyStrings.append("])" + printState.sep)
            self.pyStrings.append("write_stream_obj = Format(output_fmt)" + printState.sep)
            self.pyStrings.append("write_line = write_stream_obj.write_line(write_list_stream)" + printState.sep)
            self.pyStrings.append("sys.stdout.write(write_line)")


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
        file_id = node["args"][0]["value"] if node["args"][0].get("value") else node["args"][0]["name"]
        self.pyStrings.append(f"file_{file_id}.close()")

    def get_python_source(self):
        return "".join(self.pyStrings)


def create_python_string(outputDict):
    code_generator = PythonCodeGenerator()
    code_generator.pyStrings.extend(
        [
            "import sys\n"
            "from typing import List\n",
            "import math\n",
            "from fortran_format import *",
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
