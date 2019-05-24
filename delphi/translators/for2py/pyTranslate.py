"""
Purpose:
    Convert a Fortran AST representation into a Python
    script having the same functionalities and performing
    the same operations as the original Fortran file.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:

        python pyTranslate.py -f <pickle_file> -g <python_file> -o <outputFileList>

pickle_file: Pickled file containing the ast representation of the Fortran
             file along with other non-source code information.
python_file: The Python file on which to write the resulting Python script.
"""

import sys
import pickle
import argparse
import re
from typing import Dict
from delphi.translators.for2py.format import list_data_type
from delphi.translators.for2py import For2PyError, syntax

###############################################################################
#                                                                             #
#                          FORTRAN-TO-PYTHON MAPPINGS                         #
#                                                                             #
###############################################################################

# TYPE_MAP gives the mapping from Fortran types to Python types
TYPE_MAP = {
    "character": "str",
    "double": "float",
    "float": "float",
    "int": "int",
    "integer": "int",
    "logical": "bool",
    "real": "float",
    "str": "str",
    "string": "str",
}

# OPERATOR_MAP gives the mapping from Fortran operators to Python operators
OPERATOR_MAP = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "**": "**",
    "<=": "<=",
    ">=": ">=",
    "==": "==",
    ".ne.": "!=",
    ".gt.": ">",
    ".eq.": "==",
    ".lt.": "<",
    ".le.": "<=",
    ".ge.": ">=",
    ".and.": "and",
    ".or.": "or",
}

# INTRINSICS_MAP gives the mapping from Fortran intrinsics to Python operators
# and functions.  Each entry in this map is of the form
#
#      fortran_fn : python_tgt
#
# where fortran_fn is the Fortran function; and python_tgt is the corresponding
# Python target, specified as a tuple (py_fn, fn_type, py_mod), where:
#            -- py_fn is a Python function or operator;
#            -- fn_type is one of: 'FUNC', 'INFIXOP'; and
#            -- py_mod is the module the Python function should be imported from,
#               None if no explicit import is necessary.

INTRINSICS_MAP = {
    "abs": ("abs", "FUNC", None),
    "acos": ("acos", "FUNC", "math"),
    "acosh": ("acosh", "FUNC", "math"),
    "asin": ("asin", "FUNC", "math"),
    "asinh": ("asinh", "FUNC", "math"),
    "atan": ("atan", "FUNC", "math"),
    "atanh": ("atanh", "FUNC", "math"),
    "ceiling": ("ceil", "FUNC", "math"),
    "cos": ("cos", "FUNC", "math"),
    "cosh": ("cosh", "FUNC", "math"),
    "erf": ("erf", "FUNC", "math"),
    "erfc": ("erfc", "FUNC", "math"),
    "exp": ("exp", "FUNC", "math"),
    "floor": ("floor", "FUNC", "math"),
    "gamma": ("gamma", "FUNC", "math"),
    "hypot": ("hypot", "FUNC", "math"),
    "index": None,
    "int": ("int", "FUNC", None),
    "isnan": ("isnan", "FUNC", "math"),
    "lge": (">=", "INFIXOP", None),  # lexical string comparison
    "lgt": (">", "INFIXOP", None),  # lexical string comparison
    "lle": ("<=", "INFIXOP", None),  # lexical string comparison
    "llt": ("<", "INFIXOP", None),  # lexical string comparison
    "log": ("log", "FUNC", "math"),
    "log10": ("log10", "FUNC", "math"),
    "log_gamma": ("lgamma", "FUNC", "math"),
    "max": ("max", "FUNC", None),
    "min": ("min", "FUNC", None),
    "mod": ("%", "INFIXOP", None),
    "modulo": ("%", "INFIXOP", None),
    "sin": ("sin", "FUNC", "math"),
    "sinh": ("sinh", "FUNC", "math"),
    "sqrt": ("sqrt", "FUNC", "math"),
    "tan": ("tan", "FUNC", "math"),
    "tanh": ("tanh", "FUNC", "math"),
    "xor": ("^", "INFIXOP", None),
}

###############################################################################
#                                                                             #
#                                 TRANSLATION                                 #
#                                                                             #
###############################################################################


class PrintState:
    def __init__(
        self,
        sep="\n",
        add="    ",
        printFirst=True,
        callSource=None,
        definedVars=[],
        globalVars=[],
        functionScope="",
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
            self.sep if sep is None else sep,
            self.add if add is None else add,
            self.printFirst if printFirst is None else printFirst,
            self.callSource if callSource is None else callSource,
            self.definedVars if definedVars is None else definedVars,
            self.globalVars if globalVars is None else globalVars,
            self.functionScope if functionScope is None else functionScope,
            self.indexRef if indexRef is None else indexRef,
            self.varTypes if varTypes is None else varTypes,
        )


class PythonCodeGenerator(object):
    def __init__(self):
        self.programName = ""
        self.printFn = {}
        self.variableMap = {}
        self.imports = []
        # This list contains the private functions
        self.privFunctions = []
        # This dictionary contains the mapping of symbol names to pythonic
        # names
        self.nameMapper = {}
        # Dictionary to hold functions and its arguments
        self.funcArgs = {}
        self.getframe_expr = "sys._getframe({}).f_code.co_name"
        self.pyStrings = []
        self.stateMap = {"UNKNOWN": "r", "REPLACE": "w"}
        self.format_dict = {}
        self.declaredDerivedTVars = []
        self.declaredDerivedTypes = []

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
        self.readFormat = []

    ###########################################################################
    #                                                                         #
    #                      TOP-LEVEL PROGRAM COMPONENTS                       #
    #                                                                         #
    ###########################################################################

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

    def printFunction(self, node, printState: PrintState):
        self.pyStrings.append(f"\ndef {self.nameMapper[node['name']]}(")
        args = []
        self.funcArgs[self.nameMapper[node["name"]]] = [
            self.nameMapper[x["name"]] for x in node["args"]
        ]
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
        if printState.sep != "\n":
            printState.sep = "\n"
        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                definedVars=args,
                indexRef=True,
                functionScope=self.nameMapper[node["name"]],
            ),
        )

    def printModule(self, node, printState: PrintState):
        self.pyStrings.append("\n")
        args = []
        self.printAst(
            node["body"],
            printState.copy(
                sep="", printFirst=True, definedVars=args, indexRef=True
            ),
        )

    def printProgram(self, node, printState: PrintState):
        self.printSubroutine(node, printState)
        self.programName = self.nameMapper[node["name"]]

    ###########################################################################
    #                                                                         #
    #                               EXPRESSIONS                               #
    #                                                                         #
    ###########################################################################

    def proc_intrinsic(self, node):
        """Processes calls to intrinsic functions and returns a string that is
           the corresponding Python code."""

        intrinsic = node["name"].lower()
        assert intrinsic in syntax.F_INTRINSICS

        try:
            py_fn, py_fn_type, py_mod = INTRINSICS_MAP[intrinsic]
        except KeyError:
            raise For2PyError(f"No handler for Fortran intrinsic {intrinsic}")

        arg_list = self.get_arg_list(node)
        arg_strs = [
            self.proc_expr(arg_list[i], False) for i in range(len(arg_list))
        ]

        if py_mod != None:
            handler = f"{py_mod}.{py_fn}"
        else:
            handler = py_fn

        if py_fn_type == "FUNC":
            arguments = ", ".join(arg_strs)
            return f"{handler}({arguments})"
        elif py_fn_type == "INFIXOP":
            assert len(arg_list) == 2, f"INFIXOP with {len(arglist)} arguments"
            return f"({arg_strs[0]} {py_fn} {arg_strs[1]})"
        else:
            assert False, f"Unknown py_fn_type: {py_fn_type}"

    def get_arg_list(self, node):
        """Get_arg_list() returns the list of arguments or subscripts at a node.
           If there are no arguments or subscripts, it returns the empty list."""

        if "args" in node:
            return node["args"]

        if "subscripts" in node:
            return node["subscripts"]

        return []

    def proc_call(self, node):
        """Processes function calls, including calls to intrinsics, and returns
           a string that is the corresponding Python code.  This code assumes
           that proc_expr() has used type info to correctly identify array
           references, and that proc_call() is therefore correctly called only
           on function calls."""

        if node["name"].lower() == "index":
            var = self.nameMapper[node["args"][0]["name"]]
            toFind = node["args"][1]["value"]
            return f"{var}[0].find({toFind})"

        if node["name"].lower() in syntax.F_INTRINSICS:
            return self.proc_intrinsic(node)

        callee = self.nameMapper[f"{node['name']}"]
        args = self.get_arg_list(node)
        arg_strs = [self.proc_expr(args[i], True) for i in range(len(args))]

        # Case where a call is a print method
        if callee == "print":
            arguments = self.proc_print(arg_strs)
        else:
            arguments = ", ".join(arg_strs)
        exp_str = f"{callee}({arguments})"

        return exp_str

    def proc_print(self, arg_strs):
        arguments = ""
        for idx in range(0, len(arg_strs)):
            if self.check_var_name(arg_strs[idx]):
                arguments += f"{arg_strs[idx]}"
            else:
                arguments += f'"{arg_strs[idx]}"'
            if idx < len(arg_strs) - 1:
                arguments += ", "
        return arguments

    def check_var_name(self, name):
        if name.isalnum():
            return True
        elif "-" not in name and "_" not in name:
            return False
        return True

    def proc_literal(self, node):
        """Processes a literal value and returns a string that is the
           corresponding Python code."""

        if node["type"] == "bool":
            return node["value"].title()
        else:
            return node["value"]

    def proc_ref(self, node, wrapper):
        """Processes a reference node and returns a string that is the
           corresponding Python code.  The argument "wrapper" indicates whether
           or not the Python expression should refer to the list wrapper for
           (scalar) variables."""
        ref_str = ""
        is_derived_type_ref = False
        if (
            "is_derived_type_ref" in node
            and node["is_derived_type_ref"] == "true"
        ):
            ref_str = self.get_derived_type_ref(
                node, int(node["numPartRef"]), False
            )
            is_derived_type_ref = True
        else:
            ref_str = self.nameMapper[node["name"]]

        if "subscripts" in node:
            # array reference or function call
            if "is_array" in node and node["is_array"] == "true":
                subs = node["subscripts"]
                subs_strs = [
                    self.proc_expr(subs[i], False) for i in range(len(subs))
                ]
                subscripts = ", ".join(subs_strs)
                expr_str = f"{ref_str}.get_(({subscripts}))"
            else:
                expr_str = self.proc_call(node)
        else:
            # scalar variable
            if wrapper:
                expr_str = ref_str
            else:
                if (
                    "is_arg" in node
                    and node["is_arg"] == "true"
                    or is_derived_type_ref
                ):
                    expr_str = ref_str
                    is_derived_type_ref = False
                else:
                    if ref_str in self.declaredDerivedTVars:
                        expr_str = ref_str
                    else:
                        expr_str = ref_str + "[0]"

        return expr_str

    def proc_op(self, node):
        """Processes expressions involving operators and returns a string that
           is the corresponding Python code."""
        try:
            op_str = OPERATOR_MAP[node["operator"].lower()]
        except KeyError:
            raise For2PyError(f"unhndled operator {node['operator']}")

        assert len(node["left"]) == 1
        l_subexpr = self.proc_expr(node["left"][0], False)

        if "right" in node:
            # binary operator
            assert len(node["right"]) == 1
            r_subexpr = self.proc_expr(node["right"][0], False)
            expr_str = f"({l_subexpr} {op_str} {r_subexpr})"
        else:
            # unary operator
            expr_str = f"{op_str}({l_subexpr})"

        return expr_str

    def proc_expr(self, node, wrapper):
        """Processes an expression node and returns a string that is the
        corresponding Python code. The argument "wrapper" indicates whether or
        not the Python expression should refer to the list wrapper for (scalar)
        variables."""

        if node["tag"] == "literal":
            return self.proc_literal(node)

        if node["tag"] == "ref":
            # variable or array reference
            return self.proc_ref(node, wrapper)

        if node["tag"] == "call":
            # function call
            return self.proc_call(node)

        expr_str = None
        if node["tag"] == "op":
            # operator
            assert not wrapper
            expr_str = self.proc_op(node)

        assert expr_str != None, f">>> [proc_expr] NULL value: {node}"
        return expr_str

    def printCall(self, node: Dict[str, str], printState: PrintState):
        call_str = self.proc_call(node)
        self.pyStrings.append(call_str)
        return

    def printAst(self, root, printState: PrintState):
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

    def printArg(self, node, printState: PrintState):
        try:
            var_type = TYPE_MAP[node["type"].lower()]
        except KeyError:
            raise For2PyError(f"unrecognized type {node['type']}")

        arg_name = self.nameMapper[node["name"]]
        if "is_array" in node and node["is_array"] == "true":
            self.pyStrings.append(f"{arg_name}")
        else:
            self.pyStrings.append(f"{arg_name}: List[{var_type}]")
        printState.definedVars += [arg_name]

    ###########################################################################
    #                                                                         #
    #                                STATEMENTS                               #
    #                                                                         #
    ###########################################################################

    def printDo(self, node, printState: PrintState):
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

    def printDoWhile(self, node, printState: PrintState):
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

    def printIndex(self, node, printState: PrintState):
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
                printState.copy(
                    sep="", add="", printFirst=True, indexRef=True
                ),
            )
            self.pyStrings.append(")")
        else:
            self.pyStrings.append("+1)")

    def printIf(self, node, printState: PrintState):
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

    def printOp(self, node, printState: PrintState):
        expr_str = self.proc_expr(node, False)
        self.pyStrings.append(expr_str)

    def printLiteral(self, node, printState: PrintState):
        expr_str = self.proc_literal(node)
        self.pyStrings.append(expr_str)

    def printRef(self, node, printState: PrintState):
        ref_str = self.proc_ref(node, False)
        self.pyStrings.append(ref_str)

    def printAssignment(self, node, printState: PrintState):
        assert len(node["target"]) == 1 and len(node["value"]) == 1
        lhs, rhs = node["target"][0], node["value"][0]

        rhs_str = self.proc_expr(node["value"][0], False)

        if lhs["is_derived_type_ref"] == "true":
            assg_str = self.get_derived_type_ref(
                lhs, int(lhs["numPartRef"]), True
            )
        else:
            if lhs["hasSubscripts"] == "true":
                assert (
                    "subscripts" in lhs
                ), "lhs 'hasSubscripts' and actual 'subscripts' existence does not match. Fix 'hasSubscripts' in rectify.py."
                # target is an array element
                if "is_array" in lhs and lhs["is_array"] == "true":
                    subs = lhs["subscripts"]
                    subs_strs = [
                        self.proc_expr(subs[i], False)
                        for i in range(len(subs))
                    ]
                    subscripts = ", ".join(subs_strs)
                    assg_str = f"{lhs['name']}.set_(({subscripts}), "
                else:
                    # handling derived types might go here
                    assert False
            else:
                # target is a scalar variable
                assg_str = f"{lhs['name']}[0]"

        if "set_" in assg_str:
            assg_str += f"{rhs_str})"
        else:
            assg_str += f" = {rhs_str}"

        self.pyStrings.append(assg_str)
        return

    def printUse(self, node, printState: PrintState):
        if node.get("include"):
            self.imports.append(
                f"from delphi.translators.for2py.m_{node['arg'].lower()} "
                f"import {', '.join(node['include'])}\n"
            )
        else:
            self.imports.append(
                f"from delphi.translators.for2py.m_{node['arg'].lower()} import *\n"
            )

    def printFuncReturn(self, node, printState: PrintState):
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

    def printExit(self, node, printState: PrintState):
        if node.get("value"):
            self.pyStrings.append(f"print({node['value']})")
            self.pyStrings.append(printState.sep)
        self.pyStrings.append("return")

    def printReturn(self, node, printState: PrintState):
        self.pyStrings.append("")

    def printOpen(self, node, printState: PrintState):
        if node["args"][0].get("arg_name") == "UNIT":
            file_handle = "file_" + str(node["args"][1]["value"])
        elif node["args"][0].get("tag") == "ref":
            file_handle = "file_" + str(
                self.nameMapper[node["args"][0]["name"]]
            )
        else:
            file_handle = "file_" + str(node["args"][0]["value"])
        self.pyStrings.append(f"{file_handle} = ")
        for index, item in enumerate(node["args"]):
            if item.get("arg_name"):
                if item["arg_name"] == "FILE":
                    file_name = node["args"][index + 1]["value"]
                    open_state = "r"
                elif item["arg_name"] == "STATUS":
                    open_state = node["args"][index + 1]["value"]
                    open_state = self.stateMap[open_state]

        self.pyStrings.append(f'open("{file_name}", "{open_state}")')

    def printRead(self, node, printState: PrintState):
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
            f") = format_{format_label}_obj."
            f"read_line({file_handle}.readline())"
        )
        self.pyStrings.append(printState.sep)

        if isArray:
            tempInd = 0  # Re-initialize to zero for array index
            for item in node["args"]:
                if item["tag"] == "ref":
                    var = self.nameMapper[item["name"]]
                    if "subscripts" in item:
                        self.pyStrings.append(f"{var}.set_((")
                        self.printAst(
                            item["subscripts"],
                            printState.copy(
                                sep=", ",
                                add="",
                                printFirst=False,
                                indexRef=True,
                            ),
                        )
                        self.pyStrings.append(f"), tempVar[{tempInd}])")
                        tempInd = tempInd + 1
                        self.pyStrings.append(printState.sep)
                ind = ind + 1

    def printWrite(self, node, printState: PrintState):
        write_string = ""
        # Check whether write to file or output stream
        if node["args"][0]["value"] == "*":
            write_target = "outStream"
        else:
            write_target = "file"
            if node["args"][0]["value"]:
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
            self.pyStrings.append(f"write_list_{file_id} = ")
        elif write_target == "outStream":
            self.pyStrings.append(f"write_list_stream = ")

        # Collect the expressions to be written out.  The first two arguments to
        # a WRITE statement are the output stream and the format, so these are
        # skipped.
        args = node["args"][2:]
        args_str = []
        for i in range(len(args)):
            if (
                "is_derived_type_ref" in args[i]
                and args[i]["is_derived_type_ref"] == "true"
            ):
                args_str.append(
                    self.get_derived_type_ref(
                        args[i], int(args[i]["numPartRef"]), False
                    )
                )
            else:
                args_str.append(self.proc_expr(args[i], False))

        write_string = ", ".join(args_str)
        self.pyStrings.append(f"[{write_string}]")
        self.pyStrings.append(printState.sep)

        # If format specified and output in a file, execute write_line on file
        # handler
        if write_target == "file":
            if format_type == "specifier":
                self.pyStrings.append(
                    f"write_line = format_{format_label}_obj."
                    f"write_line(write_list_{file_id})"
                )
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"{file_handle}.write(write_line)")
            elif format_type == "runtime":
                self.pyStrings.append("output_fmt = list_output_formats([")
                for var in write_string.split(","):
                    varMatch = re.match(
                        r"^(.*?)\[\d+\]|^(.*?)[^\[]", var.strip()
                    )
                    if varMatch:
                        var = varMatch.group(1)
                        self.pyStrings.append(
                            f'"{self.variableMap[var.strip()]}",'
                        )
                self.pyStrings.append("])" + printState.sep)
                self.pyStrings.append(
                    "write_stream_obj = Format(output_fmt)" + printState.sep
                )
                self.pyStrings.append(
                    "write_line = write_stream_obj."
                    f"write_line(write_list_{file_id})"
                )
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"{file_handle}.write(write_line)")

        # If printing on stdout, handle accordingly
        elif write_target == "outStream":
            if format_type == "runtime":
                self.pyStrings.append("output_fmt = list_output_formats([")
                for var in write_string.split(","):
                    varMatch = re.match(
                        r"^(.*?)\[\d+\]|^(.*?)[^\[]", var.strip()
                    )
                    if varMatch:
                        var = varMatch.group(1)
                        self.pyStrings.append(
                            f'"{self.variableMap[var.strip()]}",'
                        )

                self.pyStrings.append("])" + printState.sep)
                self.pyStrings.append(
                    "write_stream_obj = Format(output_fmt)" + printState.sep
                )
                self.pyStrings.append(
                    "write_line = write_stream_obj."
                    + "write_line(write_list_stream)"
                    + printState.sep
                )
                self.pyStrings.append("sys.stdout.write(write_line)")
            elif format_type == "specifier":
                self.pyStrings.append(
                    f"write_line = format_{format_label}_obj."
                    "write_line(write_list_stream)"
                )
                self.pyStrings.append(printState.sep)
                self.pyStrings.append(f"sys.stdout.write(write_line)")

    def printFormat(self, node, printState: PrintState):
        type_list = []
        temp_list = []
        _re_int = re.compile(r"^\d+$")
        format_list = [token["value"] for token in node["args"]]

        for token in format_list:
            if not _re_int.match(token):
                temp_list.append(token)
            else:
                type_list.append(f"{token}({','.join(temp_list)})")
                temp_list = []
        if len(type_list) == 0:
            type_list = temp_list

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

    def printClose(self, node, printState: PrintState):
        file_id = (
            node["args"][0]["value"]
            if node["args"][0].get("value")
            else self.nameMapper[node["args"][0]["name"]]
        )
        self.pyStrings.append(f"file_{file_id}.close()")

    ###########################################################################
    #                                                                         #
    #                              DECLARATIONS                               #
    #                                                                         #
    ###########################################################################

    def printVariable(self, node, printState: PrintState):
        var_name = self.nameMapper[node["name"]]
        if var_name not in printState.definedVars + printState.globalVars:
            printState.definedVars += [var_name]
            if node.get("value"):
                initVal = node["value"][0]["value"]
            else:
                initVal = None

            varType = self.get_type(node)

            if "is_derived_type" in node and node["is_derived_type"] == "true":
                self.pyStrings.append(
                    f"{self.nameMapper[node['name']]} =  {varType}()"
                )
                self.declaredDerivedTVars.append(node["name"])
            else:
                if printState.functionScope:
                    if not var_name in self.funcArgs.get(
                        printState.functionScope
                    ):
                        self.pyStrings.append(
                            f"{var_name}: List[{varType}]" f" = [{initVal}]"
                        )
                    else:
                        self.pyStrings.append(f"{var_name}: List[{varType}]")
                else:
                    self.pyStrings.append(
                        f"{var_name}: List[{varType}] = " f"[{initVal}]"
                    )

            # The code below might cause issues on unexpected places.
            # If weird variable declarations appear, check code below

            if not printState.sep:
                printState.sep = "\n"
            self.variableMap[self.nameMapper[node["name"]]] = node["type"]
        else:
            printState.printFirst = False

    def printArray(self, node, printState: PrintState):
        """ Prints out the array declaration in a format of Array class
            object declaration. 'arrayName = Array(Type, [bounds])'
        """
        if (
            self.nameMapper[node["name"]] not in printState.definedVars
            and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            printState.definedVars += [self.nameMapper[node["name"]]]
            printState.definedVars += [node["name"]]

            var_type = self.get_type(node)

            array_range = self.get_array_dimension(node)

            self.pyStrings.append(
                f"{node['name']} = Array({var_type}, [{array_range}])"
            )

    def printDerivedType(self, node, printState: PrintState):
        derived_type_class_info = node[0]
        derived_type_variables = derived_type_class_info["derived-types"]
        num_of_variables = len(derived_type_variables)

        self.pyStrings.append(printState.sep)
        self.pyStrings.append(f"class {derived_type_class_info['type']}:\n")
        # For a record, store the declared derived type names
        self.declaredDerivedTypes.append(derived_type_class_info["type"])

        self.pyStrings.append("    def __init__(self):\n")
        for var in range(num_of_variables):
            name = derived_type_variables[var]["name"]

            # Retrieve the type of member variables and check its type
            var_type = self.get_type(derived_type_variables[var])
            is_derived_type_declaration = False
            # If the type is not one of the default types, but it's a declared derived type,
            # set the is_derived_type_declaration to True as the declaration of variable
            # with such type has different declaration syntax from the default type variables.
            if (
                var_type not in TYPE_MAP
                and var_type in self.declaredDerivedTypes
            ):
                is_derived_type_declaration = True

            if derived_type_variables[var]["is_array"] == "false":
                if not is_derived_type_declaration:
                    self.pyStrings.append(f"        self.{name} : {var_type}")
                else:
                    self.pyStrings.append(
                        f"        self.{name} = {var_type}()"
                    )

                if "value" in derived_type_variables[var]:
                    value = self.proc_literal(
                        derived_type_variables[var]["value"][0]
                    )
                    self.pyStrings.append(f" = {value}")
            else:
                array_range = self.get_array_dimension(
                    derived_type_variables[var]
                )
                self.pyStrings.append(
                    f"        self.{name} = Array({var_type}, [{array_range}])"
                )
            self.pyStrings.append(printState.sep)

    ###########################################################################
    #                                                                         #
    #                              MISCELLANEOUS                              #
    #                                                                         #
    ###########################################################################

    def initializeFileVars(self, node, printState: PrintState):
        label = node["args"][1]["value"]
        data_type = list_data_type(self.format_dict[label])
        index = 0
        for item in node["args"]:
            if item["tag"] == "ref":
                self.printVariable(
                    {
                        "name": self.nameMapper[item["name"]],
                        "type": data_type[index],
                    },
                    printState,
                )
                self.pyStrings.append(printState.sep)
                index += 1

    def nameMapping(self, ast):
        for item in ast:
            if item.get("name"):
                self.nameMapper[item["name"]] = item["name"]
            for inner in item:
                if isinstance(item[inner], list):
                    self.nameMapping(item[inner])

    def get_python_source(self):
        imports = "".join(self.imports)
        if len(imports) != 0:
            self.pyStrings.insert(1, imports)
        if self.programName != "":
            self.pyStrings.append(f"\n\n{self.programName}()\n")

        return "".join(self.pyStrings)


    def get_type(self, node):
        """ This function checks the type of a variable and returns the appropriate
        Python syntax type name. """

        variable_type = node["type"].lower()
        if variable_type in TYPE_MAP:
            return TYPE_MAP[variable_type]
        else:
            if node["is_derived_type"] == "true":
                return variable_type
            else:
                assert False, f"Unrecognized variable type: {variable_type}"


    def get_range(self, node):
        """ This function will construct the range string in 'loBound, Upbound'
        format and return to the called function. """
        loBound = "0"
        upBound = "0"

        low = node["low"]
        up = node["high"]
        # Get lower bound value
        if low[0]["tag"] == "literal":
            loBound = self.proc_literal(low[0])
        elif low[0]["tag"] == "op":
            loBound = self.proc_op(low[0])
        else:
            assert False, f"Unrecognized tag in upper bound: {low[0]['tag']}"

        # Get upper bound value
        if up[0]["tag"] == "literal":
            upBound = self.proc_literal(up[0])
        elif up[0]["tag"] == "op":
            upBound = self.proc_op(up[0])
        else:
            assert False, f"Unrecognized tag in upper bound: {up[0]['tag']}"

        return f"{loBound}, {upBound}"


    def get_array_dimension(self, node):
        """ This function is for extracting the dimensions' range information
        from the AST.  This function is needed for handling a multi-dimensional
        array(s). """

        count = 1
        array_range = ""
        for dimension in node["dimensions"]:
            if (
                "literal" in dimension
            ):  # A case where no explicit low bound set
                upBound = self.proc_literal(dimension["literal"][0])
                array_range += f"(0, {upBound})"
            elif (
                "range" in dimension
            ):  # A case where explicit low and up bounds are set
                array_range += f"({self.get_range(dimension['range'][0])})"
            else:
                assert (
                    False
                ), f"Array range case not handled. Reference node content: {node}"

            if count < int(node["count"]):
                array_range += ", "
                count += 1

        return array_range


    def get_derived_type_ref(self, node, numPartRef, is_assignment):
        """ This function forms a derived type reference and return to the
        caller """

        ref = ""
        if node["hasSubscripts"] == "true":
            subscript = node["subscripts"][0]
            if subscript["tag"] == "ref":
                index = f"{subscript['name']}[0]"
            else:
                index = subscript["value"]

            if numPartRef > 1 or not is_assignment:
                ref += f"{node['name']}.get_({index})"
            else:
                ref += f"{node['name']}.set_({index}, "
        else:
            ref += node["name"]
        numPartRef -= 1
        if "ref" in node:
            ref += f".{self.get_derived_type_ref(node['ref'][0], numPartRef, is_assignment)}"
        return ref


def index_modules(root) -> Dict:
    """ Counts the number of modules in the Fortran file including the program
    file. Each module is written out into a separate Python file.  """

    module_index_dict = {
        node["name"]: (node.get("tag"), index)
        for index, node in enumerate(root)
        if node.get("tag") in ("module", "program", "subroutine")
    }

    return module_index_dict


def create_python_source_list(outputDict: Dict):
    module_index_dict = index_modules(outputDict["ast"])
    py_sourcelist = []
    main_ast = []
    import_lines = [
        "import sys",
        "from typing import List",
        "import math",
        "from delphi.translators.for2py.format import *",
        "from delphi.translators.for2py.arrays import *",
        "from dataclasses import dataclass\n",
    ]

    for module in module_index_dict:
        if "module" in module_index_dict[module]:
            ast = [outputDict["ast"][module_index_dict[module][1]]]
        else:
            main_ast.append(outputDict["ast"][module_index_dict[module][1]])
            continue
        code_generator = PythonCodeGenerator()
        code_generator.pyStrings.append("\n".join(import_lines))

        # Fill the name mapper dictionary
        code_generator.nameMapping(ast)
        code_generator.printAst(ast, PrintState())
        py_sourcelist.append(
            (
                code_generator.get_python_source(),
                module,
                module_index_dict[module][0],
            )
        )

    # Writing the main program section
    code_generator = PythonCodeGenerator()
    code_generator.pyStrings.append("\n".join(import_lines))

    # Copy the derived type ast from the main_ast into the separate list,
    # so it can be printed outside (above) the main method
    derived_type_ast = []
    for index in list(main_ast[0]["body"]):
        if "is_derived_type" in index and index["is_derived_type"] == "true":
            if "tag" not in index:
                derived_type_ast.append(index)
                main_ast[0]["body"].remove(index)

    # Print derived type declaration(s)
    if derived_type_ast:
        code_generator.pyStrings.append("@dataclass\n")
        for i in range(len(derived_type_ast)):
            assert (
                derived_type_ast[i]["is_derived_type"] == "true"
            ), "[derived_type_ast] holds non-derived type ast"
            code_generator.nameMapping([derived_type_ast[i]])
            code_generator.printDerivedType(
                [derived_type_ast[i]], PrintState()
            )

    code_generator.nameMapping(main_ast)
    code_generator.printAst(main_ast, PrintState())
    py_sourcelist.append(
        (code_generator.get_python_source(), main_ast, "program")
    )

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
        help=(
            "Pickled version of the asts together with non-source code"
            "information"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        nargs="+",
        help="Text file containing the list of output python files being generated",
    )
    args = parser.parse_args(sys.argv[1:])
    with open(args.files[0], "rb") as f:
        outputDict = pickle.load(f)
    python_source_list = create_python_source_list(outputDict)
    outputList = []
    for item in python_source_list:
        if item[2] == "module":
            with open(f"m_{item[1].lower()}.py", "w") as f:
                outputList.append("m_" + item[1].lower() + ".py")
                f.write(item[0])
        else:
            with open(args.gen[0], "w") as f:
                outputList.append(args.gen[0])
                f.write(item[0])

    with open(args.out[0], "w") as f:
        for fileName in outputList:
            f.write(fileName + " ")
