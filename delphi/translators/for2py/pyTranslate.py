"""
Purpose:
    Convert a Fortran AST representation into a Python
    script having the same functionalities and performing
    the same operations as the original Fortran file.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        python pyTranslate.py -f <pickle_file> -g <python_file>

pickle_file: Pickled file containing the ast representation of the Fortran
file along with other non-source code information.

python_file: The Python file on which to write the resulting python script.
"""

import sys
import pickle
import argparse
import re
from typing import Dict
from delphi.translators.for2py.format import list_data_type
from delphi.translators.for2py import For2PyError
from delphi.translators.for2py import syntax

###############################################################################
#                                                                             #
#                          FORTRAN-TO-PYTHON MAPPINGS                         #
#                                                                             #
###############################################################################

# TYPE_MAP gives the mapping from Fortran types to Python types
TYPE_MAP = {
    "character" : "str",
    "double" : "float",
    "integer" : "int",
    "logical" : "bool",
    "real" : "float"
}

# OPERATOR_MAP gives the mapping from Fortran operators to Python operators
OPERATOR_MAP = {
    "+" : "+",
    "-" : "-",
    "*" : "*",
    "/" : "/",
    "**" : "**",
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
    'abs' : ('abs', 'FUNC', None),
    'acos' : ('acos', 'FUNC', 'math'),
    'acosh' : ('acosh', 'FUNC', 'math'),
    'asin' : ('asin', 'FUNC', 'math'),
    'asinh' : ('asinh', 'FUNC', 'math'),
    'atan' : ('atan', 'FUNC', 'math'),
    'atanh' : ('atanh', 'FUNC', 'math'),
    'ceiling' : ('ceil', 'FUNC', 'math'),
    'cos' : ('cos', 'FUNC', 'math'),
    'cosh' : ('cosh', 'FUNC', 'math'),
    'erf' : ('erf', 'FUNC', 'math'),
    'erfc' : ('erfc', 'FUNC', 'math'),
    'exp' : ('exp', 'FUNC', 'math'),
    'floor' : ('floor', 'FUNC', 'math'),
    'gamma' : ('gamma', 'FUNC', 'math'),
    'hypot' : ('hypot', 'FUNC', 'math'),
    'index' : None,
    'int' : ('int', 'FUNC', None),
    'isnan' : ('isnan', 'FUNC', 'math'),
    'lge' : ('>=', 'INFIXOP', None),    # lexical string comparison
    'lgt' : ('>', 'INFIXOP', None),     # lexical string comparison
    'lle' : ('<=', 'INFIXOP', None),    # lexical string comparison
    'llt' : ('<', 'INFIXOP', None),     # lexical string comparison
    'log' : ('log', 'FUNC', 'math'),
    'log10' : ('log10', 'FUNC', 'math'),
    'log_gamma' : ('lgamma', 'FUNC', 'math'),
    'max' : ('max', 'FUNC', None),
    'min' : ('min', 'FUNC', None),
    'mod' : ('%', 'INFIXOP', None),
    'modulo' : ('%', 'INFIXOP', None),
    'sin' : ('sin', 'FUNC', 'math'),
    'sinh' : ('sinh', 'FUNC', 'math'),
    'sqrt' : ('sqrt', 'FUNC', 'math'),
    'tan' : ('tan', 'FUNC', 'math'),
    'tanh' : ('tanh', 'FUNC', 'math'),
    'xor' : ('^', 'INFIXOP', None)
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
        self.imports = []
        # This list contains the private functions
        self.privFunctions = []
        # This dictionary contains the mapping of symbol names to pythonic
        # names
        self.nameMapper = {}
        # Dictionary to hold functions and its arguments
        self.funcArgs = {}
        self.mathFuncs = [
            "exp",
            "cexp",
            "cmplx",
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

    # proc_intrinsic() processes a call to an intrinsic function
    def proc_intrinsic(self, node):
        intrinsic = node["name"].lower()
        assert intrinsic in syntax.F_INTRINSICS

        try:
            py_fn, py_fn_type, py_mod = INTRINSICS_MAP[intrinsic]
        except KeyError:
            raise For2PyError(f"No handler for Fortran intrinsic {intrinsic}")

        arg_list = self.get_arg_list(node)
        arg_strs = self.proc_expr_list(arg_list, False)

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

    # get_arg_list() returns the list of arguments or subscripts at a node
    def get_arg_list(self, node):
        if "args" in node:
            return node["args"]

        if "subscripts" in node:
            return node["subscripts"]

        return []

    # proc_call(): processes function calls, including calls to intrinsics.
    # This code assumes that proc_expr() has used type info to correctly
    # identify array references, and that proc_call() is therefore correctly
    # called only on function calls.
    def proc_call(self, node):
        if node["name"].lower() == "index":
            var = self.nameMapper[node["args"][0]["name"]]
            toFind = node["args"][1]["value"]
            return f"{var}[0].find({toFind})"

        if node["name"].lower() in syntax.F_INTRINSICS:
            return self.proc_intrinsic(node)

        callee = self.nameMapper[f"{node['name']}"]
        arg_list = self.get_arg_list(node)
        arg_strs = self.proc_expr_list(arg_list, True)
        arguments = ", ".join(arg_strs)

        exp_str = f"{callee}({arguments})"
        return exp_str

    # proc_expr_list() processes a list of expressions [e1, ..., eN] and
    # returns a list of strings [s1, ..., sN] where si is the string obtained
    # from processing expression ei.  The argument wrapper indicates
    # whether the Python expression should refer to the list wrapper for 
    # (scalar) variables.
    def proc_expr_list(self, node, wrapper):
        return [self.proc_expr(node[i], wrapper) for i in range(len(node))]

    # proc_literal() processes a literal value.
    def proc_literal(self, node):
        if node["type"] == "bool":
            return node["value"].title()
        else:
            return node["value"]

    # proc_ref() processes a reference node.  The argument "wrapper" 
    # indicates whether or not the Python expression should refer to the list
    # wrapper for (scalar) variables.
    def proc_ref(self, node, wrapper):
        ref_str = self.nameMapper[node["name"]]
        if "subscripts" in node:
            # array reference or function call
            if "isArray" in node and node["isArray"]:
                subs_strs = self.proc_expr_list(node["subscripts"], False)
                subscripts = ", ".join(subs_strs)
                expr_str = f"{ref_str}.get_(({subscripts}))"
            else:
                expr_str = self.proc_call(node)
        else:
            # scalar variable
            if wrapper:
                expr_str = ref_str
            else:
                expr_str = ref_str + "[0]"

        return expr_str

    # proc_op() processes expressions involving operators.
    def proc_op(self, node):
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


    # proc_expr() processes an expression node.  The argument "wrapper" 
    # indicates whether or not the Python expression should refer to the list
    # wrapper for (scalar) variables.
    def proc_expr(self, node, wrapper):
        #print(f">>> proc_expr: {node}")

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

        #print(f">>> NODE = {node}\n    exp_str = {expr_str}")

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

    def printArg(self, node, printState: PrintState):
        try:
            var_type = TYPE_MAP[node["type"].lower()]
        except KeyError:
            raise For2PyError(f"unrecognized type {node['type']}")

        if node["arg_type"] == "arg_array":
            self.pyStrings.append(f"{self.nameMapper[node['name']]}")
        else:
            self.pyStrings.append(
                f"{self.nameMapper[node['name']]}: List[{var_type}]"
            )
        printState.definedVars += [self.nameMapper[node["name"]]]

    def printVariable(self, node, printState: PrintState):
        initial_set = False
        if (
            self.nameMapper[node["name"]] not in printState.definedVars
            and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            printState.definedVars += [self.nameMapper[node["name"]]]
            if node.get("value"):
                init_val = node["value"][0]["value"]
                initial_set = True

            if node["type"].upper() in ("INTEGER", "INT"):
                initVal = init_val if initial_set else 0
                varType = "int"
            elif node["type"].upper() in ("DOUBLE", "REAL", "FLOAT"):
                initVal = init_val if initial_set else 0.0
                varType = "float"
            elif node["type"].upper() in ("STRING", "CHARACTER", "STR"):
                initVal = init_val if initial_set else ""
                varType = "str"
            elif node["type"].upper() == "LOGICAL":
                initVal = init_val if initial_set else False
                varType = "bool"
            else:
                if node["isDevTypeVar"]:
                    initVal = init_val if initial_set else 0
                    varType = node["type"]
                else:
                    raise For2PyError(f"unrecognized type {node['type']}")

            if "isDevTypeVar" in node and node["isDevTypeVar"]:
                self.pyStrings.append(
                    f"{self.nameMapper[node['name']]} =  {varType}()"
                )
            else:
                if printState.functionScope:
                    if not self.nameMapper[node["name"]] in self.funcArgs.get(
                        printState.functionScope
                    ):
                        self.pyStrings.append(
                            f"{self.nameMapper[node['name']]}: List[{varType}]"
                            f" = [{initVal}]"
                        )
                    else:
                        self.pyStrings.append(
                            f"{self.nameMapper[node['name']]}: List[{varType}]"
                        )
                else:
                    self.pyStrings.append(
                        f"{self.nameMapper[node['name']]}: List[{varType}] = "
                        f"[{initVal}]"
                    )

            # The code below might cause issues on unexpected places.
            # If weird variable declarations appear, check code below

            if not printState.sep:
                printState.sep = "\n"
            self.variableMap[self.nameMapper[node["name"]]] = node["type"]
        else:
            printState.printFirst = False

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
                                        
        if "subscripts" in lhs:
            # target is an array element or derived type field
            if "isArray" in lhs and lhs["isArray"]:
                subs_strs = self.proc_expr_list(lhs["subscripts"], False)
                subs = ", ".join(subs_strs)
                assg_str = f"{lhs['name']}.set_(({subs}), {rhs_str})"
            else:
                # handling derived types might go here
                assert False
        else:
            # target is a scalar variable
            assg_str = f"{lhs['name']}[0] = {rhs_str}"

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
                    file_name = node["args"][index + 1]["value"][1:-1]
                    open_state = "r"
                elif item["arg_name"] == "STATUS":
                    open_state = node["args"][index + 1]["value"][1:-1]
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
                # Handles array or a variable that holds following attributes,
                # such as var.x.y
                if "subscripts" in item:
                    # If a variable is derived type
                    if item["isDevType"]:
                        # If hasSubscripts is true, the derived type object is
                        # also an array and has following subscripts.
                        # Therefore, in order to handle the syntax
                        # obj.get_((obj_index)).field.get_((field)), the code
                        # below handles obj.get_((obj_index)) first before the
                        # subscripts for the field variables
                        if "hasSubscripts" in item and item["hasSubscripts"]:
                            devObjSub = [item["subscripts"].pop(0)]
                            write_string += ".get_(("
                            for sub in devObjSub:
                                if sub["tag"] == "ref":
                                    write_string += f"{sub['name']}[0]"
                                elif sub["tag"] == "literal":
                                    write_string += f"{sub['value']}"
                            write_string += "))"
                        # This is a case where the very first variable is not
                        # an array, but has subscripts, which holds information
                        # of following derived type field variables. i.e.
                        # var.x.y HOWEVER, the code below MUST be revisted and
                        # fixed. This is a hack of hard coding in order to
                        # print a format of var.x.y fixed number of 3 times In
                        # order to fix this, the entire printWrite should be
                        # modified that can do a recursion or break passed
                        # lists from the translate.py to have more clear
                        # groups.  It cannot be handled now as it may cause a
                        # butterfly effect on all other I/O handling
                        if (
                            "hasSubscripts" in item
                            and not item["hasSubscripts"]
                            and "subscripts" in item
                            and "arrayStat" not in item
                        ):
                            isubs = item["subscripts"]
                            isubs2 = isubs[1]["subscripts"]
                            write_string += "".join(
                                (
                                    f".{isubs[0]['name']}",
                                    f".{isubs[0]['field-name']}",
                                    ", ",
                                    ".".join(
                                        (
                                            isubs[1]["name"],
                                            isubs2[0]["name"],
                                            isubs2[0]["field-name"],
                                        )
                                    ),
                                    ", ",
                                    ".".join(
                                        (
                                            isubs2[1]["name"],
                                            isubs2[1]["subscripts"][0]["name"],
                                            isubs2[1]["subscripts"][0][
                                                "field-name"
                                            ],
                                        )
                                    ),
                                )
                            )
                        if "field-name" in item:
                            write_string += f".{item['field-name']}"
                    # Handling array
                    if (
                        "hasSubscripts" in item
                        and item["hasSubscripts"]
                        and "isArray" in item
                        and item["isArray"]
                    ) or (
                        "arrayStat" in item and item["arrayStat"] == "isArray"
                    ):
                        i = 0
                        write_string += ".get_(("
                        for ind in item["subscripts"]:
                            # When an array uses another array's value as its
                            # index value
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
                                    write_string += (
                                        f"{ind['left'][0]['name']}[0] "
                                    )
                                else:
                                    assert ind["left"][0]["tag"] == "literal"
                                    write_string += (
                                        f"{ind['left'][0]['value']} "
                                    )

                                if "right" in ind:
                                    write_string += f"{ind['operator']} "

                                    if ind["right"][0]["tag"] == "ref":
                                        write_string += (
                                            f"{ind['right'][0]['name']} "
                                        )
                                    else:
                                        assert (
                                            ind["right"][0]["tag"] == "literal"
                                        )
                                        write_string += (
                                            f"{ind['right'][0]['value']} "
                                        )
                            else:
                                if ind["tag"] == "ref":
                                    write_string += f"{ind['name']}[0]"
                                elif ind["tag"] == "op":
                                    write_string += f"{ind['operator']}"
                                    assert ind["left"][0]["tag"] == "literal"
                                    write_string += (
                                        f"{ind['left'][0]['value']}"
                                    )
                                elif ind["tag"] == "literal":
                                    write_string += f"{ind['value']}"
                            if i < len(item["subscripts"]) - 1:
                                write_string += ", "
                                i = i + 1
                        write_string += "))"
                if printState.indexRef and "subscripts" not in item:
                    if "isDevType" not in item or (
                        "isDevType" in item and not item["isDevType"]
                    ):
                        write_string += "[0]"
                    elif "isDevType" in item and item["isDevType"]:
                        write_string += f".{item['field-name']}"
                write_string += ", "
        self.pyStrings.append(f"{write_string[:-2]}]")
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

    def nameMapping(self, ast):
        for item in ast:
            if item.get("name"):
                self.nameMapper[item["name"]] = item["name"]
            for inner in item:
                if isinstance(item[inner], list):
                    self.nameMapping(item[inner])

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

    def printArray(self, node, printState: PrintState):
        """ Prints out the array declaration in a format of Array class
            object declaration. 'arrayName = Array(Type, [bounds])'
        """
        if (
            self.nameMapper[node["name"]] not in printState.definedVars
            and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            printState.definedVars += [self.nameMapper[node["name"]]]
            assert int(node["count"]) > 0
            printState.definedVars += [node["name"]]

            varType = ""
            if node["type"].upper() == "INTEGER":
                varType = "int"
            elif node["type"].upper() in ("DOUBLE", "REAL"):
                varType = "float"
            elif node["type"].upper() == "CHARACTER":
                varType = "str"
            elif node["isDevTypeVar"]:
                varType = node["type"].lower() + "()"

            assert varType != ""

            self.pyStrings.append(f"{node['name']} = Array({varType}, [")
            for i in range(0, int(node["count"])):
                loBound = node["low" + str(i + 1)]
                upBound = node["up" + str(i + 1)]
                dimensions = f"({loBound}, {upBound})"
                if i < int(node["count"]) - 1:
                    self.pyStrings.append(f"{dimensions}, ")
                else:
                    self.pyStrings.append(f"{dimensions}")
            self.pyStrings.append("])")

            if node["isDevTypeVar"]:
                self.pyStrings.append(printState.sep)
                # This may require updating later when we have to deal with the
                # multi-dimensional derived type arrays
                upBound = node["up1"]
                self.pyStrings.append(
                    f"for z in range(1, {upBound}+1):" + printState.sep
                )
                self.pyStrings.append(
                    f"    obj = {node['type']}()" + printState.sep
                )
                self.pyStrings.append(
                    f"    {node['name']}.set_(z, obj)" + printState.sep
                )

    def printDerivedType(self, node, printState: PrintState):
        self.pyStrings.append("@dataclass\n")
        self.pyStrings.append(f"class {node['name']}:")
        self.pyStrings.append(printState.sep)

        curFieldType = ""
        fieldNum = 0
        for item in node:
            if f"field{fieldNum}" == item:
                if node[item][0]["type"].lower() == "integer":
                    curFieldType = "int"
                elif node[item][0]["type"].lower() in ("double", "real"):
                    curFieldType = "float"
                elif node[item][0]["type"].lower() == "character":
                    curFieldType = "str"

                fieldname = node[item][0]["field-id"]
                if "array-size" in node[item][0]:
                    self.pyStrings.append(f"    {fieldname} =")
                    self.pyStrings.append(f" Array({curFieldType}, [")
                    self.pyStrings.append(
                        f"(1, {node[item][0]['array-size']})])"
                    )
                else:
                    self.pyStrings.append(f"    {fieldname}:")
                    self.pyStrings.append(f" {curFieldType}")
                    if "value" in node[item][0]:
                        self.pyStrings.append(f" = {node[item][0]['value']}")
                    else:
                        self.pyStrings.append(" = None")
                self.pyStrings.append(printState.sep)
                fieldNum += 1

    def get_python_source(self):
        imports = "".join(self.imports)
        if len(imports) != 0:
            self.pyStrings.insert(1, imports)
        if self.programName != "":
            self.pyStrings.append(f"\n\n{self.programName}()\n")

        return "".join(self.pyStrings)


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
    derived_type_ast = []
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
    has_derived_type = False
    for index in list(main_ast[0]["body"]):
        if index["tag"] == "derived-type":
            has_derived_type = True
            derived_type_ast.append(index)
            main_ast[0]["body"].remove(index)

    # Print derived type declaration(s)
    if has_derived_type:
        code_generator.nameMapping(derived_type_ast)
        code_generator.printAst(derived_type_ast, PrintState())

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
