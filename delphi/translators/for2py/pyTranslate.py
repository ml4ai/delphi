""" This module contains code to convert a Fortran AST representation into a
Python script having the same functionalities and performing the same
operations as the original Fortran file.  """

import sys
import pickle
import argparse
import re
import json
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
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
    ".ne.": "!=",
    ".not.": "not",
    ".gt.": ">",
    ".eq.": "==",
    ".lt.": "<",
    ".le.": "<=",
    ".ge.": ">=",
    ".and.": "and",
    ".or.": "or",
    ".eqv.": "==",
    "//": "+",
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
#            -- py_mod is the module the Python function should be imported
#            from, None if no explicit import is necessary.

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
    "amax1": ("max", "FUNC", None),
    "min": ("min", "FUNC", None),
    "amin1": ("min", "FUNC", None),
    "mod": ("%", "INFIXOP", None),
    "modulo": ("%", "INFIXOP", None),
    "sin": ("sin", "FUNC", "math"),
    "sinh": ("sinh", "FUNC", "math"),
    "sqrt": ("sqrt", "FUNC", "math"),
    "tan": ("tan", "FUNC", "math"),
    "tanh": ("tanh", "FUNC", "math"),
    "xor": ("^", "INFIXOP", None),
    "rand": ("random", "FUNC", None),
    "len": ("len", "FUNC", None),
    "adjustl": ("adjustl", "FUNC", None),
    "adjustr": ("adjustr", "FUNC", None),
    "trim": ("strip", "FUNC", None),
    "real": ("float", "FUNC", None),    # This maay require change to Float32?
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
    ):
        self.sep = sep
        self.add = add
        self.printFirst = printFirst
        self.callSource = callSource
        self.definedVars = definedVars
        self.globalVars = globalVars
        self.functionScope = functionScope
        self.indexRef = indexRef

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
        )


class PythonCodeGenerator(object):
    def __init__(self):
        # Variable to hold the program name
        self.programName = ""
        # Dictionary to hold the tag mapping to the
        # print function that needs to be invoked
        self.printFn = {}
        # Dictionary to hold the declared variable
        # and its declared type
        self.variableMap = {}
        # List to hold the imports in the program
        self.imports = []
        # List to hold the private functions
        self.privFunctions = []
        # Dictionary to hold declared symbols (variable,
        # function, and format, etc) as key and map to
        # the symbol name that will be used in the python IR output
        self.nameMapper = {}
        # List to hold the declared function names
        self.functions = []
        # Dictionary to hold functions and its arguments
        self.funcArgs = {}
        # Pre-defined string format of system getframe
        self.getframe_expr = "sys._getframe({}).f_code.co_name"
        # List to hold the translated python string that
        # will be printed to the python IR output
        self.pyStrings = []
        # Dictionary holding mapping of read/write based on the file open state
        self.stateMap = {
            "unknown": "r",
            "replace": "w",
            "new": "w",
            "old": "r",
        }
        # Dictionary to hold the mapping of {label:format-code}
        self.format_dict = {}
        # Lists to hold derived type class
        self.declaredDerivedTypes = []
        # Lists to hold derived type variables
        self.declaredDerivedTVars = []
        # Dictionary to hold save variables local to a subroutine or function
        self.saved_variables = {}
        # String to hold the current name of the subroutine or function under
        # analysis
        self.current_module = ""
        # Dictionary to map all the variables (under each function) to its
        # corresponding types
        self.var_type = {}
        # String to hold the current call being processed
        self.current_call = None
        # This flag is True when the SAVE statement is in context
        self.is_save = False
        # This variable holds the current variable being inspected in the
        # select-case statement
        self.current_select = None
        # This flag remains False until the first case statement is started
        # in every select block
        self.case_started = False
        # This dictionary holds the defined arrays along with their data types
        self.array_map = {}
        # This keeps track of all declared String type variables with given
        # length.
        self.stringVars = {}
        self.module_log_map = None
        self.temp_dir = ""

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
            "cycle": self.printContinue,
            "select": self.printSelect,
            "case": self.printCase,
            "interface": self.printInterface,
        }
        self.readFormat = []

    ###########################################################################
    #                                                                         #
    #                      TOP-LEVEL PROGRAM COMPONENTS                       #
    #                                                                         #
    ###########################################################################

    def printSubroutine(self, node: Dict[str, str], printState: PrintState):
        """prints Fortran subroutine in python function syntax"""

        # Handle the save statement first since the decorator needs to be
        # just above the function definition
        args = []
        self.current_module = node["name"]
        self.printSave(node,
                       printState.copy(
                           sep=", ",
                           add="",
                           printFirst=False,
                           definedVars=args,
                           indexRef=False,
                       ),
                       )
        self.pyStrings.append(f"\ndef {self.nameMapper[node['name']]}(")

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
            ),
        )

    def printFunction(self, node, printState: PrintState):
        """prints Fortran function in python function
           syntax"""

        # Handle the save statement first since the decorator needs to be
        # just above the function definition
        args = []
        self.current_module = node["name"]
        self.printSave(node,
                       printState.copy(
                           sep=", ",
                           add="",
                           printFirst=False,
                           definedVars=args,
                           indexRef=False,
                       ),
                       )
        # self.functions.append(self.nameMapper[node['name']])
        self.pyStrings.append(f"\ndef {self.nameMapper[node['name']]}(")
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
        """
            prints the module syntax
        """
        self.pyStrings.append("\n")
        self.current_module = node["name"]
        self.saved_variables[self.current_module] = []
        args = []

        # Will need to remove any "save" tag since there will be no save
        # decorator for a module
        for item in node["body"]:
            if item.get("tag") == "save":
                node["body"].remove(item)

        self.printAst(
            node["body"],
            printState.copy(
                sep="", printFirst=True, definedVars=args, indexRef=True
            ),
        )

    def printProgram(self, node, printState: PrintState):
        """prints Fortran program in Python function syntax
           by calling printSubroutine function"""
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
        if py_mod is not None:
            handler = f"{py_mod}.{py_fn}"
        else:
            handler = py_fn

        if py_fn_type == "FUNC":
            arguments = ", ".join(arg_strs)
            if py_fn in ["adjustl", "adjustr", "strip"]:
                return f"{arguments}.{handler}()"
            else:
                return f"{handler}({arguments})"
        elif py_fn_type == "INFIXOP":
            assert len(arg_list) == 2, f"INFIXOP with {len(arg_list)} arguments"
            return f"({arg_strs[0]} {py_fn} {arg_strs[1]})"
        else:
            assert False, f"Unknown py_fn_type: {py_fn_type}"

    @staticmethod
    def get_arg_list(node):
        """Get_arg_list() returns the list of arguments or subscripts at a node.
           If there are no arguments or subscripts, it returns the empty list.
        """

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
            if self.variableMap[var]['type'].lower() == "character":
                to_find = node["args"][1]["value"]
                if len(node["args"]) == 3:
                    opt_arg = node["args"][2]["name"]
                    return f'{var}.f_index("{to_find}", ["{opt_arg}"])'
                else:
                    return f'{var}.f_index("{to_find}")'
            else:
                to_find = node["args"][1]["value"]
                return f"{var}[0].find({to_find})"

        if node["name"].lower() in syntax.F_INTRINSICS:
            return self.proc_intrinsic(node)

        self.current_call = self.nameMapper[f"{node['name']}"]
        args = self.get_arg_list(node)

        arg_strs = [self.proc_expr(args[i], True) for i in range(len(
            args))]

        # Case where a call is a print method
        if self.current_call == "print":
            arguments = self.proc_print(arg_strs)
        else:
            arguments = ", ".join(arg_strs)
        exp_str = f"{self.current_call}({arguments})"

        return exp_str

    @staticmethod
    def proc_print(arg_strs):
        arguments = ""
        for idx in range(0, len(arg_strs)):
            arguments += f"{arg_strs[idx]}"
            if idx < len(arg_strs) - 1:
                arguments += ", "
        return arguments

    @staticmethod
    def proc_literal(node):
        """Processes a literal value and returns a string that is the
           corresponding Python code."""

        if node["type"] == "bool":
            return node["value"].title()
        elif node["type"] == "char":
            if node['value'][0] in ["'", '"'] \
              and node["value"][-1] in ["'", '"']:
                return_val = node["value"][1:-1]
            else:
                return_val = node["value"]
            return f'"{return_val}"'
        else:
            return node["value"]

    def proc_ref(self, node, wrapper):
        """Processes a reference node and returns a string that is the
           corresponding Python code.  The argument "wrapper" indicates whether
           or not the Python expression should refer to the list wrapper for
           (scalar) variables."""
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
        # If the variable is a saved variable, prefix it by the function name
        # to access the saved value of the variable
        if self.current_module in self.saved_variables:
            if is_derived_type_ref:
                if ref_str.split('.')[0] in \
                            self.saved_variables[self.current_module]:
                    ref_str = f"{self.current_module}.{ref_str}"
            else:
                if ref_str in self.saved_variables[self.current_module]:
                    ref_str = f"{self.current_module}.{ref_str}"

        if "subscripts" in node:
            # array reference or function call or string indexing
            if "is_array" in node and node["is_array"] == "true":
                subs = node["subscripts"]
                subs_strs = [
                    self.proc_expr(subs[i], False) for i in range(len(subs))
                ]
                subscripts = ", ".join(subs_strs)
                expr_str = f"{ref_str}.get_(({subscripts}))"
            elif self.variableMap.get(node['name']) and \
                    self.variableMap[node['name']]['type'].lower() == \
                    "character":
                subs = node["subscripts"][0]
                subs_strs = [
                    self.proc_expr(subs[i][0], False) for i in subs
                ]
                subscripts = ", ".join(subs_strs)
                expr_str = f"{ref_str}.get_substr({subscripts})"
            else:
                expr_str = self.proc_call(node)
        else:
            # scalar variable
            if wrapper:
                expr_str = ref_str
            elif (
                    (
                        "is_arg" in node
                        and node["is_arg"] == "true"
                    )
                    or is_derived_type_ref
            ):
                expr_str = ref_str
                is_derived_type_ref = False
            elif ref_str in self.declaredDerivedTVars:
                expr_str = ref_str
            elif self.variableMap.get(ref_str) and \
                    self.variableMap[ref_str]["type"].lower() == 'character':
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
            raise For2PyError(f"unhandled operator {node['operator']}")

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
            literal = self.proc_literal(node)
            # If the literal is an argument to a function (when proc_expr has
            # been called from proc_call), the literal needs to be sent using
            # a list wrapper (e.g. f([1]) instead of f(1)
            if wrapper and self.current_call != "print":
                if node["type"] == "char":
                    return f"String({len(literal[1:-1])}, {literal})"
                return f"[{literal}]"
            else:
                return literal

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

        assert expr_str is not None, f">>> [proc_expr] NULL value: {node}"
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
            if node["type"].lower() in TYPE_MAP:
                var_type = TYPE_MAP[node["type"].lower()]
            elif node["type"].lower() == "character":
                var_type = "String"
            elif node["is_derived_type"] == "true":
                var_type = node["type"].lower()
        except KeyError:
            raise For2PyError(f"unrecognized type {node['type']}")

        arg_name = self.nameMapper[node["name"]]
        self.variableMap[arg_name] = {
            "type": node["type"],
            "parameter": False,
        }
        self.var_type.setdefault(self.current_module, []).append({
            "name": arg_name,
            "type": var_type
        })
        if "is_array" in node and node["is_array"] == "true":
            self.array_map[self.nameMapper[arg_name]] = "FromArg"
            self.pyStrings.append(f"{arg_name}: Array")
        elif var_type == "String":
            self.pyStrings.append(f"{arg_name}: String")
            printState.definedVars += [arg_name]
        else:
            if node["type"].lower() == "real":
                var_type = "Real"
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
        # If the target variable is a saved variable add the
        # subroutine/function name to it's prefix.
        name = self.nameMapper[node['name']]
        if name in self.saved_variables[self.current_module]:
            name = f"{self.current_module}.{name}"
        self.pyStrings.append(f"{name}[0] in range(")

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
        elif lhs["hasSubscripts"] == "true":
            assert (
                    "subscripts" in lhs
            ), "lhs 'hasSubscripts' and actual 'subscripts' existence does " \
               "not match.\
                                Fix 'hasSubscripts' in rectify.py."
            # target is an array element
            if "is_array" in lhs and lhs["is_array"] == "true":
                subs = lhs["subscripts"]
                subs_strs = [
                    self.proc_expr(subs[i], False)
                    for i in range(len(subs))
                ]
                subscripts = ", ".join(subs_strs)
                assg_str = f"{lhs['name']}.set_(({subscripts}), "
            elif self.variableMap[lhs['name']]['type'].lower() == "character":
                subs = lhs["subscripts"][0]
                subs_strs = [
                    self.proc_expr(subs[i][0], False) for i in subs
                ]
                subscripts = ", ".join(subs_strs)
                assg_str = f"{lhs['name']}.set_substr({subscripts}, "
            else:
                assert False
        elif self.variableMap[lhs['name']]['type'].lower() == "character":
            assg_str = f'{lhs["name"]}.set_('
        elif lhs["name"] in self.declaredDerivedTVars:
            assg_str = lhs["name"]
        else:
            # target is a scalar variable
            assg_str = f"{lhs['name']}[0]"

        # Check if this is a parameter assignment
        if lhs["is_parameter"] == "true":
            parameter_comment = f"  # PARAMETER: {assg_str[:-3].upper()}"
            self.variableMap[assg_str[:-3]]["parameter"] = True
            is_parameter = True
        else:
            is_parameter = False

        # Check if the rhs string contains a multiplication or division
        # operation and if so check if the target variable is an integer. In
        # this case, cast the rhs string with int()
        if '/' in rhs_str or '*' in rhs_str:
            for item in self.var_type[self.current_module]:
                if (lhs["is_derived_type_ref"] and item["name"] == assg_str) \
                        or (item["name"] == self.nameMapper[lhs["name"]]) and \
                        item["type"] == "int":
                    rhs_str = f"int({rhs_str})"

        # If the target variable is a saved variable add the
        # subroutine/function name to it's prefix.
        if lhs['name'] in self.saved_variables[self.current_module]:
            assg_str = f"{self.current_module}.{assg_str}"

        # Check if the lhs is a real and convert the variable to a numpy float
        # object if it is
        if (self.variableMap.get(lhs["name"]) == "REAL"
                and rhs["tag"] == "literal"):
            rhs_str = f"Float32({rhs_str})"

        if "set_" in assg_str:
            # If the assignment is to an array and the value is a string,
            # a String object has to be created
            if "is_array" in lhs and lhs["is_array"] == "true":
                check_further = False
                if self.array_map and self.array_map.get(lhs['name']):
                    check_further = True
                if check_further and \
                        "String" in self.array_map.get(lhs['name']):
                    length = self.array_map[lhs['name']][7:-1]
                    assg_str += f"String({length}, {rhs_str}))"
                else:
                    if self.array_map[lhs["name"]] == "float":
                        rhs_str = f"Float32({rhs_str})"
                    elif self.array_map[lhs["name"]] == "int":
                        rhs_str = f"int({rhs_str})"
                    else:
                        assert lhs["tag"] == "ref" , f"Unknown array type: {lhs}"

                    assg_str += f"{rhs_str})"
            else:
                # When the target is a string, the value can be a string
                # object as well. Check that and get the ._val to extract
                # actual string expression
                if node["value"][0]["tag"] == "ref":
                    if node["value"][0]["is_string"] == "true":
                        assg_str += f"{rhs_str}._val)"
                    else:
                        assg_str += f"str({rhs_str}))"
                else:
                    assg_str += f"{rhs_str})"
        else:
            assg_str += f" = {rhs_str}"

        if is_parameter:
            assg_str += parameter_comment

        self.pyStrings.append(assg_str)
        return

    def printUse(self, node, _):
        if node.get("include"):
            self.imports.append(
                f"from {self.temp_dir}.m_{node['arg'].lower()} "
                f"import {', '.join(node['include'])}\n"
            )
        else:
            self.imports.append(
                f"from {self.temp_dir}.m_"
                f"{node['arg'].lower()} import *\n"
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
                # TODO: Need to handle not just "op" but call to expressions
                #  as well. Also, current implementation does not go deep
                #  enough. This can be arbitrarily deep.
                if "value" in node:
                    val = node["value"]
                else:
                    assert (
                            "left" in node and
                            "operator" in node and
                            "right" in node
                    ), f"Something is missing. Detail of node content: {node}"

                    if node["left"][0]["tag"] == "ref":
                        left = self.proc_ref(node["left"][0], False)
                    elif node["left"][0]["tag"] == "call":
                        left = self.proc_call(node["left"][0])
                    elif node["left"][0]["tag"] == "op":
                        left = self.proc_op(node["left"][0])
                    else:
                        left = node["left"][0]["value"]
                    operator = node["operator"]
                    if node["right"][0]["tag"] == "ref":
                        right = self.proc_ref(node["right"][0], False)
                    elif node["right"][0]["tag"] == "call":
                        right = self.proc_call(node["right"][0])
                    elif node["right"][0]["tag"] == "op":
                        right = self.proc_op(node["right"][0])
                    else:
                        right = node["right"][0]["value"]

                    val = f"{left} {operator} {right}"
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
        if node['tag'] == "exit":
            self.pyStrings.append("break")
        elif node['tag'] == "stop":
            self.pyStrings.append("return")
        else:
            assert False, f"tag: {node['tag']} not being handled yet."

    def printReturn(self, node, printState: PrintState):
        self.pyStrings.append("")

    def printOpen(self, node, printState: PrintState):
        file_handle = self.get_file_handle(node)
        # We are making all file handles static i.e. SAVEing them which means
        # the file handles have to be prefixed with their subroutine/function
        # names
        if file_handle in self.saved_variables[self.current_module]:
            file_handle = f"{self.current_module}.{file_handle}"

        # At the very beginning, check if the status = OLD i.e. a condition
        # where the file has to previously exist.
        check_exist = False
        check_error = False
        for index, item in enumerate(node["args"]):
            if item.get("arg_name"):
                if item["arg_name"].lower() == "file":
                    if node["args"][index + 1]["tag"] == "literal":
                        file_name = node["args"][index + 1]["value"]
                    elif node["args"][index + 1]["tag"] == "ref":
                        file_name = node["args"][index + 1]["name"]
                    else:
                        assert False, "Can't get filename"
                    open_state = "r"
                elif item["arg_name"].lower() == "status":
                    open_state = node["args"][index + 1]["value"].replace(
                        "'", "").lower()
                    if open_state == "old":
                        check_exist = True
                    open_state = self.stateMap[open_state]
                elif item["arg_name"].lower() == "iostat":
                    check_error = True
                    error_variable = node["args"][index + 1]
                elif item["arg_name"].lower() == "position":
                    position_state = node["args"][index + 1]["value"].replace(
                        "'", "").lower()
                    if position_state == "append":
                        open_state = "a"

        if file_name in self.variableMap:
            file_is_var = True
            if self.variableMap[file_name]["type"].lower() == "character":
                file_name = f"{file_name}._val"
        else:
            file_is_var = False

        # If the file_name has quotes at the beginning and end, remove them
        if (file_name[0] == "'" and file_name[-1] == "'") or \
                (file_name[0] == '"' and file_name[-1] == '"'):
            file_name = file_name[1:-1]

        if check_error:
            assignment_tag = {
                "tag": "assignment",
                "target": [error_variable],
                "value": [{
                    "tag": "literal",
                    "type": "int",
                    "value": "0"
                }],
            }
            self.printAssignment(assignment_tag, printState)
            self.pyStrings.append(printState.sep)
        if check_exist:
            if file_is_var:
                self.pyStrings.append(f"if not os.path.exists({file_name}):")
            else:
                self.pyStrings.append(f"if not os.path.exists('{file_name}'):")
            self.pyStrings.append(printState.sep+printState.add)
            if check_error:
                assignment_tag = {
                    "tag": "assignment",
                    "target": [error_variable],
                    "value": [{
                        "tag": "literal",
                        "type": "int",
                        "value": "1"
                    }],
                }
                self.printAssignment(assignment_tag, printState)
            else:
                self.pyStrings.append(
                    f'assert False, "File: {file_name} does not '
                    f'exist."')
            self.pyStrings.append(printState.sep)

        if check_error:
            self.pyStrings.append("try:")
            self.pyStrings.append(printState.sep + printState.add)

        self.pyStrings.append(f"{file_handle} = ")
        if file_is_var:
            self.pyStrings.append(f'open({file_name}, "{open_state}")')
        else:
            self.pyStrings.append(f'open("{file_name}", "{open_state}")')

        if check_error:
            self.pyStrings.append(printState.sep + "except:")
            self.pyStrings.append(printState.sep + printState.add)
            assignment_tag = {
                "tag": "assignment",
                "target": [error_variable],
                "value": [{
                    "tag": "literal",
                    "type": "int",
                    "value": "1"
                }],
            }
            self.printAssignment(assignment_tag, printState)

    def printRead(self, node, printState: PrintState):
        if node["args"][0].get("tag") == "ref":
            file_handle = f'file_{self.nameMapper[node["args"][0]["name"]]}'
        else:
            file_handle = f'file_{node["args"][0]["value"]}'

        if node["args"][1]["type"] == "int":
            format_label = node["args"][1]["value"]

        # We are making all file handles static i.e. SAVEing them which means
        # the file handles have to be prefixed with their subroutine/function
        # names
        if file_handle in self.saved_variables[self.current_module]:
            file_handle = f"{self.current_module}.{file_handle}"

        isArray = False
        tempInd = 0
        if "subscripts" in node["args"][2]:
            array_len = len(node["args"]) - 2
            self.pyStrings.append(f"tempVar = [0] * {array_len}")
            self.pyStrings.append(printState.sep)

        ind = 0
        self.pyStrings.append("(")
        for item in node["args"][1:]:
            if item["tag"] == "ref":
                var = self.nameMapper[item["name"]]
                if "subscripts" in item:
                    isArray = True
                    self.pyStrings.append(f"tempVar[{tempInd}]")
                    tempInd = tempInd + 1
                else:
                    self.pyStrings.append(f"{var}[0]")
                if ind < len(node["args"][1:]) - 1:
                    self.pyStrings.append(", ")
            ind = ind + 1
        self.pyStrings.append(
            f",) = format_{format_label}_obj."
            f"read_line({file_handle}.readline())"
        )
        self.pyStrings.append(printState.sep)

        if isArray:
            tempInd = 0  # Re-initialize to zero for array index
            for item in node["args"][1:]:
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
        if node["args"][0].get("value") and node["args"][0]["value"] == "*":
            write_target = "outStream"
        elif node["args"][0].get("tag") and node["args"][0].get("tag") == "ref":
            write_target = "variable"
        else:
            write_target = "file"
            if node["args"][0].get("value"):
                file_id = str(node["args"][0]["value"])
            elif str(node["args"][0].get("tag")) == "ref":
                file_id = str(self.nameMapper[node["args"][0].get("name")])
            file_handle = "file_" + file_id

            # We are making all file handles static i.e. SAVEing them which
            # means the file handles have to be prefixed with their
            # subroutine/function names
            if file_handle in self.saved_variables[self.current_module]:
                file_handle = f"{self.current_module}.{file_handle}"

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
        elif write_target == "variable":
            varTarget = node["args"][0]["name"]
            self.pyStrings.append(f"{varTarget}")

        # Collect the expressions to be written out. The first two arguments to
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
                    # self.pyStrings.append(f"{var},")
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
        elif write_target == "variable":
            # TODO: This has a lot of assumptions that may require generalising.
            # Currently, this  code is written specifically targeting
            # 1020         WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)')
            # 1021      &      Trim(VarName), Trim(ModuleName)
            # like syntax, which can be found in ModuleDefs.for
            if (
                    node["args"][0].get("subscripts")
                    and len(node["args"][0]["subscripts"]) > 0
            ):
                outputString = varTarget + ".set_("
                index = "("
                for idx in node["args"][0]["subscripts"]:
                    if idx["tag"]  == "literal":
                        index += f"{idx['value']},"
                index = index.strip(',') + ")"
                outputString += f"{index}, String("
                if (
                        self.current_module in self.stringVars
                        and varTarget in self.stringVars[self.current_module]
                ):
                    strLength = self.stringVars[self.current_module][varTarget]
                outputString += f"{strLength},"
                tmpStr = "\'"
                for i in range(1, len(node["args"])):
                    if node["args"][i]["tag"] == "literal":
                        tmpStr += f'{node["args"][i]["value"]}\''
                    elif node["args"][i]["tag"] == "ref":
                        if node["args"][i]["name"].lower() in syntax.F_INTRINSICS:
                            intrString = self.proc_intrinsic(node["args"][i])
                            tmpStr += f"+{intrString}"
                outputString += f"{tmpStr}))"
                self.pyStrings.append(outputString)

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
        if (
                var_name not in printState.definedVars + printState.globalVars
                and var_name not in self.functions and var_name not in
                self.saved_variables.get(self.current_module)
        ):
            printState.definedVars += [var_name]
            if node.get("value"):
                if node["value"][0]["tag"] == "literal":
                    # initVal = node["value"][0]["value"]
                    initVal = self.proc_literal(node["value"][0])
                elif node["value"][0]["tag"] == "op":
                    initVal = self.proc_op(node["value"][0])
                else:
                    assert (
                        False
                    ), f"Tag {node['value'][0]['tag']} " \
                        f"currently not handled yet."
            else:
                initVal = None

            varType = self.get_type(node)

            if node.get("value") and node["value"][0]["tag"] == "op" and \
                    ("/" in initVal or "*" in initVal) and varType == "int":
                initVal = f"int({initVal})"

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
                elif node.get('is_string') and node["is_string"] == "true":
                    if not initVal:
                        self.pyStrings.append(f"{var_name} = "
                                              f"String({node['length']})")
                    else:
                        self.pyStrings.append(f'{var_name} = '
                                              f'String({node["length"]}, '
                                              f'{initVal})')
                else:
                    self.pyStrings.append(
                        f"{var_name}: List[{varType}] = " f"[{initVal}]"
                    )

            # The code below might cause issues on unexpected places.
            # If weird variable declarations appear, check code below

            if not printState.sep:
                printState.sep = "\n"
            self.variableMap[self.nameMapper[node["name"]]] = {
                "type": node["type"].upper(),
                "parameter": False,
            }
        else:
            # If the variable is a saved variable, add its type mapping to
            # self.var_type
            if var_name in self.saved_variables[self.current_module]:
                varType = self.get_type(node)
            self.variableMap[self.nameMapper[node["name"]]] = {
                "type": node["type"],
                "parameter": False,
            }
            printState.printFirst = False

    def printArray(self, node, printState: PrintState):
        """ Prints out the array declaration in a format of Array class
            object declaration. 'arrayName = Array(Type, [bounds])'
        """
        if (
                self.nameMapper[node["name"]] not in printState.definedVars
                and self.nameMapper[node["name"]] not in printState.globalVars
        ):
            array_arg = False
            if node["name"] in self.array_map and self.array_map[node[
             "name"]] == "FromArg":
                array_arg = True
            printState.definedVars += [self.nameMapper[node["name"]]]
            var_type = self.get_type(node)
            # If the array has string elements, then we need the length
            # information as well
            if var_type == "character":
                var_type = f"String({node['length']})"
                self.stringVars[self.current_module] = {node["name"]:node["length"]}

            self.array_map[self.nameMapper[node["name"]]] = var_type
            array_range = self.get_array_dimension(node)

            # If the printArray function is being called from printSave,
            # is_save will be True and the argument to the decorator is
            # returned.
            if self.is_save:
                save_argument = f'{{"name": "{node["name"]}", "call": Array' \
                    f'({var_type}, [{array_range}]), "type": ' \
                    f'"{TYPE_MAP[node["type"].lower()]}"}}'
                return save_argument
            else:
                # If the array variable is not SAVEd, print the
                # initialization of the array
                if node["name"] not in self.saved_variables[
                            self.current_module] and not array_arg:
                    self.pyStrings.append(
                        f"{node['name']} = Array({var_type}, [{array_range}])"
                    )
                else:
                    # If it is SAVEd, don't print the initialization code and
                    # instead delete the remnant empty line.
                    del self.pyStrings[-1]

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
            self.var_type.setdefault(derived_type_class_info["type"],
                                     []).append(
                {
                    "name": derived_type_variables[var]["name"],
                    "type": var_type
                })
            is_derived_type_declaration = False
            # If the type is not one of the default types, but it's
            # a declared derived type, set the is_derived_type_declaration
            # to True as the declaration of variable with such type has
            # different declaration syntax from the default type variables.
            if (
                    var_type not in TYPE_MAP
                    and var_type in self.declaredDerivedTypes
            ):
                is_derived_type_declaration = True

            if derived_type_variables[var]["is_array"] == "false":
                if not is_derived_type_declaration:
                    if var_type == "String":
                        str_length = derived_type_variables[var]["length"]
                        self.pyStrings.append(
                            f"        self.{name} = {var_type}({str_length}"
                        )

                        if "value" in derived_type_variables[var]:
                            value = self.proc_literal(
                                derived_type_variables[var]["value"][0]
                            )
                            self.pyStrings.append(f", \"{value}\"")
                        self.pyStrings.append(")")
                    else:
                        self.pyStrings.append(f"        self.{name}: {var_type}")
                        if "value" in derived_type_variables[var]:
                            value = self.proc_literal(
                                derived_type_variables[var]["value"][0]
                            )
                            self.pyStrings.append(f" = {value}")
                else:
                    self.pyStrings.append(
                        f"        self.{name} = {var_type}()"
                    )

            else:
                array_range = self.get_array_dimension(
                    derived_type_variables[var]
                )
                if var_type == "character":
                    var_type = f"String(" \
                               f"{derived_type_variables[var]['length']})"
                self.pyStrings.append(
                    f"        self.{name} = Array({var_type}, [{array_range}])"
                )
            self.pyStrings.append(printState.sep)

    def printSave(self, node, printState: PrintState):
        """
        This function adds the Python string to handle Fortran SAVE
        statements. It adds a decorator above a function definition and makes
        a call to static_save function with the list of saved variables as
        its argument.
        """
        self.is_save = True
        parent = node["name"]
        self.saved_variables[parent] = []
        for item in node["body"]:
            if item.get("tag") == "save":
                to_delete = item
                self.pyStrings.append("\n@static_vars([")
                variables = ''
                for var in item["var_list"]:
                    if var["tag"] == "open":
                        variable_type = "file_handle"
                    elif var["type"] == "character":
                        variable_type = f"String({var['length']})"
                    else:
                        variable_type = TYPE_MAP[var["type"].lower()]
                    if var.get("name"):
                        self.saved_variables[parent].append(var["name"])
                    if var["tag"] == "array":
                        # Call printArray to get the initialization code for
                        # Array declaration. This will be on the argument to
                        # the decorator.
                        save_argument = self.printArray(var, printState)
                        variables += f"{save_argument}, "
                    elif var["tag"] == "variable":
                        if var["is_derived_type"] == "true":
                            save_argument = f"{{'name': '{var['name']}', " \
                                f"'call': {var['type']}(), 'type': " \
                                f"'{variable_type}'}}"
                        else:
                            save_argument = {"name": var["name"],
                                             "call": [None],
                                             "type": variable_type}
                        variables += f"{save_argument}, "
                    elif var["tag"] == "open":
                        name = self.get_file_handle(var)
                        self.saved_variables[parent].append(name)
                        save_argument = f"{{'name': '{name}', 'call': None, " \
                            f"'type': 'file_handle'}}"
                        variables += f"{save_argument}, "
                self.pyStrings.append(f"{variables[:-2]}])")
                node["body"].remove(to_delete)
        self.is_save = False

    def printContinue(self, node, printState: PrintState):
        """
        This function simply adds python statement "continue"
        that is equivalent to "cycle" in Fortran.
        """
        assert (
                node['tag'] == "cycle"
        ), f"Tag must be <cycle> for <continue> statement.\
            current: {node['tag']}"
        self.pyStrings.append("continue")

    def printSelect(self, node, printState: PrintState):
        """
        This function converts the select-case statement in Fortran into an
        if-else statement block in Python
        """
        for arg in node["args"]:
            if arg['tag'] == "ref":
                self.current_select = self.proc_ref(arg, False)
        self.case_started = False
        self.printAst(node["body"], printState.copy(
                sep=printState.sep,
                printFirst=True,
                indexRef=True,
            ),
                      )

    def printCase(self, node, printState: PrintState):
        """
        This function handles each CASE statement block. This relates to one
        if-block in Python
        """
        if node.get('args'):
            if not self.case_started:
                self.case_started = True
            else:
                self.pyStrings.append("else:"+printState.sep+printState.add)
                printState.sep += printState.add
            self.pyStrings.append("if ")

            for index, arg in enumerate(node["args"]):
                if index == 0:
                    self.pyStrings.append("(")
                else:
                    self.pyStrings.append(" or (")
                if arg.get("tag") == "case_range":
                    arguments = arg['args']
                    select_string = re.sub(r'\[.*\]', '', self.current_select)
                    if self.variableMap[select_string]['type'].lower() \
                            == "character":
                        check_variable = f"{self.current_select}.__str__()"
                    else:
                        check_variable = self.current_select

                    if len(arguments) == 1:
                        self.pyStrings.append(f"{check_variable} == ")
                        self.printAst(
                            arg['args'],
                            printState.copy(
                                sep="",
                            )
                        )
                    elif len(arguments) == 2:
                        left_arg = arguments[0]
                        right_arg = arguments[1]
                        if left_arg.get("tag") == "literal" and \
                           left_arg.get("value") == "'-Inf'":
                            self.pyStrings.append(f"{check_variable} <= ")
                            self.printAst(
                                [right_arg],
                                printState.copy(
                                    sep="",
                                )
                            )
                        elif right_arg.get("tag") == "literal" and  \
                                right_arg.get("value") == "'Inf'":
                            self.pyStrings.append(f"{check_variable} >= ")
                            self.printAst(
                                [left_arg],
                                printState.copy(
                                    sep="",
                                )
                            )
                        else:
                            self.pyStrings.append(f"{check_variable} >= ")
                            self.printAst(
                                [left_arg],
                                printState.copy(
                                    sep="",
                                )
                            )
                            self.pyStrings.append(f" and "
                                                  f"{check_variable} <= ")
                            self.printAst(
                                [right_arg],
                                printState.copy(
                                    sep="",
                                )
                            )
                    else:
                        assert False, f"Invalid length of case arguments " \
                                      f"{len(arguments)}"
                    self.pyStrings.append(")")
                else:
                    assert False, f"Unhandled case argument {arg.get('tag')}"
            self.pyStrings.append(":")
        else:
            self.pyStrings.append("else:")

        self.printAst(
            node["body"],
            printState.copy(
                sep=printState.sep + printState.add,
                printFirst=True,
                indexRef=True,
            ),
        )

    def printInterface(self, node, printState: PrintState):
        """This function definition is simply a place holder for INTERFACE
        just in case of any possible usage in the future. For now, it does
        nothing and pass. Since translate.py also passes interface, this
        should not be encountered in any case."""
        assert False, "In printInterface function, which should not be in " \
                      "any case."

    ###########################################################################
    #                                                                         #
    #                              MISCELLANEOUS                              #
    #                                                                         #
    ###########################################################################

    def initializeFileVars(self, node, printState: PrintState):
        label = node["args"][1]["value"]
        data_type = list_data_type(self.format_dict[label])
        index = 0
        for item in node["args"][1:]:
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
            if isinstance(item, list) or isinstance(item, dict):
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

    def get_range(self, node):
        """This function will construct the range string in 'loBound,
           Upbound' format and return to the called function"""
        # [0]: lower bound
        # [1]: upper bound
        bounds = [0, 0]
        retrieved_bounds = [node["low"], node["high"]]
        self.get_bound(bounds, retrieved_bounds)

        return f"{bounds[0]}, {bounds[1]}"

    def get_bound(self, bounds, retrieved_bounds):
        """This function will fill the bounds list with appropriately
        retrieved bounds from the node."""
        index = 0
        for bound in retrieved_bounds:
            if bound[0]["tag"] == "literal":
                bounds[index] = self.proc_literal(bound[0])
            elif bound[0]["tag"] == "op":
                bounds[index] = self.proc_op(bound[0])
            elif bound[0]["tag"] == "ref":
                bounds[index] = self.proc_ref(bound[0], False)
            else:
                assert False, f"Unrecognized tag in retrieved_bound: " \
                    f"{bound[0]['tag']}"
            index += 1

    def get_derived_type_ref(self, node, numPartRef, is_assignment):
        """This function forms a derived type reference
        and return to the caller"""
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
        elif self.current_module in self.funcArgs and \
                node["name"] in self.funcArgs[self.current_module]:
            ref += f'{node["name"]}[0]'
        else:
            ref += node["name"]
        numPartRef -= 1
        if "ref" in node:
            derived_type_ref = self.get_derived_type_ref(
                node['ref'][0], numPartRef, is_assignment)
            ref += f".{derived_type_ref}"
        return ref

    def get_type(self, node):
        """ This function checks the type of a variable and returns
            the appropriate Python syntax type name. """

        variable_type = node["type"].lower()
        var_name = self.nameMapper[node["name"]]

        if variable_type in TYPE_MAP:
            mapped_type = TYPE_MAP[variable_type]
            if node.get("is_derived_type") == "false":
                self.set_default_var_type(var_name, mapped_type)
            return mapped_type
        else:
            if node["is_derived_type"] == "true":
                if variable_type == "character":
                    variable_type = "String"
                self.set_default_var_type(var_name, variable_type)
                # Add each element of the derived type into self.var_type as
                # well
                if variable_type in self.var_type:
                    for var in self.var_type[variable_type]:
                        self.var_type[self.current_module].append({
                            "name": f"{var_name}.{var['name']}",
                            "type": var['type']
                        })
            elif (
                    node["is_derived_type"] == "false"
                    and variable_type == "character"
            ):
                variable_type == "str"
                self.set_default_var_type(var_name, variable_type)
            else:
                assert False, f"Unrecognized variable type: {variable_type}"

            return variable_type

    def set_default_var_type(self, var_name, var_type):
        """ This function sets the default variable type of the declared
        variable."""

        self.var_type.setdefault(self.current_module, []).append({
            "name": var_name,
            "type": var_type
        })

    def get_array_dimension(self, node):
        """ This function is for extracting the dimensions' range information
            from the AST. This function is needed for handling a multi-dimensional
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
            elif (
                    "name" in dimension
            ): # A case where variable name given as a size with no explicit lower bound.
                array_range += f"(0, {dimension['name']})"
            else:
                assert (
                    False
                ), f"Array range case not handled. Reference node content: {node}"

            if count < int(node["count"]):
                array_range += ", "
                count += 1

        return array_range

    def get_file_handle(self, node):
        if node["args"][0].get("arg_name") == "unit":
            file_handle = f'file_{node["args"][1]["value"]}'
        elif node["args"][0].get("tag") == "ref":
            file_handle = f'file_{self.nameMapper[node["args"][0]["name"]]}'
        else:
            file_handle = f'file_{node["args"][0]["value"]}'

        return file_handle


def index_modules(root) -> Dict:
    """
        Counts the number of modules in the Fortran file including the program
        file. Each module is written out into a separate Python file.
    """

    module_index_dict = {
        node["name"]: (node.get("tag"), index)
        for index, node in enumerate(root)
        if node.get("tag") in ("module", "program", "subroutine", "function")
    }

    return module_index_dict


def get_python_sources_and_variable_map(outputDict: Dict,
                                        module_log_file_path: str,
                                        temp_dir: str):
    module_index_dict = index_modules(outputDict["ast"])
    py_sourcelist = []
    main_ast = []
    import_lines = [
        "import sys",
        "import os",
        "from typing import List",
        "import math",
        "from delphi.translators.for2py.format import *",
        "from delphi.translators.for2py.arrays import *",
        "from delphi.translators.for2py.static_save import *",
        "from delphi.translators.for2py.strings import *",
        "from dataclasses import dataclass",
        "from delphi.translators.for2py.types_ext import Float32",
        "import delphi.translators.for2py.math_ext as math",
        "from numbers import Real",
        "from random import random\n",
    ]

    with open(module_log_file_path) as json_f:
        module_logs = json.load(json_f)

    for module in module_index_dict:
        code_generator = PythonCodeGenerator()
        code_generator.module_log_map = module_logs
        code_generator.temp_dir = temp_dir.replace("/", ".")
        code_generator.pyStrings.append("\n".join(import_lines))
        if "module" in module_index_dict[module]:
            ast = [outputDict["ast"][module_index_dict[module][1]]]
            # Copy the derived type ast from the main_ast into the separate
            # list, so it can be printed outside (above) the main method
            derived_type_ast = []
            for index in list(ast[0]["body"]):
                if "is_derived_type" in index and index["is_derived_type"] == \
                        "true":
                    if "tag" not in index:
                        derived_type_ast.append(index)
                        ast[0]["body"].remove(index)
                if "tag" in index and index["tag"] == "use":
                    # Look at the all the files/modules that are imported by
                    # the current file/module and populate the list of
                    # declared derived  types. This will have to be done
                    # eventually for other data types as well.
                    imported_system = index["arg"]
                    code_generator.declaredDerivedTypes.extend(
                        code_generator.module_log_map["mod_info"][
                            imported_system]["derived_type_list"])

            # Print derived type declaration(s)
            if derived_type_ast:
                for i in range(len(derived_type_ast)):
                    code_generator.pyStrings.append("\n@dataclass")
                    assert (
                            derived_type_ast[i]["is_derived_type"] == "true"
                    ), "[derived_type_ast] holds non-derived type ast"
                    code_generator.nameMapping([derived_type_ast[i]])
                    code_generator.printDerivedType(
                        [derived_type_ast[i]], PrintState()
                    )
        else:
            main_ast.append(outputDict["ast"][module_index_dict[module][1]])
            continue

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
    code_generator.module_log_map = module_logs
    code_generator.temp_dir = temp_dir.replace("/", ".")
    code_generator.functions = outputDict["functionList"]
    code_generator.pyStrings.append("\n".join(import_lines))

    # Some Fortran files do not hold any PROGRAM function, such as files created
    # for module holder purpose.
    if main_ast:
        # Copy the derived type ast from the main_ast into the separate list,
        # so it can be printed outside (above) the main method
        derived_type_ast = []
        for index in list(main_ast[0]["body"]):
            if "is_derived_type" in index and index["is_derived_type"] == "true":
                if "tag" not in index:
                    derived_type_ast.append(index)
                    main_ast[0]["body"].remove(index)
                if "tag" in index and index["tag"] == "use":
                    # Look at the all the files/modules that are imported by
                    # the current file/module and populate the list of
                    # declared derived  types. This will have to be done
                    # eventually for other data types as well.
                    imported_system = index["arg"]
                    code_generator.declaredDerivedTypes.extend(
                        code_generator.module_log_map["mod_info"][
                            imported_system]["derived_type_list"])

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

    return (py_sourcelist, code_generator.variableMap)


def parse_args():
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
        "-t",
        "--target",
        nargs="+",
        required=True,
        help=(
            "Target directory to store the output files in"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        nargs="+",
        help="Text file containing the list of output python files being "
             "generated",
    )
    args = parser.parse_args(sys.argv[1:])

    pickleFile = args.files[0]
    pyFile = args.gen[0]
    outFile = args.out[0]
    targetDir = args.target[0]

    return (pickleFile, pyFile, targetDir, outFile)


if __name__ == "__main__":
    (pickleFile, pyFile, targetDir, outFile) = parse_args()

    try:
        with open(pickleFile, "rb") as f:
            outputDict = pickle.load(f)
    except IOError:
        raise For2PyError(f"Unable to read file {pickleFile}.")

    (python_source_list, variable_map) = create_python_source_list(outputDict)
    outputList = []
    for item in python_source_list:
        if item[2] == "module":
            modFile = f"{targetDir}m_{item[1].lower()}.py"
            try:
                with open(modFile, "w") as f:
                    outputList.append(modFile)
                    f.write(item[0])
            except IOError:
                raise For2PyError(f"Unable to write to {modFile}")
        else:
            try:
                with open(pyFile, "w") as f:
                    outputList.append(pyFile)
                    f.write(item[0])
            except IOError:
                raise For2PyError(f"Unable to write to {pyFile}.")

    variable_map_file = f"{pyFile[:-3]}_variables_pickle"
    try:
        with open(variable_map_file, "wb") as f:
            pickle.dump(variable_map, f)
    except IOError:
        raise For2PyError(f"Unable to write to {variable_map_file}.")

    try:
        with open(outFile, "w") as f:
            for fileName in outputList:
                f.write(fileName + " ")
    except IOError:
        raise For2PyError(f"Unable to write to {outFile}.")

