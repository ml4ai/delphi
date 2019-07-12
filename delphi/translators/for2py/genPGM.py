#!/usr/bin/python3.6

import ast
import sys
import tokenize
from datetime import datetime
import re
import argparse
from functools import reduce
import json
from delphi.translators.for2py.genCode import genCode, PrintState
from delphi.translators.for2py.mod_index_generator import get_index
from delphi.translators.for2py import For2PyError
from typing import List, Dict, Iterable, Optional
from itertools import chain, product
import operator
import os
import uuid

###########################################################################
#                                                                         #
#                            GLOBAL VARIABLES                             #
#                                                                         #
###########################################################################

# The BINOPS dictionary holds operators for all the arithmetic and
# comparative functions
BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Eq: operator.eq,
    ast.LtE: operator.le,
}

# The ANNOTATE_MAP dictionary is used to map Python ast data types into data
# types for the lambdas
ANNOTATE_MAP = {
    "real": "Real",
    "integer": "int",
    "string": "str",
    "array": "[]",
    "bool": "bool",
}

# This dictionary helps in reverse mapping the data types for internal
# computations
REVERSE_ANNOTATE_MAP = {
    "float": "real",
    "Real": "real",
    "int": "integer",
    "list": "array",
    "str": "string",
    "bool": "bool",
}

# The UNNECESSARY_TYPES tuple holds the ast types to ignore
UNNECESSARY_TYPES = (
    ast.Mult,
    ast.Add,
    ast.Sub,
    ast.Pow,
    ast.Div,
    ast.USub,
    ast.Eq,
    ast.LtE,
)

# Regular expression to match python statements that need to be bypassed in
# the GrFN and lambda files. Currently contains I/O statements.
BYPASS_IO = r"^format_\d+$|^format_\d+_obj$|^file_\d+$|^write_list_\d+$|" \
            r"^write_line$|^format_\d+_obj" \
            r".*|^Format$|^list_output_formats$|^write_list_stream$"
RE_BYPASS_IO = re.compile(BYPASS_IO, re.I)


class GrFNState:
    def __init__(
        self,
        lambda_strings: Optional[List[str]],
        last_definitions: Optional[Dict] = {},
        next_definitions: Optional[Dict] = {},
        last_definition_default=0,
        function_name=None,
        variable_types: Optional[Dict] = {},
        start: Optional[Dict] = {},
        scope_path: Optional[List] = [],
    ):
        self.lambda_strings = lambda_strings
        self.last_definitions = last_definitions
        self.next_definitions = next_definitions
        self.last_definition_default = last_definition_default
        self.function_name = function_name
        self.variable_types = variable_types
        self.start = start
        self.scope_path = scope_path

    def copy(
        self,
        lambda_strings: Optional[List[str]] = None,
        last_definitions: Optional[Dict] = None,
        next_definitions: Optional[Dict] = None,
        last_definition_default=None,
        function_name=None,
        variable_types: Optional[Dict] = None,
        start: Optional[Dict] = None,
        scope_path: Optional[List] = None,
    ):
        return GrFNState(
            self.lambda_strings if lambda_strings is None else lambda_strings,
            self.last_definitions if last_definitions is None else
            last_definitions,
            self.next_definitions if next_definitions is None else
            next_definitions,
            self.last_definition_default if last_definition_default is None
            else last_definition_default,
            self.function_name if function_name is None else function_name,
            self.variable_types if variable_types is None else variable_types,
            self.start if start is None else start,
            self.scope_path if scope_path is None else scope_path,
        )


class GrFNGenerator(object):
    def __init__(self,
                 annotated_assigned=[],
                 elif_grfn=[],
                 function_definitions=[]
                 ):
        self.annotated_assigned = annotated_assigned
        self.elif_grfn = elif_grfn
        self.function_definitions = function_definitions
        self.exclude_list = []
        self.mode_mapper = {}
        self.alias_mapper = {}
        self.name_mapper = {}
        self.function_names = {}
        self.types = (list, ast.Module, ast.FunctionDef)

    def gen_grfn(self, node, state, call_source):
        """
            This function generates the GrFN structure by parsing through the
            python AST
        """

        # Look for code that is not inside any function. This will generally
        # involve
        if state.function_name is None and not any(
            isinstance(node, t) for t in self.types
        ):
            if isinstance(node, ast.Call):
                return [{"start": node.func.id}]
            elif isinstance(node, ast.Expr):
                return self.gen_grfn(node.value, state, "start")
            elif isinstance(node, ast.If):
                return self.gen_grfn(node.body, state, "start")
            else:
                return []

        if isinstance(node, list):
            return self.process_list(node, state, call_source)
        elif isinstance(node, ast.FunctionDef):
            # Function: name, args, body, decorator_list, returns
            return self.process_function_definition(node, state)
        elif isinstance(node, ast.arguments):
            # arguments: ('args', 'vararg', 'kwonlyargs', 'kw_defaults',
            # 'kwarg', 'defaults')
            return self.process_arguments(node, state, call_source)
        elif isinstance(node, ast.arg):
            # arg: ('arg', 'annotation')
            return self.process_arg(node, state, call_source)
        elif isinstance(node, ast.Load):
            # Load: ()
            return self.process_load(node, state, call_source)
        elif isinstance(node, ast.Store):
            # Store: ()
            return self.process_store(node, state, call_source)
        elif isinstance(node, ast.Index):
            # Index: ('value',)
            return self.process_index(node, state)
        elif isinstance(node, ast.Num):
            # Num: ('n',)
            return self.process_num(node)
        elif isinstance(node, ast.List):
            # List: ('elts', 'ctx')
            return self.process_list_ast(node, state)
        elif isinstance(node, ast.Str):
            # Str: ('s',)
            return self.process_str(node)
        elif isinstance(node, ast.For):
            # For: ('target', 'iter', 'body', 'orelse')
            return self.process_for(node, state, call_source)
        elif isinstance(node, ast.If):
            # If: ('test', 'body', 'orelse')
            return self.process_if(node, state, call_source)
        elif isinstance(node, ast.UnaryOp):
            # UnaryOp: ('op', 'operand')
            return self.process_unary_operation(node, state, call_source)
        elif isinstance(node, ast.BinOp):
            # BinOp: ('left', 'op', 'right')
            return self.process_binary_operation(node, state, call_source)
        elif any(isinstance(node, node_type) for node_type in
                 UNNECESSARY_TYPES):
            # Mult: ()
            return self.process_unnecessary_types(node, state, call_source)
        elif isinstance(node, ast.Expr):
            # Expr: ('value',)
            return self.process_expression(node, state, call_source)
        elif isinstance(node, ast.Compare):
            # Compare: ('left', 'ops', 'comparators')
            return self.process_compare(node, state, call_source)
        elif isinstance(node, ast.Subscript):
            # Subscript: ('value', 'slice', 'ctx')
            return self.process_subscript(node, state, call_source)
        elif isinstance(node, ast.Name):
            # Name: ('id', 'ctx')
            return self.process_name(node, state, call_source)
        elif isinstance(node, ast.AnnAssign):
            # AnnAssign: ('target', 'annotation', 'value', 'simple')
            return self.process_annotated_assign(node, state, call_source)
        elif isinstance(node, ast.Assign):
            # Assign: ('targets', 'value')
            return self.process_assign(node, state, call_source)
        elif isinstance(node, ast.Tuple):
            # Tuple: ('elts', 'ctx')
            return self.process_tuple(node, state, call_source)
        elif isinstance(node, ast.Call):
            # Call: ('func', 'args', 'keywords')
            return self.process_call(node, state, call_source)
        elif isinstance(node, ast.Module):
            # Module: body
            return self.process_module(node, state, call_source)
        elif isinstance(node, ast.BoolOp):
            # BoolOp: body
            return self.process_boolean_operation(node, state, call_source)
        elif isinstance(node, ast.Attribute):
            return self.process_attribute(node, state, call_source)
        elif isinstance(node, ast.AST):
            return self.process_ast(node, state, call_source)
        else:
            return self.process_nomatch(node, state, call_source)

    def process_list(self, node, state, call_source):
        """
         If there are one or more ast nodes inside the `body` of a node,
         there appear as a list. Process each node in the list and chain them
         together into a single list of GrFN dictionaries.
        """
        return list(
            chain.from_iterable(
                [
                    self.gen_grfn(cur, state, call_source)
                    for cur in node
                ]
            )
        )

    def process_function_definition(self, node, state):
        """
            This function processes the function definition i.e. functionDef
            instance. It appends GrFN dictionaries to the `functions` key in
            the main GrFN JSON. These dictionaries consist of the
            function_assign_grfn of the function body and the
            function_container_grfn of the function. Every call to this
            function adds these along with the identifier_spec_grfn to the
            main GrFN JSON.
        """

        # Add the function name to the list that stores all the functions
        # defined in the program
        self.function_definitions.append(node.name)

        local_last_definitions = state.last_definitions.copy()
        local_next_definitions = state.next_definitions.copy()
        local_variable_types = state.variable_types.copy()
        scope_path = state.scope_path.copy()

        # If the scope_path is empty, add _TOP to the list to denote that
        # this is the outermost scope
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        scope_path.append(node.name)

        if node.decorator_list:
            # This is still a work-in-progress function since a complete
            # representation of SAVEd variables has not been decided for GrFN.
            # Currently, if the decorator function is static_vars (for
            # SAVEd variables), their types are loaded in the variable_types
            # dictionary.
            function_state = state.copy(
                last_definitions=local_last_definitions,
                next_definitions=local_next_definitions,
                function_name=node.name,
                variable_types=local_variable_types,
            )
            process_decorators(node.decorator_list, function_state)

        # Check if the function contains arguments or not. This determines
        # whether the function is the outermost scope (does not contain
        # arguments) or it is not (contains arguments). For non-outermost
        # scopes, indexing starts from -1. For outermost scopes, indexing
        # starts from 0
        # TODO: What do you do when a non-outermost scope function does not
        #  have arguments. Current assumption is that the function without
        #  arguments is the outermost function i.e. call to the `start`
        #  function. But there can be functions without arguments which are not
        #  the `start` functions but instead some inner functions.
        if len(node.args.args) == 0:
            function_state = state.copy(
                last_definitions=local_last_definitions,
                next_definitions=local_next_definitions,
                function_name=node.name,
                variable_types=local_variable_types,
            )
        else:
            function_state = state.copy(
                last_definitions={},
                next_definitions={},
                function_name=node.name,
                variable_types=local_variable_types,
                last_definition_default=-1,
            )

        # Get the list of arguments from the function definition
        argument_list = self.gen_grfn(node.args, function_state, "functiondef")
        # Enter the body of the function and recursively generate the GrFN of
        # the function body
        body_grfn = self.gen_grfn(node.body, function_state, "functiondef")
        # Get the function_reference_spec, function_assign_spec and
        # identifier_spec for the function
        function_reference_grfn, function_assign_grfn, identifier_grfn = \
            get_body_and_functions(body_grfn)

        # This function removes all variables related to I/O from the
        # variable list. This will be done until a specification for I/O is
        # defined in GrFN
        variables = remove_io_variables(
                        list(local_last_definitions.keys()))

        function_container_grfn = {
            "name": node.name,
            "type": "container",
            "input": [
                {"name": arg, "domain": local_variable_types[arg]} for arg in
                argument_list
            ],
            "variables": [
                {"name": var, "domain": local_variable_types[var]}
                for var in variables
            ],
            "body": function_reference_grfn,
        }

        function_assign_grfn.append(function_container_grfn)

        pgm = {"functions": function_assign_grfn,
               "identifiers": identifier_grfn}

        return [pgm]

    def process_arguments(self, node, state, call_source):
        """
            This function returns a list of arguments defined in a function
            definition. `node.args` is a list of `arg` nodes which are
            iteratively processed to get the argument name.
        """
        return [
            self.gen_grfn(arg, state, call_source)
            for arg in node.args
        ]

    @staticmethod
    def process_arg(node, state, call_source):
        """
            This function processes a function argument.
        """
        # Variables are declared as List() objects in the intermediate Python
        # representation in order to mimic the pass-by-reference property of
        # Fortran. So, arguments have `annotations` which hold the type() of
        # the variable i.e. x[Int], y[Float], etc.
        assert (
            node.annotation
        ), "Found argument without annotation. This should not happen."
        state.variable_types[node.arg] = get_variable_type(node.annotation)

        if state.last_definitions.get(node.arg) is None:
            if call_source == "functiondef":
                state.last_definitions[node.arg] = -1
            else:
                assert False, ("Call source is not ast.FunctionDef. "
                               "Handle this by setting state.last_definitions["
                               "node.arg] = 0 in place of the assert False. "
                               "But this case should not occur in general.")
        else:
            assert False, ("The argument variable was already defined "
                           "resulting in state.last_definitions containing an "
                           "entry to this argument. Resolve this by setting "
                           "state.last_definitions[node.arg] += 1. But this "
                           "case should not occur in general.")

        return node.arg

    def process_index(self, node, state):
        """
            This function handles the Index node of the ast. The Index node
            is a `slice` value which appears when a `[]` indexing occurs.
            For example: x[Real], a[0], etc. So, the `value` of the index can
            either be an ast.Name (x[Real]) or an ast.Num (a[0]), or any
            other ast type. So, we forward the `value` to its respective ast
            handler.
        """
        self.gen_grfn(node.value, state, "index")

    @staticmethod
    def process_num(node):
        """
            This function handles the ast.Num of the ast tree. This node only
            contains a numeric value in its body. For example: Num(n=0),
            Num(n=17.27), etc. So, we return the numeric value in a
            <function_assign_body_literal_spec> form.

        """
        # TODO: According to new specification, the following structure
        #  should be used: {"type": "literal, "value": {"dtype": <type>,
        #  "value": <value>}}. Confirm with Clay.
        return [
            {"type": "literal", "dtype": getDType(node.n), "value": node.n}
        ]

    def process_list_ast(self, node, state):
        """
            This function handles ast.List which represents Python lists. The
            ast.List has an `elts` element which is a list of all the elements
            of the list. This is most notably encountered in annotated
            assignment of variables to [None] (Example: day: List[int] = [
            None]). This is handled by calling `gen_grfn` on every element of
            the list i.e. every element of `elts`.
        """
        elements = [
            element[0]
            for element in [
                self.gen_grfn(list_element, state, "List")
                for list_element in node.elts
            ]
        ]
        return elements if len(elements) == 1 else [{"list": elements}]

    @staticmethod
    def process_str(node):
        """
            This function handles the ast.Str of the ast tree. This node only
            contains a string value in its body. For example: Str(s='lorem'),
            Str(s='Estimate: '), etc. So, we return the string value in a
            <function_assign_body_literal_spec> form where the dtype is a
            string.
        """
        # TODO: According to new specification, the following structure
        #  should be used: {"type": "literal, "value": {"dtype": <type>,
        #  "value": <value>}}. Confirm with Clay.
        return [
            {"type": "literal", "dtype": "string", "value": node.s}
        ]

    def process_for(self, node, state, call_source):
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        scope_path.append("loop")

        state = state.copy(
            last_definitions=state.last_definitions.copy(),
            next_definitions=state.next_definitions.copy(),
            last_definition_default=state.last_definition_default,
            function_name=state.function_name,
            variable_types=state.variable_types.copy(),
            lambda_strings=state.lambda_strings,
            start=state.start.copy(),
            scope_path=scope_path,
        )

        if self.gen_grfn(node.orelse, state, "for"):
            raise For2PyError("For/Else in for not supported.")

        indexVar = self.gen_grfn(node.target, state, "for")
        if len(indexVar) != 1 or "var" not in indexVar[0]:
            raise For2PyError("Only one index variable is supported.")
        indexName = indexVar[0]["var"]["variable"]
        loopIter = self.gen_grfn(node.iter, state, "for")
        if (
                len(loopIter) != 1
                or "call" not in loopIter[0]
                or loopIter[0]["call"]["function"] != "range"
        ):
            raise For2PyError("Can only iterate over a range.")

        rangeCall = loopIter[0]["call"]
        if (
                len(rangeCall["inputs"]) != 2
                or len(rangeCall["inputs"][0]) != 1
                or len(rangeCall["inputs"][1]) != 1
                or (
                "type" in rangeCall["inputs"][0]
                and rangeCall["inputs"][0]["type"] == "literal"
        )
                or (
                "type" in rangeCall["inputs"][1]
                and rangeCall["inputs"][1]["type"] == "literal"
        )
        ):
            raise For2PyError("Can only iterate over a constant range.")

        if len(rangeCall["inputs"]) == 2:
            iterationRange = {
                "start": rangeCall["inputs"][0][0],
                "end": rangeCall["inputs"][1][0],
            }
        elif len(rangeCall["inputs"]) == 3:
            iterationRange = {
                "start": rangeCall["inputs"][0][0],
                "end": rangeCall["inputs"][1][0],
                "step": rangeCall["inputs"][2][0],
            }

        loopLastDef = {}
        loopState = state.copy(
            last_definitions=loopLastDef, next_definitions={},
            last_definition_default=-1
        )

        loopState.last_definitions[indexName] = None

        loop = self.gen_grfn(node.body, loopState, "for")

        loopBody, loopFns, iden_spec = get_body_and_functions(loop)

        # If loopLastDef[x] == 0, this means that the variable was not
        # declared before the loop and is being declared/defined within
        # the loop. So we need to remove that from the variable_list
        variable_list = [x for x in loopLastDef if (x != indexName and
                                                    state.last_definitions[
                                                        x] !=
                                                    0)]

        variables = [
            {"name": variable, "domain": state.variable_types[variable]}
            for variable in variable_list
        ]

        # Removing the indexing of the loop index variable from the loopName
        # loopName = get_function_name(
        #     f"{state.function_name}__loop_plate__{
        #     indexName}", {}
        # )

        loopName = state.function_name + "__loop_plate__" + indexName

        loopFn = {
            "name": loopName,
            "type": "loop_plate",
            "input": variables,
            "index_variable": indexName,
            "index_iteration_range": iterationRange,
            "body": loopBody,
        }

        id_specList = self.make_identifier_spec(
            loopName, indexName, {}, state
        )
        loopCall = {
            "name": loopName,
            "input": [
                {
                    "name": variable,
                    "index": loopState.last_definitions[variable]
                }
                for variable in variable_list
            ],
            "output": {}
        }

        pgm = {
            "functions": loopFns + [loopFn],
            "body": [loopCall],
            "identifiers": [],
        }

        for id_spec in id_specList:
            pgm["identifiers"].append(id_spec)

        return [pgm]

    def process_if(self, node, state, call_source):
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        scope_path.append("if")

        state.scope_path = scope_path

        if call_source == "if":
            pgm = {"functions": [], "body": [], "identifiers": []}

            condSrcs = self.gen_grfn(node.test, state, "if")

            startDefs = state.last_definitions.copy()
            ifDefs = startDefs.copy()
            elseDefs = startDefs.copy()
            ifState = state.copy(last_definitions=ifDefs)
            elseState = state.copy(last_definitions=elseDefs)
            ifPgm = self.gen_grfn(node.body, ifState, "if")
            elsePgm = self.gen_grfn(node.orelse, elseState, "if")

            updatedDefs = [
                var
                for var in set(startDefs.keys())
                    .union(ifDefs.keys())
                    .union(elseDefs.keys())
                if var not in startDefs
                   or ifDefs[var] != startDefs[var]
                   or elseDefs[var] != startDefs[var]
            ]

            pgm["functions"] += reduce(
                (lambda x, y: x + y["functions"]), [[]] + ifPgm
            ) + reduce((lambda x, y: x + y["functions"]), [[]] + elsePgm)

            pgm["body"] += reduce(
                (lambda x, y: x + y["body"]), [[]] + ifPgm
            ) + reduce((lambda x, y: x + y["body"]), [[]] + elsePgm)

            self.elif_grfn = [
                pgm,
                condSrcs,
                node.test,
                node.lineno,
                node,
                updatedDefs,
                ifDefs,
                state,
            ]
            return []

        pgm = {"functions": [], "body": [], "identifiers": []}

        condSrcs = self.gen_grfn(node.test, state, "if")

        # Making the index of IF_X_X start from 1 instead of 2
        condNum = state.next_definitions.get("#cond",
                                             state.last_definition_default
                                             + 2)
        state.next_definitions["#cond"] = condNum + 1

        condName = f"IF_{condNum}"
        state.variable_types[condName] = "bool"
        state.last_definitions[condName] = 0
        function_name = get_function_name(self.function_names,
                                          f"{state.function_name}__condition__"
                                          f"{condName}", {})

        # The condName is of the form 'IF_1' and the index holds the
        # ordering of the condName. This means that index should increment
        # of every new 'IF' statement. Previously, it was set to set to 0.
        # But, 'function_name' holds the current index of 'condName'
        # So, extract the index from 'function_name'.
        # condOutput = {"variable": condName, "index": 0}
        condOutput = {"variable": condName, "index": int(function_name[-1])}

        fn = {
            "name": function_name,
            "type": "condition",
            "target": condName,
            "reference": node.lineno,
            "sources": [
                {"name": src["var"]["variable"], "type": "variable"}
                for src in condSrcs
                if "var" in src
            ],
        }

        id_specList = self.make_identifier_spec(
            function_name,
            condOutput,
            [src["var"] for src in condSrcs if "var" in src],
            state,
        )

        for id_spec in id_specList:
            pgm["identifiers"].append(id_spec)

        body = {
            "name": function_name,
            "output": condOutput,
            "input": [src["var"] for src in condSrcs if "var" in src],
        }

        pgm["functions"].append(fn)
        pgm["body"].append(body)
        lambda_string = genFn(
            node.test,
            function_name,
            None,
            [src["var"]["variable"] for src in condSrcs if "var" in src],
            state,
        )
        state.lambda_strings.append(lambda_string)
        startDefs = state.last_definitions.copy()
        ifDefs = startDefs.copy()
        elseDefs = startDefs.copy()
        ifState = state.copy(last_definitions=ifDefs)
        elseState = state.copy(last_definitions=elseDefs)
        ifPgm = self.gen_grfn(node.body, ifState, "if")
        elsePgm = self.gen_grfn(node.orelse, elseState, "if")

        pgm["functions"] += reduce(
            (lambda x, y: x + y["functions"]), [[]] + ifPgm
        ) + reduce((lambda x, y: x + y["functions"]), [[]] + elsePgm)

        pgm["body"] += reduce(
            (lambda x, y: x + y["body"]), [[]] + ifPgm
        ) + reduce((lambda x, y: x + y["body"]), [[]] + elsePgm)

        updatedDefs = [
            var
            for var in set(startDefs.keys())
                .union(ifDefs.keys())
                .union(elseDefs.keys())
            if var not in startDefs
               or ifDefs[var] != startDefs[var]
               or elseDefs[var] != startDefs[var]
        ]

        defVersions = {
            key: [
                version
                for version in [
                    startDefs.get(key),
                    ifDefs.get(key),
                    elseDefs.get(key),
                ]
                if version is not None
            ]
            for key in updatedDefs
        }

        for updatedDef in defVersions:
            versions = defVersions[updatedDef]
            inputs = (
                [
                    condOutput,
                    {"variable": updatedDef, "index": versions[-1]},
                    {"variable": updatedDef, "index": versions[-2]},
                ]
                if len(versions) > 1
                else [
                    condOutput,
                    {"variable": updatedDef, "index": versions[0]},
                ]
            )

            output = {
                "variable": updatedDef,
                "index": getNextDef(
                    updatedDef,
                    state.last_definitions,
                    state.next_definitions,
                    state.last_definition_default,
                ),
            }

            function_name = get_function_name(
                self.function_names,
                f"{state.function_name}__decision__{updatedDef}", output
            )

            fn = {
                "name": function_name,
                "type": "decision",
                "target": updatedDef,
                "reference": node.lineno,
                "sources": [
                    {
                        "name": f"{var['variable']}_{var['index']}",
                        "type": "variable",
                    }
                    for var in inputs
                ],
            }

            # Check for buggy __decision__ tag containing of only IF_ blocks
            # More information required on how __decision__ tags are made
            # This seems to be in development phase and documentation is
            # missing from the GrFN spec as well. Actual removal (or not)
            # of this tag depends on further information about this

            if "IF_" in updatedDef:
                count = 0
                for var in inputs:
                    if "IF_" in var["variable"]:
                        count += 1
                if count == len(inputs):
                    continue

            body = {"name": function_name, "output": output, "input":
                inputs}

            id_specList = self.make_identifier_spec(
                function_name, output, inputs, state
            )

            for id_spec in id_specList:
                pgm["identifiers"].append(id_spec)

            lambda_string = genFn(
                node,
                function_name,
                updatedDef,
                [f"{src['variable']}_{src['index']}" for src in inputs],
                state,
            )
            state.lambda_strings.append(lambda_string)

            pgm["functions"].append(fn)
            pgm["body"].append(body)

        # Previous ELIF Block is filled??
        if len(self.elif_grfn) > 0:

            condSrcs = self.elif_grfn[1]

            for item in self.elif_grfn[0]["functions"]:
                pgm["functions"].append(item)

            for item in self.elif_grfn[0]["body"]:
                pgm["body"].append(item)

            state.next_definitions["#cond"] = condNum + 1

            condName = f"IF_{condNum}"
            state.variable_types[condName] = "bool"
            state.last_definitions[condName] = 0
            function_name = get_function_name(
                self.function_names,
                f"{state.function_name}__condition__{condName}", {}
            )
            condOutput = {
                "variable": condName,
                "index": int(function_name[-1])
            }

            fn = {
                "name": function_name,
                "type": "condition",
                "target": condName,
                "reference": self.elif_grfn[3],
                "sources": [
                    {
                        "name": src["var"]["variable"],
                        "type": "variable",
                    }
                    for src in condSrcs
                    if "var" in src
                ],
            }

            id_specList = self.make_identifier_spec(
                function_name,
                condOutput,
                [src["var"] for src in condSrcs if "var" in src],
                state,
            )

            for id_spec in id_specList:
                pgm["identifiers"].append(id_spec)

            body = {
                "name": function_name,
                "output": condOutput,
                "input": [
                    src["var"] for src in condSrcs if "var" in src
                ],
            }
            pgm["functions"].append(fn)
            pgm["body"].append(body)

            lambda_string = genFn(
                self.elif_grfn[2],
                function_name,
                None,
                [
                    src["var"]["variable"]
                    for src in condSrcs
                    if "var" in src
                ],
                state,
            )
            state.lambda_strings.append(lambda_string)

            startDefs = state.last_definitions.copy()
            ifDefs = self.elif_grfn[6]
            elseDefs = startDefs.copy()

            updatedDefs = self.elif_grfn[5]

            defVersions = {
                key: [
                    version
                    for version in [
                        startDefs.get(key),
                        ifDefs.get(key),
                        elseDefs.get(key),
                    ]
                    if version is not None
                ]
                for key in updatedDefs
            }

            for updatedDef in defVersions:
                versions = defVersions[updatedDef]
                inputs = (
                    [
                        condOutput,
                        {
                            "variable": updatedDef,
                            "index": versions[-1],
                        },
                        {
                            "variable": updatedDef,
                            "index": versions[-2],
                        },
                    ]
                    if len(versions) > 1
                    else [
                        condOutput,
                        {"variable": updatedDef, "index": versions[0]},
                    ]
                )

                output = {
                    "variable": updatedDef,
                    "index": getNextDef(
                        updatedDef,
                        state.last_definitions,
                        state.next_definitions,
                        state.last_definition_default,
                    ),
                }
                function_name = get_function_name(
                    self.function_names,
                    f"{state.function_name}__decision__{updatedDef}",
                    output,
                )
                fn = {
                    "name": function_name,
                    "type": "decision",
                    "target": updatedDef,
                    "reference": self.elif_grfn[3],
                    "sources": [
                        {
                            "name": f"{var['variable']}_{var['index']}",
                            "type": "variable",
                        }
                        for var in inputs
                    ],
                }

                # Check for buggy __decision__ tag containing of only
                # IF_ blocks. More information required on how
                # __decision__ tags are made.
                # This seems to be in development phase and documentation is
                # missing from the GrFN spec as well. Actual removal
                # (or not) of this tag depends on further information
                # about this

                if "IF_" in updatedDef:
                    count = 0
                    for var in inputs:
                        if "IF_" in var["variable"]:
                            count += 1
                    if count == len(inputs):
                        continue

                body = {
                    "name": function_name,
                    "output": output,
                    "input": inputs,
                }

                id_specList = self.make_identifier_spec(
                    function_name, output, inputs, state
                )

                for id_spec in id_specList:
                    pgm["identifiers"].append(id_spec)

                lambda_string = genFn(
                    self.elif_grfn[4],
                    function_name,
                    updatedDef,
                    [
                        f"{src['variable']}_{src['index']}"
                        for src in inputs
                    ],
                    state,
                )
                state.lambda_strings.append(lambda_string)

                pgm["functions"].append(fn)
                pgm["body"].append(body)

            self.elif_grfn = []

        return [pgm]

    def process_unary_operation(self, node, state, call_source):
        return self.gen_grfn(node.operand, state, "unaryop")

    def process_binary_operation(self, node, state,
                                 call_source):
        if isinstance(node.left, ast.Num) and isinstance(
                node.right, ast.Num
        ):
            for op in BINOPS:
                if isinstance(node.op, op):
                    val = BINOPS[type(node.op)](node.left.n, node.right.n)
                    return [
                        {
                            "value": val,
                            "dtype": getDType(val),
                            "type": "literal",
                        }
                    ]

        opPgm = self.gen_grfn(
            node.left, state, "binop"
        ) + self.gen_grfn(node.right, state, "binop")
        return opPgm

        return opPgm

    def process_unnecessary_types(self, node, state,
                                  call_source):
        t = node.__repr__().split()[0][2:]
        sys.stdout.write(f"Found {t}, which should be unnecessary\n")

    def process_expression(self, node, state, call_source):
        exprs = self.gen_grfn(node.value, state, "expr")
        pgm = {"functions": [], "body": [], "identifiers": []}
        for expr in exprs:
            if "call" in expr:
                call = expr["call"]
                body = {
                    "function": call["function"],
                    "output": {},
                    "input": [],
                }
                if re.match(r"file_\d+\.write", body["function"]):
                    return []
                for arg in call["inputs"]:
                    if len(arg) == 1:
                        if "var" in arg[0]:
                            body["input"].append(arg[0]["var"])
                    else:
                        raise For2PyError(
                            "Only 1 input per argument supported right now."
                        )
                pgm["body"].append(body)
            else:
                raise For2PyError(f"Unsupported expr: {expr}.")
        return [pgm]

    def process_compare(self, node, state, call_source):
        return self.gen_grfn(
            node.left, state, "compare"
        ) + self.gen_grfn(node.comparators, state, "compare")

    def process_subscript(self, node, state, call_source):
        if not isinstance(node.slice.value, ast.Num):
            raise For2PyError("can't handle arrays right now.")

        val = self.gen_grfn(node.value, state, "subscript")
        if val:
            if val[0]["var"]["variable"] in self.annotated_assigned:
                if isinstance(node.ctx, ast.Store):
                    val[0]["var"]["index"] = getNextDef(
                        val[0]["var"]["variable"],
                        state.last_definitions,
                        state.next_definitions,
                        state.last_definition_default,
                    )
            elif val[0]["var"]["index"] == -1:
                if isinstance(node.ctx, ast.Store):
                    val[0]["var"]["index"] = getNextDef(
                        val[0]["var"]["variable"],
                        state.last_definitions,
                        state.next_definitions,
                        state.last_definition_default,
                    )
                    self.annotated_assigned.append(val[0]["var"]["variable"])
            else:
                self.annotated_assigned.append(val[0]["var"]["variable"])

        return val

    def process_name(self, node, state, call_source):
        if not re.match(r'i_g_n_o_r_e___m_e__.*', node.id):
            lastDef = getLastDef(node.id, state.last_definitions,
                                 state.last_definition_default)
            if (
                    isinstance(node.ctx, ast.Store)
                    and state.next_definitions.get(node.id)
                    and call_source != "annassign"
            ):
                lastDef = getNextDef(
                    node.id,
                    state.last_definitions,
                    state.next_definitions,
                    state.last_definition_default,
                )

            return [{"var": {"variable": node.id, "index": lastDef}}]

    def process_annotated_assign(self, node, state,
                                 call_source):
        if isinstance(node.value, ast.List):
            targets = self.gen_grfn(node.target, state,
                                  "annassign")
            for target in targets:
                state.variable_types[target["var"]["variable"]] = \
                    get_variable_type(
                        node.annotation
                    )
                if target["var"]["variable"] not in self.annotated_assigned:
                    self.annotated_assigned.append(target["var"]["variable"])
            return []

        sources = self.gen_grfn(node.value, state,
                              "annassign")
        targets = self.gen_grfn(node.target, state,
                              "annassign")

        pgm = {"functions": [], "body": [], "identifiers": []}

        for target in targets:
            state.variable_types[target["var"]["variable"]] = \
                get_variable_type(
                    node.annotation
                )
            name = get_function_name(
                self.function_names,
                f"{state.function_name}__assign__"
                f"{target['var']['variable']}",
                {},
            )
            fn = self.make_fn_dict(name, target, sources, node)
            body = self.make_body_dict(name, target, sources, state)

            if len(sources) > 0:
                lambda_string = genFn(
                    node,
                    name,
                    target["var"]["variable"],
                    [
                        src["var"]["variable"]
                        for src in sources
                        if "var" in src
                    ],
                    state,
                )
                state.lambda_strings.append(lambda_string)

            # In the case of assignments of the form: "ud: List[float]"
            # an assignment function will be created with an empty input
            # list. Also, the function dictionary will be empty. We do
            # not want such assignments in the GrFN so check for an empty
            # <fn> dictionary and return [] if found
            if len(fn) == 0:
                return []
            if not fn["sources"] and len(sources) == 1:
                fn["body"] = {
                    "type": "literal",
                    "dtype": sources[0]["dtype"],
                    "value": f"{sources[0]['value']}",
                }

            for id_spec in body[1]:
                pgm["identifiers"].append(id_spec)

            pgm["functions"].append(fn)
            pgm["body"].append(body[0])

        return [pgm]

    def process_assign(self, node, state, call_source):
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        state.scope_path = scope_path

        sources = self.gen_grfn(node.value, state, "assign")

        targets = reduce(
            (lambda x, y: x.append(y)),
            [
                self.gen_grfn(target, state, "assign")
                for target in node.targets
            ],
        )

        pgm = {"functions": [], "body": [], "identifiers": []}

        for target in targets:
            source_list = []
            if target.get("list"):
                targets = ",".join(
                    [x["var"]["variable"] for x in target["list"]]
                )
                target = {"var": {"variable": targets, "index": 1}}

            # Check whether this is an alias assignment i.e. of the form
            # y=x where y is now the alias of variable x
            self.check_alias(target, sources)

            # state.variable_types[target["var"]["variable"]] =
            # get_variable_type(
            #     node.annotation)

            name = get_function_name(
                self.function_names,
                f"{state.function_name}__assign__"
                f"{target['var']['variable']}",
                target,
            )

            fn = self.make_fn_dict(name, target, sources, node)
            if len(fn) == 0:
                return []
            body = self.make_body_dict(name, target, sources, state)

            source_list = self.make_source_list_dict(sources)

            lambda_string = genFn(
                node, name, target["var"]["variable"], source_list, state
            )
            state.lambda_strings.append(lambda_string)
            if not fn["sources"] and len(sources) == 1:
                if sources[0].get("list"):
                    dtypes = set()
                    value = list()
                    for item in sources[0]["list"]:
                        dtypes.add(item["dtype"])
                        value.append(item["value"])
                    dtype = list(dtypes)
                elif sources[0].get("call") and \
                        sources[0]["call"]["function"] == "Float32":
                    dtype = sources[0]["call"]["inputs"][0][0]["dtype"]
                    value = f"{sources[0]['call']['inputs'][0][0]['value']}"
                else:
                    dtype = sources[0]["dtype"]
                    value = f"{sources[0]['value']}"
                fn["body"] = {
                    "type": "literal",
                    "dtype": dtype,
                    "value": value,
                }

            for id_spec in body[1]:
                pgm["identifiers"].append(id_spec)

            pgm["functions"].append(fn)
            pgm["body"].append(body[0])

        return [pgm]

    def process_tuple(self, node, state, call_source):
        elements = []
        for element in [
            self.gen_grfn(elmt, state, "ctx") for elmt in
            node.elts
        ]:
            elements.append(element[0])

        return elements if len(elements) == 1 else [{"list": elements}]

    def process_call(self, node, state, call_source):
        if isinstance(node.func, ast.Attribute):
            # Check if there is a <sys> call. Bypass it if exists.
            if isinstance(node.func.value, ast.Attribute):
                if node.func.value.value.id == "sys":
                    return []
            fnNode = node.func
            module = fnNode.value.id
            function_name = fnNode.attr
            function_name = module + "." + function_name
        else:
            function_name = node.func.id
        inputs = []

        for arg in node.args:
            arg = self.gen_grfn(arg, state, "call")
            inputs.append(arg)

        call = {"call": {"function": function_name, "inputs": inputs}}

        return [call]

    def process_module(self, node, state, call_source):
        pgms = []
        for cur in node.body:
            pgm = self.gen_grfn(cur, state, "module")
            pgms += pgm
        return [mergeDicts(pgms)]

    def process_boolean_operation(self, node, state,
                                  call_source):
        pgms = []
        boolOp = {ast.And: "and", ast.Or: "or"}

        for key in boolOp:
            if isinstance(node.op, key):
                pgms.append([{"boolOp": boolOp[key]}])

        for item in node.values:
            pgms.append(self.gen_grfn(item, state, "boolop"))

        return pgms

    def process_attribute(self, node, state, call_source):
        # Handle Attributes
        # This is a fix on `feature_save` branch to bypass the SAVE statement
        # feature where a SAVEd variable is referenced as
        # <function_name>.<variable_name>. So the code below only returns the
        # <variable_name> which is stored under `node.attr`. The `node.id`
        # stores the <function_name> which is being ignored.

        # When a computations float value is extracted using the Float32
        # class's _val method, an ast.Attribute will be present, just
        if node.attr == "_val":
            return self.gen_grfn(node.value, state,
                               call_source)
        else:
            lastDef = getLastDef(node.attr, state.last_definitions,
                                 state.last_definition_default)

            return [{"var": {"variable": node.attr, "index": lastDef}}]

    def process_ast(self, node, state, call_source):
        sys.stderr.write(
            f"No handler for AST.{node.__class__.__name__} in gen_grfn, "
            f"fields: {node._fields}\n"
        )

    def process_nomatch(self, node, state, call_source):
        sys.stderr.write(
            f"No handler for {node.__class__.__name__} in gen_grfn, "
            f"value: {str(node)}\n"
        )



    # This function checks whether an assignment is an alias created. An alias
    # is created when an assignment of the form y=x happens such that y is now
    # an alias of x because it is an exact copy of x. If it is an alias
    # assignment, the dictionary alias_mapper will get populated.
    def check_alias(self, target, sources):
        target_index = (
            target["var"]["variable"] + "_" + str(target["var"]["index"])
        )
        if len(sources) == 1 and sources[0].get("var") != None:
            if self.alias_mapper.get(target_index):
                self.alias_mapper[target_index].append(
                    sources[0]["var"]["variable"]
                    + "_"
                    + str(sources[0]["var"]["index"])
                )
            else:
                self.alias_mapper[target_index] = [
                    sources[0]["var"]["variable"]
                    + "_"
                    + str(sources[0]["var"]["index"])
                ]

    def make_iden_dict(self, name, targets, scope_path, holder):
        # Check for aliases
        if isinstance(targets, dict):
            aliases = self.alias_mapper.get(
                targets["variable"] + "_" + str(targets["index"]), "None"
            )
        elif isinstance(targets, str):
            aliases = self.alias_mapper.get(targets, "None")

        # First, check whether the information is from a variable or a
        # holder(assign, loop, if, etc). Assign the base_name accordingly

        if holder == "body":
            # If we are making the identifier specification of a body holder,
            # the base_name will be the holder
            if isinstance(targets, dict):
                base_name = (
                    name
                    + "$"
                    + targets["variable"]
                    + "_"
                    + str(targets["index"])
                )
            elif isinstance(targets, str):
                base_name = name + "$" + targets
            gensyms_tag = "h"

        elif holder == "variable":
            # The base name will just be the name of the identifier
            base_name = targets
            gensyms_tag = "v"

        # The name space should get the entire directory scope of the fortran
        # file under which it is defined. For PETASCE.for, all modules are
        # defined in the same fortran file so the namespace will be the same
        # for all identifiers

        # TODO handle multiple file namespaces that handle multiple fortran
        #  file namespacing

        # TODO Is the namespace path for the python intermediates or the
        #  original FORTRAN code? Currently, it captures the intermediate
        #  python file's path
        name_space = self.mode_mapper["file_name"][1].split("/")
        name_space = ".".join(name_space)

        # The scope captures the scope within the file where it exists. The
        # context of modules can be implemented here.
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        elif scope_path[0] == "_TOP" and len(scope_path) > 1:
            scope_path.remove("_TOP")
        scope_path = ".".join(scope_path)

        # TODO Source code reference: This is the line number in the Python
        # (or FORTRAN?) file. According to meeting on the 21st Feb, 2019,
        # this was the same as namespace. Exactly same though? Need clarity.

        source_reference = name_space

        iden_dict = {
            "base_name": base_name,
            "scope": scope_path,
            "namespace": name_space,
            "aliases": aliases,
            "source_references": source_reference,
            "gensyms": generage_gensysm(gensyms_tag),
        }

        return iden_dict

    # Create the identifier specification for each identifier
    def make_identifier_spec(self, name, targets, sources, state):
        scope_path = state.scope_path
        for_id = 1
        if_id = 1
        identifier_list = []

        for item, scope in enumerate(scope_path):
            if scope == "loop":
                scope_path[item] = scope + "$" + str(for_id)
                for_id += 1
            elif scope == "if":
                scope_path[item] = scope + "$" + str(for_id)
                if_id += 1

        # Identify which kind of identifier it is
        name_regex = r"(?P<scope>\w+)__(?P<type>\w+)__(?P<basename>\w+)"
        match = re.match(name_regex, name)
        if match:
            if match.group("type") == "assign":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "condition":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "loop_plate":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "decision":
                iden_dict = self.make_iden_dict(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_iden_dict(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)

            return identifier_list

    def make_body_dict(self, name, target, sources, state):
        source_list = []
        for src in sources:
            if "var" in src:
                source_list.append(src["var"])
            if "call" in src:
                source_list.extend(self.make_call_index_dict(src))

        # Removing duplicates
        unique_source = []
        [unique_source.append(obj) for obj in source_list if obj not in
         unique_source]
        source_list = unique_source

        id_spec = self.make_identifier_spec(
            name, target["var"], source_list, state
        )

        body = {"name": name, "output": target["var"], "input": source_list}
        return [body, id_spec]

    def make_call_index_dict(self, source):
        source_list = []
        for item in source["call"]["inputs"]:
            for ip in item:
                if "var" in ip:
                    source_list.append(ip["var"])
                elif "call" in ip:
                    source_list.extend(self.make_call_index_dict(ip))

        return source_list

    def make_source_list_dict(self, sourceDict):
        source_list = []
        for src in sourceDict:
            if "var" in src:
                source_list.append(src["var"]["variable"])
            elif "call" in src:
                for ip in src["call"]["inputs"]:
                    source_list.extend(self.make_source_list_dict(ip))

        # Removing duplicates
        unique_source = []
        [unique_source.append(obj) for obj in source_list if obj not in
         unique_source]
        source_list = unique_source

        return source_list


    def make_fn_dict(self, name, target, sources, node):
        source = []
        fn = {}

        # Preprocessing and removing certain Assigns which only pertain to the
        # Python code and do not relate to the FORTRAN code in any way.
        bypass_match_target = RE_BYPASS_IO.match(target["var"][ "variable"])

        if bypass_match_target:
            self.exclude_list.append(target["var"]["variable"])
            return fn

        for src in sources:
            if "call" in src:
                # Bypassing identifiers who have I/O constructs on their source
                # fields too.s
                # Example: (i[0],) = format_10_obj.read_line(file_10.readline())
                # 'i' is bypassed here
                # TODO this is only for PETASCE02.for. Will need to include 'i'
                #  in the long run
                bypass_match_source = RE_BYPASS_IO.match(src["call"][
                                                              "function"])
                if bypass_match_source:
                    if "var" in src:
                        self.exclude_list.append(src["var"]["variable"])
                    self.exclude_list.append(target["var"]["variable"])
                    return fn
                for source_ins in make_call_body_dict(src):
                    source.append(source_ins)
            if "var" in src:
                variable = src["var"]["variable"]
                source.append({"name": variable, "type": "variable"})

            # Removing duplicates
            unique_source = []
            [unique_source.append(obj) for obj in source if obj not in
             unique_source]
            source = unique_source

            if re.match(r"\d+", target["var"]["variable"]) and "list" in src:
                # This is a write to a file
                return fn
            fn = {
                "name": name,
                "type": "assign",
                "target": target["var"]["variable"],
                "sources": source,
                "reference": node.lineno,
            }

        return fn

    def process_load(self, node, state, call_source):
        raise For2PyError(f"Found ast.Load, which should not happen. "
                          f"From source: {call_source}")

    def process_store(self, node, state, call_source):
        raise For2PyError(f"Found ast.Store, which should not happen. "
                          f"From source: {call_source}")


def process_decorators(node, state):
    """
        Go through each decorator and extract relevant information.
        Currently this function only checks for the static_vars decorator
        for the SAVEd variables and updates variable_types with the data
        type of each variable.
    """
    for decorator in node:
        decorator_function_name = decorator.func.id
        if decorator_function_name == "static_vars":
            for arg in decorator.args[0].elts:
                variable = arg.values[0].s
                variable_type = arg.values[2].s
                state.variable_types[variable] = REVERSE_ANNOTATE_MAP[variable_type]


def genFn(node, function_name: str, returnVal: bool, inputs, state):
    lambda_strings = []
    argument_strings = []

    # Sort the arguments in the function call as it is used in the operation
    input_list = sorted(set(inputs), key=inputs.index)

    # Add type annotations to the function arguments
    for ip in input_list:
        annotation = state.variable_types.get(ip)
        if not annotation:
            # variable_types does not contain annotations for variables for
            # indexing
            # such as 'abc_1', etc. Check if the such variables exist and
            # assign appropriate annotations
            key_match = lambda var, dicn: ([i for i in dicn if i in var])
            annotation = state.variable_types[key_match(ip,
                                                    state.variable_types)[0]]
        annotation = ANNOTATE_MAP[annotation]
        argument_strings.append(f"{ip}: {annotation}")

    lambda_strings.append(
        f"def {function_name}({', '.join(argument_strings)}):\n    "
    )
    # If a `decision` tag comes up, override the call to genCode to manually
    # enter the python script for the lambda file.
    if "__decision__" in function_name:
        code = f"{inputs[2]} if {inputs[0]} else {inputs[1]}"
    else:
        code = genCode(node, PrintState("\n    "))
    if returnVal:
        lambda_strings.append(f"return {code}")
    else:
        lines = code.split("\n")
        indent = re.search("[^ ]", lines[-1]).start()
        lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
        lambda_strings.append("\n".join(lines))
    lambda_strings.append("\n\n")
    return "".join(lambda_strings)


def mergeDicts(dicts: Iterable[Dict]) -> Dict:
    fields = set(chain.from_iterable(d.keys() for d in dicts))

    merged_dict = {field: [] for field in fields}
    for field, d in product(fields, dicts):
        if field in d:
            if isinstance(d[field], list):
                merged_dict[field] += d[field]
            else:
                merged_dict[field].append(d[field])

    return merged_dict


def get_function_name(function_names, basename, target):
    # First, check whether the basename is a 'decision' block. If it is, we
    # need to get it's index from the index of its corresponding identifier's
    # 'assign' block. We do not use the index of the 'decision' block as that
    # will not correspond with that of the 'assign' block.  For example: for
    # petpt__decision__albedo, its index will be the index of the latest
    # petpt__assign__albedo + 1

    if "__decision__" in basename:
        part_match = re.match(
            r"(?P<body>\S+)__decision__(?P<identifier>\S+)", basename
        )
        if part_match:
            new_basename = (
                part_match.group("body")
                + "__assign__"
                + part_match.group("identifier")
            )
    else:
        new_basename = basename
    fnId = function_names.get(new_basename, 0)
    if len(target) > 0:
        if target.get("var"):
            fnId = target["var"]["index"]
        elif target.get("variable"):
            fnId = target["index"]
    if fnId < 0:
        fnId = function_names.get(new_basename, 0)
    function_name = f"{basename}_{fnId}"
    function_names[basename] = fnId + 1
    return function_name


def getLastDef(var, last_definitions, last_definition_default):
    index = last_definition_default

    # Preprocessing and removing certain Assigns which only pertain to the
    # Python code and do not relate to the FORTRAN code in any way.
    bypass_match = RE_BYPASS_IO.match(var)

    if not bypass_match:
        if var in last_definitions:
            index = last_definitions[var]
        else:
            last_definitions[var] = index
        return index
    else:
        return 0


def getNextDef(var, last_definitions, next_definitions, last_definition_default):
    index = next_definitions.get(var, last_definition_default + 1)
    next_definitions[var] = index + 1
    last_definitions[var] = index
    return index


def get_variable_type(annNode):
    # wrapped in list
    if isinstance(annNode, ast.Subscript):
        dType = annNode.slice.value.id
    else:
        dType = annNode.id
    try:
        if REVERSE_ANNOTATE_MAP.get(dType):
            return REVERSE_ANNOTATE_MAP[dType]
        else:
            sys.stderr.write(
                "Unsupported type (only float, int, list, and str "
                "supported as of now).\n"
            )
    except AttributeError:
        raise For2PyError("Unsupported type (annNode is None).")


def getDType(val):
    if isinstance(val, int):
        dtype = "integer"
    elif isinstance(val, float):
        dtype = "real"
    elif isinstance(val, str):
        dtype = "string"
    else:
        raise For2PyError(f"num: {type(val)}.")
    return dtype


def get_body_and_functions(pgm):
    body = list(chain.from_iterable(stmt["body"] for stmt in pgm))
    fns = list(chain.from_iterable(stmt["functions"] for stmt in pgm))
    iden_spec = list(chain.from_iterable(stmt["identifiers"] for stmt in pgm))
    return body, fns, iden_spec


def generage_gensysm(tag):
    """
        The gensym is used to uniquely identify any identifier in the
        program. Python's uuid library is used to generate a unique 12 digit
        HEX string. The uuid4() function of 'uuid' focuses on randomness.
        Each and every bit of a UUID v4 is generated randomly and with no
        inherent logic. To every gensym, we add a tag signifying the data
        type it represents. 'v' is for variables and 'h' is for holders.
    """
    return uuid.uuid4().hex[:12] + "_" + tag


def make_call_body_dict(source):
    """
    We are going to remove addition of functions such as "max", "exp", "sin",
    etc to the source list. The following two lines when commented helps us do
    that. If user-defined functions come up as sources, some other approach
    might be required.
    """
    # TODO Try with user defined functions and see if the below two lines need
    #  to be reworked
    # name = source["call"]["function"]
    # source_list.append({"name": name, "type": "function"})

    source_list = []
    for ip in source["call"]["inputs"]:
        if isinstance(ip, list):
            for item in ip:
                if "var" in item:
                    variable = item["var"]["variable"]
                    source_list.append({"name": variable, "type": "variable"})
                elif item.get("dtype") == "string":
                    # TODO Do repetitions in this like in the above check need
                    #  to be removed?
                    source_list.append(
                        {"name": item["value"], "type": "variable"}
                    )
                elif "call" in item:
                    source_list.extend(make_call_body_dict(item))

    return source_list


def remove_io_variables(variable_list):
    """
        This function scans each variable from a list of currently defined
        variables and removes those which are related to I/O such as format
        variables, file handles, write lists and write_lines.
    """
    io_regex = re.compile(r"(format_\d+_obj)|(file_\d+)|(write_list_\d+)|"
                          r"(write_line)")
    io_match_list = [io_regex.match(var) for var in variable_list]

    return [var for var in variable_list if io_match_list[
        variable_list.index(var)] is None]


def dump_ast(node, annotate_fields=True, include_attributes=False, indent="  "):
    """
        Return a formatted dump of the tree in *node*. This is mainly useful for
        debugging purposes. The returned string will show the names and the
        values for fields. This makes the code impossible to evaluate,
        so if evaluation is wanted *annotate_fields* must be set to False.
        Attributes such as line numbers and column offsets are not dumped by
        default. If this is wanted, *include_attributes* can be set to True.
    """

    def _format(node, level=0):
        if isinstance(node, ast.AST):
            fields = [(a, _format(b, level)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend(
                    [
                        (a, _format(getattr(node, a), level))
                        for a in node._attributes
                    ]
                )
            return "".join(
                [
                    node.__class__.__name__,
                    "(",
                    ", ".join(
                        ("%s=%s" % field for field in fields)
                        if annotate_fields
                        else (b for a, b in fields)
                    ),
                    ")",
                ]
            )
        elif isinstance(node, list):
            lines = ["["]
            lines.extend(
                (
                    indent * (level + 2) + _format(x, level + 2) + ","
                    for x in node
                )
            )
            if len(lines) > 1:
                lines.append(indent * (level + 1) + "]")
            else:
                lines[-1] += "]"
            return "\n".join(lines)
        return repr(node)

    if not isinstance(node, ast.AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)
    return _format(node)


# Get the absolute path of the python files whose PGMs are being generated.
# TODO: For now the path is started from the directory "for2py" but need further
#  discussion on this
def get_path(file_name: str, instance: str):
    absolute_path = os.path.abspath(file_name)

    if instance == "namespace":
        path_match = re.match(r".*/(for2py/.*).py$", absolute_path)
    elif instance == "source":
        path_match = re.match(r".*/(for2py/.*$)", absolute_path)
    else:
        path_match = None

    if path_match:
        return path_match.group(1).split("/")
    else:
        return file_name


def create_grfn_dict(
    lambda_file: str,
    asts: List,
    file_name: str,
    mode_mapper_dict: dict,
    save_file=False,
) -> Dict:
    """
        Create a Python dict representing the GrFN, with additional metadata for
        JSON output.
    """

    lambda_string_list = [
                    "from numbers import Real\n",
                    "import delphi.translators.for2py.math_ext as math\n\n"
    ]

    state = GrFNState(lambda_string_list)
    generator = GrFNGenerator()
    generator.mode_mapper = mode_mapper_dict
    grfn = generator.gen_grfn(asts, state, "")[0]

    # If the GrFN has a `start` node, it will refer to the name of the
    # PROGRAM module which will be the entry point of the GrFN.
    if grfn.get("start"):
        grfn["start"] = grfn["start"][0]
    else:
        # TODO: If the PROGRAM module is not detected, the entry point will be
        #  the last function in the `funtction_defs` list of functions
        grfn["start"] = generator.function_definitions[-1]

    grfn["source"] = [[get_path(file_name, "source")]]

    # dateCreated stores the date and time on which the lambda and GrFN files
    # were created. It is stored in the YYYMMDD format
    grfn["dateCreated"] = f"{datetime.today().strftime('%Y%m%d')}"

    with open(lambda_file, "w") as f:
        f.write("".join(lambda_string_list))

    # View the PGM file that will be used to build a scope tree
    if save_file:
        json.dump(grfn, open(file_name[:file_name.rfind(".")] + ".json", "w"))

    return grfn


def generate_ast(filename: str):
    """
        This function generates the AST of a python file using Python's ast
        module.
    """
    return ast.parse(tokenize.open(filename).read())


def get_asts_from_files(file_list: List[str], printast=False) -> List:
    """
        This function returns the AST of each python file in the
        python_file_list.
    """
    ast_list = []
    for file in file_list:
        ast_list.append(generate_ast(file))
        if printast:
            # If the printAst flag is set, print the AST to console
            print(dump_ast(ast_list[-1]))
    return ast_list


def process_files(python_list: List[str], grfn_suffix: str, lambda_suffix:
                  str, print_ast_flag=False):
    """
        This function takes in the list of python files to convert into GrFN 
        and generates each file's AST along with starting the GrFN generation
        process. 
    """
    module_mapper = {}
    ast_list = get_asts_from_files(python_list, print_ast_flag)

    # Regular expression to identify the path and name of all python files
    filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).py")

    # First, find the main python file in order to populate the module
    # mapper
    for item in python_list:
        file_match = re.match(filename_regex, item)
        assert file_match, "Invalid filename."

        path = file_match.group("path")
        filename = file_match.group("filename")

        # Ignore all python files of modules created by `pyTranslate.py`
        # since these module files do not contain a corresponding XML file.
        if not filename.startswith("m_"):
            xml_file = f"{path}rectified_{filename}.xml"
            # Calling the `get_index` function in `mod_index_generator.py` to
            # map all variables and objects in the various files
            module_mapper = get_index(xml_file)[0]

    for index, ast_string in enumerate(ast_list):
        lambda_file = python_list[index][:-3] + "_" + lambda_suffix
        grfn_file = python_list[index][:-3] + "_" + grfn_suffix
        grfn_dict = create_grfn_dict(
            lambda_file, [ast_string], python_list[index], module_mapper
        )

        # Write each GrFN JSON into a file
        with open(grfn_file, "w") as file_handle:
            file_handle.write(json.dumps(grfn_dict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="A list of python files to generate a PGM for",
    )
    parser.add_argument(
        "-p",
        "--grfn_suffix",
        nargs=1,
        required=True,
        help="Filename for the output PGM",
    )
    parser.add_argument(
        "-l",
        "--lambda_suffix",
        nargs=1,
        required=True,
        help="Filename for output lambda functions",
    )
    parser.add_argument(
        "-o",
        "--out",
        nargs=1,
        required=True,
        help="Text file containing the list of output python files being "
             "generated",
    )
    parser.add_argument(
        "-a",
        "--print_ast",
        action="store_true",
        required=False,
        help="Print ASTs",
    )
    arguments = parser.parse_args(sys.argv[1:])

    # Read the outputFile which contains the name of all the python files
    # generated by `pyTranslate.py`. Multiple files occur in the case of
    # modules since each module is written out into a separate python file.
    with open(arguments.out[0], "r") as f:
        python_files = f.read()

    # The Python file names are space separated. Append each one to a list.
    python_file_list = python_files.rstrip().split(" ")

    grfn_suffix = arguments.grfn_suffix[0]
    lambda_suffix = arguments.lambda_suffix[0]
    print_ast = arguments.print_ast

    process_files(python_file_list, grfn_suffix, lambda_suffix, print_ast)
