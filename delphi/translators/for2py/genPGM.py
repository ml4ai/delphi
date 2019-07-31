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
    "float": "real",
    "Real": "real",
    "integer": "int",
    "int": "integer",
    "string": "str",
    "str": "string",
    "array": "[]",
    "list": "array",
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
                 function_definitions=[]
                 ):
        self.annotated_assigned = annotated_assigned
        self.function_definitions = function_definitions
        self.fortran_file = None
        self.exclude_list = []
        self.mode_mapper = {}
        self.name_mapper = {}
        self.function_names = {}
        self.outer_count = 0
        self.types = (list, ast.Module, ast.FunctionDef)
        self.elif_condition_number = None

        self.process_grfn = {
            "ast.FunctionDef": self.process_function_definition,
            "ast.arguments": self.process_arguments,
            "ast.arg": self.process_arg,
            "ast.Load": self.process_load,
            "ast.Store": self.process_store,
            "ast.Index": self.process_index,
            "ast.Num": self.process_num,
            "ast.List": self.process_list_ast,
            "ast.Str": self.process_str,
            "ast.For": self.process_for,
            "ast.If": self.process_if,
            "ast.UnaryOp": self.process_unary_operation,
            "ast.BinOp": self.process_binary_operation,
            "ast.BoolOp": self.process_boolean_operation,
            "ast.Expr": self.process_expression,
            "ast.Compare": self.process_compare,
            "ast.Subscript": self.process_subscript,
            "ast.Name": self.process_name,
            "ast.AnnAssign": self.process_annotated_assign,
            "ast.Assign": self.process_assign,
            "ast.Tuple": self.process_tuple,
            "ast.Call": self.process_call,
            "ast.Module": self.process_module,
            "ast.Attribute": self.process_attribute,
            "ast.AST": self.process_ast,
        }

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
        elif isinstance(node, list):
            return self.process_list(node, state, call_source)
        elif any(isinstance(node, node_type) for node_type in
                 UNNECESSARY_TYPES):
            return self.process_unnecessary_types(node, state, call_source)
        else:
            node_name = node.__repr__().split()[0][2:]
            if self.process_grfn.get(node_name):
                return self.process_grfn[node_name](node, state, call_source)
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

    def process_function_definition(self, node, state, *_):
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
            self._process_decorators(node.decorator_list, function_state)

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

        # The following is a test to make sure that there is only one
        # function without arguments and that is the outermost function. All
        # of the models that we currently handle have this structure and
        # we'll have to think about how to handle cases that have more than
        # one non-argument function.
        if len(node.args.args) == 0:
            self.outer_count += 1
            assert self.outer_count == 1, "There is more than one function " \
                                          "without arguments in this system. " \
                                          "This is not currently handled."

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
            self._get_body_and_functions(body_grfn)

        # This function removes all variables related to I/O from the
        # variable list. This will be done until a specification for I/O is
        # defined in GrFN
        variables = self._remove_io_variables(list(
            local_last_definitions.keys()))

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

    def process_arg(self, node, state, call_source):
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
        state.variable_types[node.arg] = self._get_variable_type(
            node.annotation)

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

    def process_index(self, node, state, *_):
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
    def process_num(node, *_):
        """
            This function handles the ast.Num of the ast tree. This node only
            contains a numeric value in its body. For example: Num(n=0),
            Num(n=17.27), etc. So, we return the numeric value in a
            <function_assign_body_literal_spec> form.

        """
        # TODO: According to new specification, the following structure
        #  should be used: {"type": "literal, "value": {"dtype": <type>,
        #  "value": <value>}}. Confirm with Clay.
        data_type = ANNOTATE_MAP.get(type(node.n).__name__)
        if data_type:
            return [
                {"type": "literal", "dtype": data_type, "value": node.n}
            ]
        else:
            assert False, f"Unidentified data type of variable: {node.n}"

    def process_list_ast(self, node, state, *_):
        """
            This function handles ast.List which represents Python lists. The
            ast.List has an `elts` element which is a list of all the elements
            of the list. This is most notably encountered in annotated
            assignment of variables to [None] (Example: day: List[int] = [
            None]). This is handled by calling `gen_grfn` on every element of
            the list i.e. every element of `elts`.
        """
        # TODO: Will using element[0] work every time? Test on cases like
        #  [x+y, 4,5]. Here `x+y` should result in a [{spec_for_a},
        #  {spec_for_b}] format. So, element[0] will only take {spec_for_a}.
        #  Such cases not encountered yet but can occur.
        elements = [
            element[0]
            for element in [
                self.gen_grfn(list_element, state, "List")
                for list_element in node.elts
            ]
        ]
        return elements if len(elements) == 1 else [{"list": elements}]

    @staticmethod
    def process_str(node, *_):
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

    def process_for(self, node, state, *_):
        """
            This function handles the ast.For node of the AST.
        """
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

        # Currently For-Else on Python is not supported
        if self.gen_grfn(node.orelse, state, "for"):
            raise For2PyError("For/Else in for not supported.")

        index_variable = self.gen_grfn(node.target, state, "for")

        # Currently, only one variable is supported as a loop variable
        if len(index_variable) != 1 or "var" not in index_variable[0]:
            raise For2PyError("Only one index variable is supported.")

        index_name = index_variable[0]["var"]["variable"]

        # Currently, the loop iterator is strictly a `range` function.
        # TODO Will need to expand this over to arrays as iterators for loops
        #  and other possible loop iterators
        loop_iterator = self.gen_grfn(node.iter, state, "for")
        if (
                len(loop_iterator) != 1
                or "call" not in loop_iterator[0]
                or loop_iterator[0]["call"]["function"] != "range"
        ):
            raise For2PyError("Can only iterate over a range.")

        range_call = loop_iterator[0]["call"]

        # Perform some sanity checks for the loop range call
        if (
                len(range_call["inputs"][0]) != 1
                or len(range_call["inputs"][1]) != 1
                or (
                "type" in range_call["inputs"][0]
                and range_call["inputs"][0]["type"] == "literal"
                )
                or (
                "type" in range_call["inputs"][1]
                and range_call["inputs"][1]["type"] == "literal"
                )
        ):
            raise For2PyError("Can only iterate over a constant range.")

        if len(range_call["inputs"]) == 1:
            iteration_range = {
                "start": 0,
                "end": range_call["inputs"][0][0]-1,
            }
        elif len(range_call["inputs"]) == 2:
            iteration_range = {
                "start": range_call["inputs"][0][0],
                "end": range_call["inputs"][1][0],
            }
        elif len(range_call["inputs"]) == 3:
            iteration_range = {
                "start": range_call["inputs"][0][0],
                "end": range_call["inputs"][1][0],
                "step": range_call["inputs"][2][0],
            }
        else:
            raise For2PyError(f"Invalid number of arguments in range: "
                              f"{len(range_call['inputs'])}")

        loop_last_definition = {}
        loop_state = state.copy(
            last_definitions=loop_last_definition, next_definitions={},
            last_definition_default=-1
        )

        loop_state.last_definitions[index_name] = None
        loop = self.gen_grfn(node.body, loop_state, "for")

        # TODO Take a look at this function
        loop_body, loop_functions, identifier_specification = \
            self._get_body_and_functions(loop)

        # If loop_last_definition[x] == 0, this means that the variable was not
        # declared before the loop and is being declared/defined within
        # the loop. So we need to remove that from the variable_list
        variable_list = [x for x in loop_last_definition if (x != index_name and
                         state.last_definitions[x] != 0)]

        # TODO change this
        variables = [
            {"name": variable, "domain": state.variable_types[variable]}
            for variable in variable_list
        ]

        loop_plate = state.function_name + "__loop_plate__" + index_name

        # TODO This will be removed and be part of the container spec
        loop_function = {
            "name": loop_plate,
            "type": "loop_plate",
            "input": variables,
            "index_variable": index_name,
            "index_iteration_range": iteration_range,
            "body": loop_body,
        }

        id_spec_list = self.make_identifier_spec(
            loop_plate, index_name, {}, state
        )

        # TODO change this
        loop_call = {
            "name": loop_plate,
            "input": [
                {
                    "name": variable,
                    "index": loop_state.last_definitions[variable]
                }
                for variable in variable_list
            ],
            "output": {}
        }

        grfn = {
            "functions": loop_functions + [loop_function],
            "body": [loop_call],
            "identifiers": [],
        }

        for id_spec in id_spec_list:
            grfn["identifiers"].append(id_spec)

        return [grfn]

    def process_if(self, node, state, call_source):
        """
            This function handles the ast.IF node of the AST. It goes through
            the IF body and generates the `decision` and `condition` type of
            the `<function_assign_def>`.
        """
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("_TOP")
        scope_path.append("if")

        state.scope_path = scope_path
        grfn = {"functions": [], "body": [], "identifiers": []}

        # Get the GrFN schema of the test condition of the `IF` command
        condition_sources = self.gen_grfn(node.test, state, "if")

        # The index of the IF_x_x variable will start from 0
        if state.last_definition_default in (-1, 0):
            default_if_index = state.last_definition_default + 1
        else:
            assert False, f"Invalid value of last_definition_default:" \
                f"{state.last_definition_default}"

        if call_source != "if":
            condition_number = state.next_definitions.get("#cond",
                                                          default_if_index)
            state.next_definitions["#cond"] = condition_number + 1
        else:
            condition_number = self.elif_condition_number

        condition_name = f"IF_{condition_number}"
        state.variable_types[condition_name] = "bool"

        # TODO: Is this used anywhere?
        state.last_definitions[condition_name] = 0

        function_name = self._get_function_name(
            self.function_names,
            f"{state.function_name}__condition__"
            f"{condition_name}", {}
        )

        # The condition_name is of the form 'IF_1' and the index holds the
        # ordering of the condition_name. This means that index should increment
        # for every new 'IF' statement. Previously, it was set to set to 0.
        # But, 'function_name' holds the current index of 'condition_name'
        # So, extract the index from 'function_name'.
        condition_output = {"variable": condition_name, "index": int(
            function_name[-1])}

        # TODO
        # Creating the condition block
        fn = {
            "name": function_name,
            "type": "condition",
            "target": condition_name,
            "reference": node.lineno,
            "sources": [
                {"name": src["var"]["variable"], "type": "variable"}
                for src in condition_sources
                if "var" in src
            ],
        }

        # TODO
        id_spec_list = self.make_identifier_spec(
            function_name,
            condition_output,
            [src["var"] for src in condition_sources if "var" in src],
            state,
        )

        # TODO
        for id_spec in id_spec_list:
            grfn["identifiers"].append(id_spec)

        # TODO
        body = {
            "name": function_name,
            "output": condition_output,
            "input": [src["var"] for src in condition_sources if "var" in src],
        }

        grfn["functions"].append(fn)
        grfn["body"].append(body)

        lambda_string = self._generate_lambda_function(
            node.test,
            function_name,
            None,
            [src["var"]["variable"] for src in condition_sources if
             "var" in src],
            state,
        )
        state.lambda_strings.append(lambda_string)

        start_definitions = state.last_definitions.copy()
        if_definitions = start_definitions.copy()
        else_definitions = start_definitions.copy()
        if_state = state.copy(last_definitions=if_definitions)
        else_state = state.copy(last_definitions=else_definitions)
        if_grfn = self.gen_grfn(node.body, if_state, "if")
        # Note that `else_grfn` will be empty if the else block contains
        # another `if-else` block
        else_node_name = node.orelse.__repr__().split()[0][3:]
        if else_node_name != "ast.If":
            else_grfn = self.gen_grfn(node.orelse, else_state, "if")
        else:
            else_grfn = []
            self.elif_condition_number = condition_number

        for spec in if_grfn:
            grfn["functions"] += spec["functions"]
            grfn["body"] += spec["body"]

        for spec in else_grfn:
            grfn["functions"] += spec["functions"]
            grfn["body"] += spec["body"]

        updated_definitions = [
            var
            for var in set(start_definitions.keys())
            .union(if_definitions.keys())
            .union(else_definitions.keys())
            if var not in start_definitions
            or if_definitions[var] != start_definitions[var]
            or else_definitions[var] != start_definitions[var]
        ]

        # For every updated variable in the `if-else` block, get a list of
        # all defined indices of that variable
        defined_versions = {}
        for key in updated_definitions:
            defined_versions[key] = [
                version
                for version in [
                    start_definitions.get(key),
                    if_definitions.get(key),
                    else_definitions.get(key),
                ]
                if version is not None
            ]

        # For every updated identifier, we need one __decision__ block. So
        # iterate over all updated identifiers.
        for updated_definition in defined_versions:
            versions = defined_versions[updated_definition]
            inputs = (
                [
                    condition_output,
                    {"variable": updated_definition, "index": versions[-1]},
                    {"variable": updated_definition, "index": versions[-2]},
                ]
                if len(versions) > 1
                else [
                    condition_output,
                    {"variable": updated_definition, "index": versions[0]},
                ]
            )

            output = {
                "variable": updated_definition,
                "index": self._get_next_definition(
                    updated_definition,
                    state.last_definitions,
                    state.next_definitions,
                    state.last_definition_default,
                ),
            }

            function_name = self._get_function_name(
                self.function_names,
                f"{state.function_name}__decision__{updated_definition}", output
            )

            # TODO
            # Creating the __decision__ block
            fn = {
                "name": function_name,
                "type": "decision",
                "target": updated_definition,
                "reference": node.lineno,
                "sources": [
                    {
                        "name": f"{var['variable']}_{var['index']}",
                        "type": "variable",
                    }
                    for var in inputs
                ],
            }

            # TODO
            body = {"name": function_name, "output": output, "input": inputs}

            # TODO
            id_spec_list = self.make_identifier_spec(
                function_name, output, inputs, state
            )

            # TODO
            for id_spec in id_spec_list:
                grfn["identifiers"].append(id_spec)

            lambda_string = self._generate_lambda_function(
                node,
                function_name,
                updated_definition,
                [f"{src['variable']}_{src['index']}" for src in inputs],
                state,
            )
            state.lambda_strings.append(lambda_string)

            grfn["functions"].append(fn)
            grfn["body"].append(body)

        if else_node_name == "ast.If":
            # else_definitions = state.last_definitions.copy()
            else_state = state.copy(last_definitions=state.last_definitions)
            elseif_grfn = self.gen_grfn(node.orelse, else_state, "if")
            for spec in elseif_grfn:
                grfn["functions"] += spec["functions"]
                grfn["body"] += spec["body"]
                grfn["identifiers"] += spec["identifiers"]

        return [grfn]

    def process_unary_operation(self, node, state, *_):
        """
            This function processes unary operations in Python represented by
            ast.UnaryOp. This node has an `op` key which contains the
            operation (e.g. USub for -, UAdd for +, Not, Invert) and an
            `operand` key which contains the operand of the operation. This
            operand can in itself be any Python object (Number, Function
            call, Binary Operation, Unary Operation, etc. So, iteratively
            call the respective ast handler for the operand.
        """
        return self.gen_grfn(node.operand, state, "unaryop")

    def process_binary_operation(self, node, state, *_):
        """
            This function handles binary operations i.e. ast.BinOp
        """
        # If both the left and right operands are numbers (ast.Num), we can
        # simply perform the respective operation on these two numbers and
        # represent this computation in a GrFN spec.
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
            for op in BINOPS:
                if isinstance(node.op, op):
                    val = BINOPS[type(node.op)](node.left.n, node.right.n)
                    data_type = ANNOTATE_MAP.get(type(val).__name__)
                    if data_type:
                        return [
                            {
                                "type": "literal",
                                "dtype": data_type,
                                "value": val,
                            }
                        ]
                    else:
                        assert False, f"Unidentified data type of: {val}"
            assert False, ("Both operands are numbers but no operator "
                           "available to handle their computation. Either add "
                           "a handler if possible or remove this assert and "
                           "allow the code below to handle such cases.")

        # If the operands are anything other than numbers (ast.Str,
        # ast.BinOp, etc), call `gen_grfn` on each side so their respective
        # ast handlers will process them and return a [{grfn_spec}, ..] form
        # for each side. Add these two sides together to give a single [{
        # grfn_spec}, ...] form.
        operation_grfn = self.gen_grfn(node.left, state, "binop") \
            + self.gen_grfn(node.right, state, "binop")

        return operation_grfn

    def process_boolean_operation(self, node, state, *_):
        """
            This function will process the ast.BoolOp node that handles
            boolean operations i.e. AND, OR, etc.
        """
        # TODO: No example of this to test on. This looks like deprecated
        #  format. Will need to be rechecked.
        grfn_list = []
        operation = {ast.And: "and", ast.Or: "or"}

        for key in operation:
            if isinstance(node.op, key):
                grfn_list.append([{"boolean_operation": operation[key]}])

        for item in node.values:
            grfn_list.append(self.gen_grfn(item, state, "boolop"))

        return grfn_list

    @staticmethod
    def process_unnecessary_types(node, *_):
        """
            This function handles various ast tags which are unnecessary and
            need not be handled since we do not require to parse them
        """
        node_name = node.__repr__().split()[0][2:]
        assert False, f"Found {node_name}, which should be unnecessary"

    def process_expression(self, node, state, *_):
        """
            This function handles the ast.Expr node i.e. the expression node.
            This node appears on function calls such as when calling a
            function, calling print(), etc.
        """
        expressions = self.gen_grfn(node.value, state, "expr")
        grfn = {"functions": [], "body": [], "identifiers": []}
        for expr in expressions:
            if "call" in expr:
                call = expr["call"]
                body = {
                    "function": call["function"],
                    "output": {},
                    "input": [],
                }
                # If the call is to the write() function of a file handle,
                # bypass this expression as I/O is not handled currently
                # TODO: Will need to be handled once I/O is handled
                if re.match(r"file_\d+\.write", body["function"]):
                    return []
                for arg in call["inputs"]:
                    if len(arg) == 1:
                        # TODO: Only variables are represented in function
                        #  arguments. But a function can have strings as
                        #  arguments as well. Do we add that?
                        if "var" in arg[0]:
                            body["input"].append(arg[0]["var"])
                    else:
                        raise For2PyError(
                            "Only 1 input per argument supported right now."
                        )
                grfn["body"].append(body)
            else:
                raise For2PyError(f"Unsupported expr: {expr}.")
        return [grfn]

    def process_compare(self, node, state, *_):
        """
            This function handles ast.Compare i.e. the comparator tag which
            appears on logical comparison i.e. ==, <, >, <=, etc. This
            generally occurs within an `if` statement but can occur elsewhere
            as well.
        """
        return self.gen_grfn(node.left, state, "compare") \
            + self.gen_grfn(node.comparators, state, "compare")

    def process_subscript(self, node, state, *_):
        """
            This function handles the ast.Subscript i.e. subscript tag of the
            ast. This tag appears on variable names that are indexed i.e.
            x[0], y[5], var[float], etc. Subscript nodes will have a `slice`
            tag which gives a information inside the [] of the call.
        """
        # The value inside the [] should be a number for now.
        # TODO: Remove this and handle further for implementations of arrays,
        #  reference of dictionary item, etc
        if not isinstance(node.slice.value, ast.Num):
            raise For2PyError("can't handle arrays right now.")

        val = self.gen_grfn(node.value, state, "subscript")
        if val:
            if val[0]["var"]["variable"] in self.annotated_assigned:
                if isinstance(node.ctx, ast.Store):
                    val[0]["var"]["index"] = self._get_next_definition(
                        val[0]["var"]["variable"],
                        state.last_definitions,
                        state.next_definitions,
                        state.last_definition_default,
                    )
            elif val[0]["var"]["index"] == -1:
                if isinstance(node.ctx, ast.Store):
                    val[0]["var"]["index"] = self._get_next_definition(
                        val[0]["var"]["variable"],
                        state.last_definitions,
                        state.next_definitions,
                        state.last_definition_default,
                    )
                    self.annotated_assigned.append(val[0]["var"]["variable"])
            else:
                self.annotated_assigned.append(val[0]["var"]["variable"])
        else:
            assert False, "No variable name found for subscript node."

        return val

    def process_name(self, node, state, call_source):
        """
            This function handles the ast.Name node of the AST. This node
            represents any variable in the code.
        """
        # Currently, bypassing any `i_g_n_o_r_e___m_e__` variables which are
        # used for comment extraction.
        if not re.match(r'i_g_n_o_r_e___m_e__.*', node.id):
            last_definition = self._get_last_definition(
                node.id,
                state.last_definitions,
                state.last_definition_default
            )

            # Only increment the index of the variable if it is on the RHS of
            # the assignment/operation i.e. Store(). Also, we don't increment
            # it when the operation is an annotated assignment (of the form
            # max_rain: List[float] = [None])
            if (
                    isinstance(node.ctx, ast.Store)
                    and state.next_definitions.get(node.id)
                    and call_source != "annassign"
            ):
                last_definition = self._get_next_definition(
                    node.id,
                    state.last_definitions,
                    state.next_definitions,
                    state.last_definition_default,
                )

            return [{"var": {"variable": node.id, "index": last_definition}}]

    def process_annotated_assign(self, node, state, *_):
        """
            This function handles annotated assignment operations i.e.
            ast.AnnAssign. This tag appears when a variable has been assigned
            with an annotation e.g. x: int = 5, y: List[float] = None, etc.
        """
        # If the assignment value of a variable is of type List, retrieve the
        # targets. As of now, the RHS will be a list only during initial
        # variable definition i.e. day: List[int] = [None]. So, we only
        # update our data structures that hold the variable type mapping
        # and annotated variable mappings. Nothing will be added to the
        # GrFN.
        # TODO: Will this change once arrays/lists are implemented?
        if isinstance(node.value, ast.List):
            targets = self.gen_grfn(node.target, state, "annassign")
            for target in targets:
                state.variable_types[target["var"]["variable"]] = \
                    self._get_variable_type(node.annotation)
                if target["var"]["variable"] not in self.annotated_assigned:
                    self.annotated_assigned.append(target["var"]["variable"])
            return []

        sources = self.gen_grfn(node.value, state, "annassign")
        targets = self.gen_grfn(node.target, state, "annassign")

        grfn = {"functions": [], "body": [], "identifiers": []}

        for target in targets:
            state.variable_types[target["var"]["variable"]] = \
                self._get_variable_type(node.annotation)
            name = self._get_function_name(
                self.function_names,
                f"{state.function_name}__assign__"
                f"{target['var']['variable']}",
                {},
            )
            fn = self.make_fn_dict(name, target, sources, node)
            body = self.make_body_dict(name, target, sources, state)

            if len(sources) > 0:
                lambda_string = self._generate_lambda_function(
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
                grfn["identifiers"].append(id_spec)

            grfn["functions"].append(fn)
            grfn["body"].append(body[0])

        return [grfn]

    def process_assign(self, node, state, *_):
        """
            This function handles an assignment operation (ast.Assign).
        """
        # If the scope path is empty, this has to be the top of the program
        # scope, so start the scope with a `_TOP` string to denote that the
        # operation lies at the top of the scope.
        if len(state.scope_path) == 0:
            state.scope_path.append("_TOP")

        # Get the GrFN element of the RHS side of the assignment which are
        # the variables involved in the assignment operations.
        sources = self.gen_grfn(node.value, state, "assign")

        # This reduce function is useful when a single assignment operation
        # has multiple targets (E.g: a = b = 5). Currently, the translated
        # python code does not appear in this way and only a single target
        # will be present.
        targets = reduce(
            (lambda x, y: x.append(y)),
            [
                self.gen_grfn(target, state, "assign")
                for target in node.targets
            ],
        )

        grfn = {"functions": [], "body": [], "identifiers": []}

        # Again as above, only a single target appears in current version.
        # The for loop seems unnecessary but will be required when multiple
        # targets start appearing.
        for target in targets:
            # If the target is a list of variables, the grfn notation for the
            # target will be a list of variable names i.e. "[a, b, c]"
            # TODO: This does not seem right. Discuss with Clay and Paul
            #  about what a proper notation for this would be
            if target.get("list"):
                targets = ",".join(
                    [x["var"]["variable"] for x in target["list"]]
                )
                target = {"var": {"variable": targets, "index": 1}}

            name = self._get_function_name(
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

            lambda_string = self._generate_lambda_function(
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
                grfn["identifiers"].append(id_spec)

            grfn["functions"].append(fn)
            grfn["body"].append(body[0])

        return [grfn]

    def process_tuple(self, node, state, *_):
        """
            This function handles the ast.Tuple node of the AST. This handled
            in the same way `process_list_ast` is handled.
        """
        elements = [
            element[0]
            for element in [
                self.gen_grfn(list_element, state, "ctx")
                for list_element in node.elts
            ]
        ]

        return elements if len(elements) == 1 else [{"list": elements}]

    def process_call(self, node, state, *_):
        """
            This function handles the ast.Call node of the AST. This node
            denotes the call to a function. The body contains of the function
            name and its arguments.
        """
        # Check if the call is in the form of <module>.<function> (E.g.
        # math.exp, math.cos, etc). The `module` part here is captured by the
        # attribute tag.
        if isinstance(node.func, ast.Attribute):
            # Check if there is a <sys> call. Bypass it if exists.
            if isinstance(node.func.value, ast.Attribute) and \
                    node.func.value.value.id == "sys":
                return []
            function_node = node.func
            module = function_node.value.id
            function_name = function_node.attr
            function_name = module + "." + function_name
        else:
            function_name = node.func.id

        inputs = []
        for arg in node.args:
            argument = self.gen_grfn(arg, state, "call")
            inputs.append(argument)

        call = {"call": {"function": function_name, "inputs": inputs}}
        return [call]

    def process_module(self, node, state, *_):
        """
            This function handles the ast.Module node in the AST. The module
            node is the starting point of the AST and its body consists of
            the entire ast of the python code.
        """
        grfn_list = []
        for cur in node.body:
            grfn = self.gen_grfn(cur, state, "module")
            grfn_list += grfn
        return [self._merge_dictionary(grfn_list)]

    def process_attribute(self, node, state, call_source):
        """
            Handle Attributes: This is a fix on `feature_save` branch to
            bypass the SAVE statement feature where a SAVEd variable is
            referenced as <function_name>.<variable_name>. So the code below
            only returns the <variable_name> which is stored under
            `node.attr`. The `node.id` stores the <function_name> which is
            being ignored.
        """
        # When a computations float value is extracted using the Float32
        # class's _val method, an ast.Attribute will be present
        if node.attr == "_val":
            return self.gen_grfn(node.value, state, call_source)
        else:
            # TODO: This section of the code should be the same as
            #  `process_name`. Verify this.
            last_definition = self._get_last_definition(
                node.attr,
                state.last_definitions,
                state.last_definition_default
            )

            return [{"var": {"variable": node.attr, "index": last_definition}}]

    @staticmethod
    def process_ast(node, *_):
        sys.stderr.write(
            f"No handler for AST.{node.__class__.__name__} in gen_grfn, "
            f"fields: {node._fields}\n"
        )

    def process_load(self, node, state, call_source):
        raise For2PyError(f"Found ast.Load, which should not happen. "
                          f"From source: {call_source}")

    def process_store(self, node, state, call_source):
        raise For2PyError(f"Found ast.Store, which should not happen. "
                          f"From source: {call_source}")

    @staticmethod
    def process_nomatch(node, *_):
        sys.stderr.write(
            f"No handler for {node.__class__.__name__} in gen_grfn, "
            f"value: {str(node)}\n"
        )

    def make_identifier_dictionary(self, name, targets, scope_path, holder):
        """
            This function creates the dictionary for an identifier
        """
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

        # TODO handle multiple file namespaces that handle multiple fortran
        #  file namespacing

        namespace_path_list = get_path(self.fortran_file, "namespace")
        namespace_path = ".".join(namespace_path_list)

        # TODO Hack: Currently only the last element of the
        #  `namespace_path_list` is being returned as the `namespace_path` in
        #  order to make it consistent with the handwritten SIR-Demo GrFN
        #  JSON. Will need a more generic path for later instances.
        namespace_path = namespace_path_list[-1]

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

        source_reference = namespace_path

        iden_dict = {
            "base_name": base_name,
            "scope": scope_path,
            "namespace": namespace_path,
            "source_references": source_reference,
            "gensyms": self._generate_gensym(gensyms_tag),
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
                iden_dict = self.make_identifier_dictionary(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_identifier_dictionary(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "condition":
                iden_dict = self.make_identifier_dictionary(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_identifier_dictionary(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "loop_plate":
                iden_dict = self.make_identifier_dictionary(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_identifier_dictionary(
                            name,
                            item["variable"] + "_" + str(item["index"]),
                            scope_path,
                            "variable",
                        )
                        identifier_list.append(iden_dict)
            elif match.group("type") == "decision":
                iden_dict = self.make_identifier_dictionary(
                    match.group("type"), targets, scope_path, "body"
                )
                identifier_list.append(iden_dict)
                if len(sources) > 0:
                    for item in sources:
                        iden_dict = self.make_identifier_dictionary(
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

    def make_source_list_dict(self, source_dictionary):
        source_list = []
        for src in source_dictionary:
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
        bypass_match_target = RE_BYPASS_IO.match(target["var"]["variable"])

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
                for source_ins in self.make_call_body_dict(src):
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

    @staticmethod
    def _remove_io_variables(variable_list):
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

    def make_call_body_dict(self, source):
        """
        We are going to remove addition of functions such as "max", "exp",
        "sin",
        etc to the source list. The following two lines when commented helps
        us do
        that. If user-defined functions come up as sources, some other approach
        might be required.
        """
        # TODO Try with user defined functions and see if the below two lines
        #  need
        #  to be reworked
        # name = source["call"]["function"]
        # source_list.append({"name": name, "type": "function"})

        source_list = []
        for ip in source["call"]["inputs"]:
            if isinstance(ip, list):
                for item in ip:
                    if "var" in item:
                        variable = item["var"]["variable"]
                        source_list.append(
                            {"name": variable, "type": "variable"})
                    elif item.get("dtype") == "string":
                        # TODO Do repetitions in this like in the above check
                        #  need
                        #  to be removed?
                        source_list.append(
                            {"name": item["value"], "type": "variable"}
                        )
                    elif "call" in item:
                        source_list.extend(self.make_call_body_dict(item))

        return source_list

    @staticmethod
    def _process_decorators(node, state):
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
                    state.variable_types[variable] = ANNOTATE_MAP[variable_type]

    @staticmethod
    def _merge_dictionary(dicts: Iterable[Dict]) -> Dict:
        """
            This function merges the entire dictionary created by `gen_grfn`
            into
            another dictionary in a managed manner. The `dicts` argument is a
            list of form [{}, {}, {}] where each {} dictionary is the grfn
            specification of a function. It contains `functions` and
            `identifiers` as its keys. Additionally, if the python code has a
            starting point, that is also present in the last {} of `dicts`. The
            function merges the values from the `functions` key of each {} in
            `dicts` into a single key of the same name. Similarly, it does this
            for every unique key in the `dicts` dictionaries.
        """
        fields = set(chain.from_iterable(d.keys() for d in dicts))
        merged_dict = {field: [] for field in fields}

        # Create a cross-product between each unique key and each grfn
        # dictionary
        for field, d in product(fields, dicts):
            if field in d:
                if isinstance(d[field], list):
                    merged_dict[field] += d[field]
                else:
                    merged_dict[field].append(d[field])

        return merged_dict

    @staticmethod
    def _get_function_name(function_names, basename, target):
        """
            This function creates the function name in GrFN format for every
            assign/container function encounter in the program.
        """

        # First, check whether the basename is a 'decision' block. If it is, we
        # need to get it's index from the index of its corresponding
        # identifier's
        # 'assign' block. We do not use the index of the 'decision' block as
        # that
        # will not correspond with that of the 'assign' block.  For example: for
        # petpt__decision__albedo, its index will be the index of the latest
        # petpt__assign__albedo + 1
        if "__decision__" in basename:
            new_basename = basename.replace("__decision__", "__assign__")
        else:
            new_basename = basename
        function_id = function_names.get(new_basename, 0)
        if len(target) > 0:
            if target.get("var"):
                function_id = target["var"]["index"]
            elif target.get("variable"):
                function_id = target["index"]
        if function_id < 0:
            function_id = function_names.get(new_basename, 0)
        function_name = f"{basename}_{function_id}"
        function_names[basename] = function_id + 1
        return function_name

    @staticmethod
    def _get_last_definition(var, last_definitions, last_definition_default):
        """
            This function returns the last (current) definition (index) of a
            variable.
        """
        index = last_definition_default

        # Pre-processing and removing certain Assigns which only pertain to the
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

    @staticmethod
    def _get_next_definition(var, last_definitions, next_definitions,
                             last_definition_default):
        """
            This function returns the next definition i.e. index of a variable.
        """
        # The dictionary `next_definitions` holds the next index of all current
        # variables in scope. If the variable is not found (happens when it is
        # assigned for the first time in a scope), its index will be one greater
        # than the last definition default.
        index = next_definitions.get(var, last_definition_default + 1)
        # Update the next definition index of this variable by incrementing
        # it by
        # 1. This will be used the next time when this variable is referenced on
        # the LHS side of an assignment.
        next_definitions[var] = index + 1
        # Also update the `last_definitions` dictionary which holds the current
        # index of all variables in scope.
        last_definitions[var] = index
        return index

    @staticmethod
    def _get_variable_type(annotation_node):
        """
            This function returns the data type of a variable using the
            annotation information used to define that variable
        """
        # If the variable has been wrapped in a list like x: List[int],
        # `annotation_node` will be a Subscript node
        if isinstance(annotation_node, ast.Subscript):
            data_type = annotation_node.slice.value.id
        else:
            data_type = annotation_node.id
        if ANNOTATE_MAP.get(data_type):
            return ANNOTATE_MAP[data_type]
        else:
            sys.stderr.write(
                "Unsupported type (only float, int, list, real, bool and str "
                "supported as of now).\n"
            )

    @staticmethod
    def _get_body_and_functions(grfn):
        body = list(chain.from_iterable(stmt["body"] for stmt in grfn))
        fns = list(chain.from_iterable(stmt["functions"] for stmt in grfn))
        identifier_specification = list(chain.from_iterable(stmt["identifiers"]
                                                            for stmt in grfn))
        return body, fns, identifier_specification

    @staticmethod
    def _generate_gensym(tag):
        """
            The gensym is used to uniquely identify any identifier in the
            program. Python's uuid library is used to generate a unique 12 digit
            HEX string. The uuid4() function of 'uuid' focuses on randomness.
            Each and every bit of a UUID v4 is generated randomly and with no
            inherent logic. To every gensym, we add a tag signifying the data
            type it represents. 'v' is for variables and 'h' is for holders.
        """
        return uuid.uuid4().hex[:12] + "_" + tag

    @staticmethod
    def _generate_lambda_function(node, function_name: str, return_value: bool,
                                  inputs, state):
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
                annotation = state.variable_types[
                    key_match(ip, state.variable_types)[0]
                ]
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
        if return_value:
            lambda_strings.append(f"return {code}")
        else:
            lines = code.split("\n")
            indent = re.search("[^ ]", lines[-1]).start()
            lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
            lambda_strings.append("\n".join(lines))
        lambda_strings.append("\n\n")
        return "".join(lambda_strings)


def get_path(file_name: str, instance: str):
    """
        This function returns the path of a file starting from the root of
        the delphi repository. The returned path varies depending on whether
        it is for a namespace or a source variable, which is denoted by the
        `instance` argument variable. It is important to note that the path
        refers to that of the original system being analyzed i.e. the Fortran
        code and not the intermediate Python file which is used to generate
        the AST.
    """
    if instance == "source":
        source_match = re.match(r'[./]*(.*)', file_name)
        assert source_match, f"Original Fortran source file for {file_name} " \
            f"not found."
        return source_match.group(1)
    elif instance == "namespace":
        source_match = re.match(r'[./]*(.*)\.', file_name)
        assert source_match, f"Namespace path for {file_name} not found."
        return source_match.group(1).split("/")
    else:
        assert False, f"Error when trying to get the path of file {file_name}."


def dump_ast(node, annotate_fields=True, include_attributes=False, indent="  "):
    """
        Return a formatted dump of the tree in *node*. This is mainly useful for
        debugging purposes. The returned string will show the names and the
        values for fields. This makes the code impossible to evaluate,
        so if evaluation is wanted *annotate_fields* must be set to False.
        Attributes such as line numbers and column offsets are not dumped by
        default. If this is wanted, *include_attributes* can be set to True.
    """

    def _format(ast_node, level=0):
        if isinstance(ast_node, ast.AST):
            fields = [(a, _format(b, level)) for a, b in
                      ast.iter_fields(ast_node)]
            if include_attributes and ast_node._attributes:
                fields.extend(
                    [
                        (a, _format(getattr(ast_node, a), level))
                        for a in ast_node._attributes
                    ]
                )
            return "".join(
                [
                    ast_node.__class__.__name__,
                    "(",
                    ", ".join(
                        ("%s=%s" % field for field in fields)
                        if annotate_fields
                        else (b for a, b in fields)
                    ),
                    ")",
                ]
            )
        elif isinstance(ast_node, list):
            lines = ["["]
            lines.extend(
                (
                    indent * (level + 2) + _format(x, level + 2) + ","
                    for x in ast_node
                )
            )
            if len(lines) > 1:
                lines.append(indent * (level + 1) + "]")
            else:
                lines[-1] += "]"
            return "\n".join(lines)
        return repr(ast_node)

    if not isinstance(node, ast.AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)
    return _format(node)


def create_grfn_dict(
    lambda_file: str,
    asts: List,
    file_name: str,
    mode_mapper_dict: dict,
    original_file: str,
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
    generator.fortran_file = original_file
    grfn = generator.gen_grfn(asts, state, "")[0]

    # If the GrFN has a `start` node, it will refer to the name of the
    # PROGRAM module which will be the entry point of the GrFN.
    if grfn.get("start"):
        grfn["start"] = grfn["start"][0]
    else:
        # TODO: If the PROGRAM module is not detected, the entry point will be
        #  the last function in the `function_defs` list of functions
        grfn["start"] = generator.function_definitions[-1]

    # Get the file path of the original Fortran code being analyzed
    source_file = get_path(original_file, "source")
    grfn["source"] = [source_file]

    # dateCreated stores the date and time on which the lambda and GrFN files
    # were created. It is stored in the YYYMMDD format
    grfn["date_created"] = f"{datetime.utcnow().isoformat('T')}Z"

    with open(lambda_file, "w") as lambda_fh:
        lambda_fh.write("".join(lambda_string_list))

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


def get_system_name(pyfile_list: List[str]):
    """
    This function returns the name of the system under analysis. Generally,
    the system is the one which is not prefixed by `m_` (which represents
    modules).
    """
    system_name = None
    path = None
    for file in pyfile_list:
        if not file.startswith("m_"):
            system_name_match = re.match(r'.*/(.*)\.py', file)
            assert system_name_match, f"System name for file {file} not found."
            system_name = system_name_match.group(1)

            path_match = re.match(r'(.*)/.*', file)
            assert path_match, "Target path not found"
            path = path_match.group(1)

    if not (system_name or path):
        assert False, f"Error when trying to find the system name of the " \
            f"analyzed program."

    return system_name, path


def generate_system_def(python_list: List[str], component_list: List[str]):
    """
        This function generates the system definition for the system under
        analysis and writes this to the main system file.
    """
    (system_name, path) = get_system_name(python_list)
    system_filename = f"{path}/system.json"
    grfn_components = []
    for component in component_list:
        grfn_components.append({
            "file_path": component,
            "imports": []
        })
    with open(system_filename, "w") as system_file:
        system_def = {
            "date_created": f"{datetime.utcnow().isoformat('T')}Z",
            "name": system_name,
            "components": grfn_components
        }
        system_file.write(json.dumps(system_def, indent=2))


def process_files(python_list: List[str], grfn_tail: str, lambda_tail:
                  str, original_file: str, print_ast_flag=False):
    """
        This function takes in the list of python files to convert into GrFN 
        and generates each file's AST along with starting the GrFN generation
        process. 
    """
    module_mapper = {}
    grfn_filepath_list = []
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
        lambda_file = python_list[index][:-3] + "_" + lambda_tail
        grfn_file = python_list[index][:-3] + "_" + grfn_tail
        grfn_dict = create_grfn_dict(
            lambda_file, [ast_string], python_list[index], module_mapper,
            original_file
        )
        grfn_filepath_list.append(grfn_file)
        # Write each GrFN JSON into a file
        with open(grfn_file, "w") as file_handle:
            file_handle.write(json.dumps(grfn_dict, indent=2))

    # Finally, write the <systems.json> file which gives a mapping of all the
    # GrFN files related to the system.
    generate_system_def(python_list, grfn_filepath_list)


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
    parser.add_argument(
        "-g",
        "--original_file",
        nargs=1,
        required=True,
        help="Filename of the original Fortran file",
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
    fortran_file = arguments.original_file[0]
    print_ast = arguments.print_ast

    process_files(python_file_list, grfn_suffix, lambda_suffix, fortran_file,
                  print_ast)
