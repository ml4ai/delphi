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
from delphi.translators.for2py.get_comments import get_comments
from delphi.translators.for2py import For2PyError
from typing import List, Dict, Iterable, Optional
from itertools import chain, product
import operator
import uuid

###########################################################################
#                                                                         #
#                            GLOBAL VARIABLES                             #
#                                                                         #
###########################################################################

# The BINOPS dictionary holds operators for all the arithmetic and
# comparative functions
# TODO Take this inside class
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
# TODO Take this inside class
ANNOTATE_MAP = {
    "real": "real",
    "float": "real",
    "Real": "real",
    "integer": "int",
    "int": "integer",
    "string": "str",
    "str": "string",
    "array": "[]",
    "list": "array",
    "bool": "bool",
    "file_handle": "fh",
}

# The UNNECESSARY_TYPES tuple holds the ast types to ignore
# TODO Take this inside class
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
# TODO Take this inside class
BYPASS_IO = r"^format_\d+$|^format_\d+_obj$|^file_\d+$|^write_list_\d+$|" \
            r"^write_line$|^format_\d+_obj" \
            r".*|^Format$|^list_output_formats$|^write_list_stream$|^file_\d" \
            r"+\.write$"
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
        arrays: Optional[Dict] = {},
        array_types: Optional[Dict] = {},
        array_assign_name: Optional=None
    ):
        self.lambda_strings = lambda_strings
        self.last_definitions = last_definitions
        self.next_definitions = next_definitions
        self.last_definition_default = last_definition_default
        self.function_name = function_name
        self.variable_types = variable_types
        self.start = start
        self.scope_path = scope_path
        self.arrays = arrays
        self.array_types = array_types
        self.array_assign_name = array_assign_name

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
        arrays: Optional[Dict] = None,
        array_types: Optional[Dict] = None,
        array_assign_name: Optional = None,
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
            self.arrays if arrays is None else arrays,
            self.array_types if array_types is None else array_types,
            self.array_assign_name if array_assign_name is None else array_assign_name,
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
        self.loop_input = []
        self.update_functions = {}
        self.mode_mapper = {}
        self.name_mapper = {}
        self.function_argument_map = {}
        self.arrays = {}
        self.array_types = {}
        self.array_assign_name = None
        self.outer_count = 0
        self.types = (list, ast.Module, ast.FunctionDef)
        self.elif_condition_number = None
        self.current_scope = None
        self.loop_index = -1
        self.parent_loop_state = None

        self.gensym_tag_map = {
            "container": 'c',
            "variable": 'v',
            "function": 'f',
            "holder": 'h'  # TODO Change/Remove this
        }
        self.type_def_map = {
            "real": "float",
            "integer": "integer",
            "string": "string",
            "bool": "boolean",
        }
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
            "ast.NameConstant": self._process_nameconstant,
            "ast.Return": self.process_return_value,
            "ast.While": self.process_while,
        }

    def gen_grfn(self, node, state, call_source):
        """
            This function generates the GrFN structure by parsing through the
            python AST
        """
        # DEBUG
        print ("node: ", node)
        # Look for code that is not inside any function.
        if state.function_name is None and not any(
            isinstance(node, t) for t in self.types
        ):
            # If the node is of instance ast.Call, it is the starting point
            # of the system.
            if isinstance(node, ast.Call):
                start_function_name = self.generate_container_id_name(
                    self.fortran_file, ["@global"], node.func.id)
                return [{"start": start_function_name}]
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
         they appear as a list. Process each node in the list and chain them
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
        return_value = []
        return_list = []

        local_last_definitions = state.last_definitions.copy()
        local_next_definitions = state.next_definitions.copy()
        local_variable_types = state.variable_types.copy()
        scope_path = state.scope_path.copy()

        # If the scope_path is empty, add @global to the list to denote that
        # this is the outermost scope
        if len(scope_path) == 0:
            scope_path.append("@global")

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
                last_definition_default=0,
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
        # Keep a map of the arguments for each function. This will be used in
        # `process_for` to identify arguments which are function arguments
        # from those that are not
        self.function_argument_map[node.name] = {
            "name": "",
            "updated_list": "",
            "updated_indices": [],
            "argument_list": ""
        }
        self.function_argument_map[node.name]["argument_list"] = argument_list
        # Update the current scope so that for every identifier inside the
        # body, the scope information is updated
        self.current_scope = node.name
        # Create the variable definition for the arguments
        argument_variable_grfn = []
        for argument in argument_list:
            argument_variable_grfn.append(
                self.generate_variable_definition(argument, function_state)
            )

        # Generate the `variable_identifier_name` for each container
        # argument.
        # TODO Currently only variables are handled as container arguments.
        #  Create test cases of other containers as container arguments and
        #  extend this functionality.
        argument_list = [f"@variable::{x}::{function_state.last_definitions[x]}"
                         for x in argument_list]

        # Enter the body of the function and recursively generate the GrFN of
        # the function body
        body_grfn = self.gen_grfn(node.body, function_state, "functiondef")

        # Get the `return_value` from the body. We want to append it separately.
        # TODO There can be multiple return values. `return_value` should be
        #  a list and you should append to it.
        for body in body_grfn:
            for function in body["functions"]:
                if function.get("type") == "return":
                    return_value = function["value"]

        # TODO The return value cannot always be a `variable`. It can be
        #  literals as well. Add that functionality here.
        if return_value:
            for value in return_value:
                return_list.append(f"@variable::{value['var']['variable']}::"
                                   f"{value['var']['index']}")
        else:
            return_list = None

        # Get the function_reference_spec, function_assign_spec and
        # identifier_spec for the function
        function_variable_grfn, function_assign_grfn, body_container_grfn = \
            self._get_variables_and_functions(body_grfn)
        # Combine the variable grfn of the arguments with that of the
        # container body
        container_variables = argument_variable_grfn + function_variable_grfn
        # Find the list of updated identifiers
        if argument_list:
            updated_identifiers = self._find_updated(argument_variable_grfn,
                                                     function_variable_grfn)
        else:
            updated_identifiers = []
        self.function_argument_map[node.name]["updated_list"] = \
            updated_identifiers

        # Get a list of all argument names
        argument_name_list = []
        for item in argument_list:
            argument_name_list.append(item.split("::")[1])

        # Now, find the indices of updated arguments
        for arg in updated_identifiers:
            updated_argument = arg.split("::")[1]
            argument_index = argument_name_list.index(updated_argument)
            self.function_argument_map[node.name]["updated_indices"].append(
                argument_index)

        # Create a gensym for the function container
        container_gensym = self.generate_gensym("container")

        container_id_name = self.generate_container_id_name(
            self.fortran_file, scope_path, node.name)
        self.function_argument_map[node.name]["name"] = container_id_name
        # Add the function name to the list that stores all the functions
        # defined in the program
        self.function_definitions.append(container_id_name)

        function_container_grfn = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "repeat": False,
            "arguments": argument_list,
            "updated": updated_identifiers,
            "return_value": return_list,
            "body": function_assign_grfn,
        }

        function_container_grfn = [function_container_grfn] + \
            body_container_grfn

        # function_assign_grfn.append(function_container_grfn)
        pgm = {"containers": function_container_grfn,
               "variables": container_variables,
               }

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
        # A the variable i.e. x[Int], y[Float], etc.
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
        data_type = ANNOTATE_MAP.get(type(node.n).__name__)
        if data_type:
            # TODO Change this format. Since the spec has changed,
            #  this format is no longer required. Go for a simpler format.
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
        # Update the scope
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("@global")
        scope_path.append("loop")

        # Check: Currently For-Else on Python is not supported
        if self.gen_grfn(node.orelse, state, "for"):
            raise For2PyError("For/Else in for not supported.")

        # Initialize intermediate variables
        container_argument = []
        container_repeat = True
        container_return_value = ""
        container_updated = []
        function_output = ""
        function_updated = []
        function_input = []
        loop_condition_inputs = []
        loop_variables_grfn = []
        loop_functions_grfn = []

        # Increment the loop index universally across the program
        if self.loop_index > -1:
            self.loop_index += 1
        else:
            self.loop_index = 0

        # Get the main function name (e.g. foo.loop$0.loop$1 then `foo`)
        main_function_name = self.current_scope.split('.')[0]
        # First, get the `container_id_name` of the loop container
        container_id_name = self.generate_container_id_name(
            self.fortran_file, self.current_scope, f"loop${self.loop_index}")

        # Update the scope of the loop container so that everything inside
        # the body of the loop will have the below scope
        self.current_scope = f"{self.current_scope}.loop${self.loop_index}"

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

        index_variable = self.gen_grfn(node.target, state, "for")
        # Check: Currently, only one variable is supported as a loop variable
        if len(index_variable) != 1 or "var" not in index_variable[0]:
            raise For2PyError("Only one index variable is supported.")
        index_name = index_variable[0]["var"]["variable"]

        # Define a new empty state that will be used for mapping the state of
        # the operations within the for-loop container
        loop_last_definition = {}
        loop_state = state.copy(
            last_definitions=loop_last_definition, next_definitions={},
            last_definition_default=-1
        )

        # We want the loop_state to have state information about variables
        # defined one scope above its current parent scope. The below code
        # allows us to do that
        if self.parent_loop_state:
            for var in self.parent_loop_state.last_definitions:
                if var not in state.last_definitions:
                    # state.last_definitions[var] = \
                        # self.parent_loop_state.last_definitions[var]
                    state.last_definitions[var] = -1

        loop_iterator = self.gen_grfn(node.iter, state, "for")
        # Check: Only the `range` function is supported as a loop iterator at
        # this moment
        if (
            len(loop_iterator) != 1
            or "call" not in loop_iterator[0]
            or loop_iterator[0]["call"]["function"] != "range"
        ):
            raise For2PyError("Can only iterate over a range.")

        range_call = loop_iterator[0]["call"]
        loop_condition_inputs.append(f"@variable::{index_name}::0")
        for ip in range_call["inputs"]:
            for var in ip:
                if "var" in var:
                    function_input.append(f"@variable::"
                                          f"{var['var']['variable']}::"
                                          f"{var['var']['index']}")
                    container_argument.append(f"@variable::"
                                              f"{var['var']['variable']}::-1")
                    loop_condition_inputs.append(
                        f"@variable::"
                        f"{var['var']['variable']}::-1")

        # Save the current state of the system so that it can used by a
        # nested loop to get information about the variables declared in its
        # outermost scopes.
        self.parent_loop_state = state

        # Define some condition and break variables in the loop state
        loop_state.last_definitions[index_name] = 0
        loop_state.last_definitions["IF_0"] = 0
        loop_state.last_definitions["BK"] = 0
        loop_state.variable_types["IF_0"] = "bool"
        loop_state.variable_types["BK"] = "bool"

        # Now, create the `variable` spec, `function name` and `container
        # wiring` for the loop index, check condition and break decisions.
        index_variable_grfn = self.generate_variable_definition(index_name,
                                                                loop_state)
        index_function_name = self.generate_function_name(
            "__assign__",
            index_variable_grfn["name"],
            None
        )
        index_function = {
            "function": index_function_name,
            "input": [],
            "output": f"@variable::{index_name}::0",
            "updated": []
        }

        loop_check_variable = self.generate_variable_definition("IF_0",
                                                                loop_state)
        loop_check_function_name = self.generate_function_name(
            "__condition__",
            loop_check_variable["name"],
            None
        )
        loop_condition_function = {
            "function": loop_check_function_name,
            "input": loop_condition_inputs,
            "output": f"@variable::IF_0::0",
            "updated": []
        }

        loop_break_variable = self.generate_variable_definition("BK",
                                                                loop_state)
        loop_break_function_name = self.generate_function_name(
            "__decision__",
            loop_break_variable["name"],
            None
        )
        loop_break_function = {
            "function": loop_break_function_name,
            "input": [f"@variable::IF_0::0"],
            "output": f"@variable::BK::0",
            "updated": []
        }

        # Parse through the body of the loop container
        loop = self.gen_grfn(node.body, loop_state, "for")
        # Separate the body grfn into `variables` and `functions` sub parts
        body_variables_grfn, body_functions_grfn, body_container_grfn = \
            self._get_variables_and_functions(loop)

        # Get a list of all variables that were used as inputs within the
        # loop body (nested as well).
        # print(body_functions_grfn)
        # print(body_variables_grfn)
        loop_body_inputs = []
        for function in body_functions_grfn:
            if function['function']['type'] == 'lambda':
                for ip in function['input']:
                    input_var = ip.split('::')[1]
                    loop_body_inputs.append(input_var)
            elif function['function']['type'] == 'container':
                # The same code as above but separating it out just in case
                # some extra checks are added in the future
                for ip in function['input']:
                    input_var = ip.split('::')[1]
                    loop_body_inputs.append(input_var)

        # Remove any duplicates since variables can be used multiple times in
        # various assignments within the body
        loop_body_inputs = list(set(loop_body_inputs))
        # Remove the index name since it is not an input to the container
        # print(index_name)
        # print(loop_body_inputs)
        loop_body_inputs.remove(index_name)

        # Now, we remove the variables which were defined inside the loop
        # body itself and not taken as an input from outside the loop body
        filtered_loop_body_inputs = []
        for input_var in loop_body_inputs:
            # We filter out those variables which have -1 index in `state` (
            # which means it did not have a defined value above the loop
            # body) and is not a function argument (since they have an index
            # of -1 as well but have a defined value)
            if not (state.last_definitions[input_var] == -1 and input_var not in
                    self.function_argument_map[main_function_name][
                        "argument_list"]
                    ):
                filtered_loop_body_inputs.append(input_var)

        for item in filtered_loop_body_inputs:
            function_input.append(f"@variable::{item}::"
                                  f"{state.last_definitions[item]}")
            container_argument.append(f"@variable::{item}::-1")

        # TODO: Think about removing (or retaining) variables which even
        #  though defined outside the loop, are defined again inside the loop
        #  and then used by an operation after it.
        #  E.g. x = 5
        #       for ___ :
        #           x = 2
        #           for ___:
        #               y = x + 2
        #  Here, loop$1 will have `x` as an input but will loop$0 have `x` as
        #  an input as well?
        #  Currently, such variables are included in the `input`/`argument`
        #  field.

        # Now, we list out all variables that have been updated/defined
        # inside the body of the loop
        loop_body_outputs = []
        for function in body_functions_grfn:
            if function['function']['type'] == 'lambda':
                output_var = function["output"].split('::')[1]
                loop_body_outputs.append(output_var)
            elif function['function']['type'] == 'container':
                for ip in function['updated']:
                    output_var = ip.split('::')[1]
                    loop_body_outputs.append(output_var)

        for item in loop_body_outputs:
            # TODO the indexing variables in of function block and container
            #  block will be different. Figure about the differences and
            #  implement them.
            function_updated.append(f"@variable::{item}::"
                                    f"{state.last_definitions[item]+1}")
            container_updated.append(f"@variable::{item}::"
                                     f"{loop_state.last_definitions[item]}")

        # TODO: For the `loop_body_outputs`, all variables that were
        #  defined/updated inside the loop body are included. Sometimes,
        #  some variables are defined inside the loop body, used within that
        #  body and then not used or re-assigned to another value outside the
        #  loop body. Do we include such variables in the updated list?
        #  Another heuristic to think about is whether to keep only those
        #  variables in the `updated` list which are in the `input` list.

        loop_variables_grfn.append(index_variable_grfn)
        loop_variables_grfn.append(loop_check_variable)
        loop_variables_grfn.append(loop_break_variable)

        loop_functions_grfn.append(index_function)
        loop_functions_grfn.append(loop_condition_function)
        loop_functions_grfn.append(loop_break_function)

        loop_variables_grfn += body_variables_grfn
        loop_functions_grfn += body_functions_grfn

        # Finally, add the index increment variable and function grfn to the
        # body grfn
        loop_state.last_definitions[index_name] = 1
        index_increment_grfn = self.generate_variable_definition(index_name,
                                                                 loop_state)
        index_increment_function_name = self.generate_function_name(
            "__assign_",
            index_increment_grfn["name"],
            None
        )
        index_increment_function = {
            "function": index_increment_function_name,
            "input": [f"@variable::{index_name}::0"],
            "output": f"@variable::{index_name}::1",
            "updated": []
        }
        loop_variables_grfn.append(index_increment_grfn)
        loop_functions_grfn.append(index_increment_function)

        container_gensym = self.generate_gensym("container")

        loop_container = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "repeat": container_repeat,
            "arguments": container_argument,
            "updated": container_updated,
            "return_value": container_return_value,
            "body": loop_functions_grfn,
        }
        loop_function = {
            "function": {
                "name": container_id_name,
                "type": "container"
            },
            "input": function_input,
            "output": function_output,
            "updated": function_updated
        }
        loop_container = [loop_container] + body_container_grfn
        loop_variables = body_variables_grfn + loop_variables_grfn
        grfn = {
            "containers": loop_container,
            "variables": loop_variables,
            "functions": [loop_function]
        }

        # Change the current scope back to its previous form.
        self.current_scope = '.'.join(self.current_scope.split('.')[:-1])
        return [grfn]

    def process_while(self, node, state, *_):
        """
            This function handles the while loop. The functionality will be
            very similar to that of the for loop described in `process_for`
        """
        # Update the scope
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("@global")
        scope_path.append("loop")

        # Initialize intermediate variables
        container_argument = []
        container_repeat = True
        container_return_value = ""
        container_updated = []
        function_output = ""
        function_updated = []
        function_input = []
        loop_condition_inputs = []
        loop_variables_grfn = []
        loop_functions_grfn = []

        # Increment the loop index universally across the program
        if self.loop_index > -1:
            self.loop_index += 1
        else:
            self.loop_index = 0

        # Get the main function name (e.g. foo.loop$0.loop$1 then `foo`)
        main_function_name = self.current_scope.split('.')[0]
        # First, get the `container_id_name` of the loop container
        container_id_name = self.generate_container_id_name(
            self.fortran_file, self.current_scope,
            f"loop${self.loop_index}")

        # Update the scope of the loop container so that everything inside
        # the body of the loop will have the below scope
        self.current_scope = f"{self.current_scope}.loop${self.loop_index}"

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

        loop_test = self.gen_grfn(node.test, state, "while")

        # Define a new empty state that will be used for mapping the state of
        # the operations within the for-loop container
        loop_last_definition = {}
        loop_state = state.copy(
            last_definitions=loop_last_definition, next_definitions={},
            last_definition_default=-1
        )

        # We want the loop_state to have state information about variables
        # defined one scope above its current parent scope. The below code
        # allows us to do that
        if self.parent_loop_state:
            for var in self.parent_loop_state.last_definitions:
                if var not in state.last_definitions:
                    # state.last_definitions[var] = \
                        # self.parent_loop_state.last_definitions[var]
                    state.last_definitions[var] = -1

        # Now populate the IF and BK functions for the loop by identifying
        # the loop conditionals
        # TODO Add a test to check for loop validity in this area. Need to
        #  test with more types of while loops to finalize on a test condition

        for item in loop_test:
            if not isinstance(item, list):
                item = [item]
            for var in item:
                if 'var' in var:
                    function_input.append(f"@variable::"
                                          f"{var['var']['variable']}::"
                                          f"{var['var']['index']}")
                    container_argument.append(f"@variable::"
                                              f"{var['var']['variable']}::-1")
                    loop_condition_inputs.append(
                        f"@variable::"
                        f"{var['var']['variable']}::-1")

        # Save the current state of the system so that it can used by a
        # nested loop to get information about the variables declared in its
        # outermost scopes.
        self.parent_loop_state = state

        # Define some condition and break variables in the loop state
        loop_state.last_definitions["IF_0"] = 0
        loop_state.last_definitions["BK"] = 0
        loop_state.variable_types["IF_0"] = "bool"
        loop_state.variable_types["BK"] = "bool"

        # Now, create the `variable` spec, `function name` and `container
        # wiring` for the check condition and break decisions.

        loop_check_variable = self.generate_variable_definition("IF_0",
                                                                loop_state)
        loop_check_function_name = self.generate_function_name(
            "__condition__",
            loop_check_variable["name"],
            None
        )
        loop_condition_function = {
            "function": loop_check_function_name,
            "input": loop_condition_inputs,
            "output": f"@variable::IF_0::0",
            "updated": []
        }

        loop_break_variable = self.generate_variable_definition("BK",
                                                                loop_state)
        loop_break_function_name = self.generate_function_name(
            "__decision__",
            loop_break_variable["name"],
            None
        )
        loop_break_function = {
            "function": loop_break_function_name,
            "input": [f"@variable::IF_0::0"],
            "output": f"@variable::BK::0",
            "updated": []
        }
        # Parse through the body of the loop container
        loop = self.gen_grfn(node.body, loop_state, "for")
        # Separate the body grfn into `variables` and `functions` sub parts
        body_variables_grfn, body_functions_grfn, body_container_grfn = \
            self._get_variables_and_functions(loop)

        # Get a list of all variables that were used as inputs within the
        # loop body (nested as well).
        loop_body_inputs = []
        for function in body_functions_grfn:
            if function['function']['type'] == 'lambda':
                for ip in function['input']:
                    input_var = ip.split('::')[1]
                    loop_body_inputs.append(input_var)
            elif function['function']['type'] == 'container':
                # The same code as above but separating it out just in case
                # some extra checks are added in the future
                for ip in function['input']:
                    input_var = ip.split('::')[1]
                    loop_body_inputs.append(input_var)

        # Remove any duplicates since variables can be used multiple times in
        # various assignments within the body
        loop_body_inputs = list(set(loop_body_inputs))

        # Now, we remove the variables which were defined inside the loop
        # body itself and not taken as an input from outside the loop body
        filtered_loop_body_inputs = []
        for input_var in loop_body_inputs:
            # We filter out those variables which have -1 index in `state` (
            # which means it did not have a defined value above the loop
            # body) and is not a function argument (since they have an index
            # of -1 as well but have a defined value)
            if not (state.last_definitions[input_var] == -1 and input_var not in
                    self.function_argument_map[main_function_name][
                        "argument_list"]
                    ):
                filtered_loop_body_inputs.append(input_var)

        for item in filtered_loop_body_inputs:
            function_input.append(f"@variable::{item}::"
                                  f"{state.last_definitions[item]}")
            container_argument.append(f"@variable::{item}::-1")

        # TODO: Think about removing (or retaining) variables which even
        #  though defined outside the loop, are defined again inside the loop
        #  and then used by an operation after it.
        #  E.g. x = 5
        #       for ___ :
        #           x = 2
        #           for ___:
        #               y = x + 2
        #  Here, loop$1 will have `x` as an input but will loop$0 have `x` as
        #  an input as well?
        #  Currently, such variables are included in the `input`/`argument`
        #  field.

        # Now, we list out all variables that have been updated/defined
        # inside the body of the loop
        loop_body_outputs = []
        for function in body_functions_grfn:
            if function['function']['type'] == 'lambda':
                output_var = function["output"].split('::')[1]
                loop_body_outputs.append(output_var)
            elif function['function']['type'] == 'container':
                for ip in function['updated']:
                    output_var = ip.split('::')[1]
                    loop_body_outputs.append(output_var)

        for item in loop_body_outputs:
            # TODO the indexing variables in of function block and container
            #  block will be different. Figure about the differences and
            #  implement them.
            function_updated.append(f"@variable::{item}::"
                                    f"{loop_state.last_definitions[item]}")
            container_updated.append(f"@variable::{item}::"
                                     f"{loop_state.last_definitions[item]}")

        # TODO: For the `loop_body_outputs`, all variables that were
        #  defined/updated inside the loop body are included. Sometimes,
        #  some variables are defined inside the loop body, used within that
        #  body and then not used or re-assigned to another value outside the
        #  loop body. Do we include such variables in the updated list?
        #  Another heuristic to think about is whether to keep only those
        #  variables in the `updated` list which are in the `input` list.

        loop_variables_grfn.append(loop_check_variable)
        loop_variables_grfn.append(loop_break_variable)

        loop_functions_grfn.append(loop_condition_function)
        loop_functions_grfn.append(loop_break_function)

        loop_variables_grfn += body_variables_grfn
        loop_functions_grfn += body_functions_grfn

        container_gensym = self.generate_gensym("container")

        loop_container = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "repeat": container_repeat,
            "arguments": container_argument,
            "updated": container_updated,
            "return_value": container_return_value,
            "body": loop_functions_grfn,
        }
        loop_function = {
            "function": {
                "name": container_id_name,
                "type": "container"
            },
            "input": function_input,
            "output": function_output,
            "updated": function_updated
        }
        loop_container = [loop_container] + body_container_grfn
        loop_variables = body_variables_grfn + loop_variables_grfn
        grfn = {
            "containers": loop_container,
            "variables": loop_variables,
            "functions": [loop_function]
        }
        self.current_scope = '.'.join(self.current_scope.split('.')[:-1])

        return [grfn]

    def process_if(self, node, state, call_source):
        """
            This function handles the ast.IF node of the AST. It goes through
            the IF body and generates the `decision` and `condition` type of
            the `<function_assign_def>`.
        """
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("@global")
        state.scope_path = scope_path

        grfn = {"functions": [], "variables": [], "containers": []}
        # Get the GrFN schema of the test condition of the `IF` command
        condition_sources = self.gen_grfn(node.test, state, "if")
        # The index of the IF_x_x variable will start from 0
        if state.last_definition_default in (-1, 0):
            # default_if_index = state.last_definition_default + 1
            default_if_index = 0
        else:
            assert False, f"Invalid value of last_definition_default:" \
                f"{state.last_definition_default}"

        if call_source != "if":
            condition_number = state.next_definitions.get("#cond",
                                                          default_if_index)
            state.next_definitions["#cond"] = condition_number + 1
            condition_name = f"IF_{condition_number}"
            condition_index = self._get_last_definition(condition_name,
                                                        state.last_definitions,
                                                        0)
        else:
            condition_number = self.elif_condition_number
            condition_name = f"IF_{condition_number}"
            condition_index = self._get_next_definition(
                condition_name,
                state.last_definitions,
                state.next_definitions,
                0)

        state.variable_types[condition_name] = "bool"
        state.last_definitions[condition_name] = condition_index
        variable_spec = self.generate_variable_definition(condition_name, state)
        function_name = self.generate_function_name("__condition__",
                                                    variable_spec["name"],
                                                    None
                                                    )
        # Getting the output variable
        output_regex = re.compile(r'.*::(?P<output>.*?)::(?P<index>.*$)')
        output_match = output_regex.match(variable_spec['name'])
        if output_match:
            output = output_match.group('output')
            index = output_match.group('index')
            output_variable = f"@variable::{output}::" \
                              f"{index}"
            condition_output = {"variable": output, "index": int(index)}
        else:
            assert False, f"Could not match output variable for " \
                          f"{variable_spec['name']}"

        fn = {
            "function": function_name,
            "input": [
                f"@variable::{src['var']['variable']}::{src['var']['index']}"
                for src in condition_sources
                if 'var' in src
            ],
            "output": output_variable,
            "updated": []
        }
        grfn["variables"].append(variable_spec)
        grfn["functions"].append(fn)

        # TODO Update this
        lambda_string = self._generate_lambda_function(
            node.test,
            function_name["name"],
            False,
            False,
            [src["var"]['variable'] for src in condition_sources if
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
            grfn["variables"] += spec["variables"]

        for spec in else_grfn:
            grfn["functions"] += spec["functions"]
            grfn["variables"] += spec["variables"]

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
            # For `__decision__` nodes, change the index of the inputs into 1
            # (for True) and 0 (for False) instead of the old indices.
            # So, a `decision` lambda function will have the false value
            # first, the true value second, and then the conditional
            # variable. The fixed version of the lambda will look like
            # this:
            # def SIR_Gillespie_SD__gillespie__loop_2__decision__n_S__1(
            # n_S_0, n_S_1, IF_1):
            #       return n_S_1 if IF_1 else n_S_0
            # In the code below, change "index": 0 to "index": versions[-1]
            # and "index": 1 to "index": versions[-2] to revert to the old form.
            inputs = (
                [
                    {"variable": updated_definition, "index": 0},
                    {"variable": updated_definition, "index": 1},
                    condition_output,
                ]
                if len(versions) > 1
                else [
                    {"variable": updated_definition, "index": versions[0]},
                    condition_output,
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
            variable_spec = self.generate_variable_definition(
                updated_definition, state)
            function_name = self.generate_function_name("__decision__",
                                                        variable_spec['name'],
                                                        None)
            fn = {
                "function": function_name,
                "input": [
                    f"@variable::{var['variable']}::{var['index']}"
                    for var in inputs
                ],
                "output": f"@variable::{output['variable']}::{output['index']}",
                "updated": []
            }
            lambda_string = self._generate_lambda_function(
                node,
                function_name["name"],
                False,
                True,
                [f"{src['variable']}_{src['index']}" for src in inputs],
                state,
            )
            state.lambda_strings.append(lambda_string)

            grfn["functions"].append(fn)
            grfn["variables"].append(variable_spec)

        if else_node_name == "ast.If":
            # else_definitions = state.last_definitions.copy()
            else_state = state.copy(last_definitions=state.last_definitions)
            elseif_grfn = self.gen_grfn(node.orelse, else_state, "if")
            for spec in elseif_grfn:
                grfn["functions"] += spec["functions"]
                grfn["variables"] += spec["variables"]

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
        # DEBUG
        print ("expressions: ", expressions)
        print ("state.variable_types: ", state.variable_types)
        grfn = {"functions": [], "variables": [], "containers": []}

        for expr in expressions:
            if "call" not in expr:
                assert False, f"Unsupported expr: {expr}."
        for expr in expressions:
            array_set = False
            call = expr["call"]
            function_name = call["function"]
            io_match = self._check_io_variables(function_name)
            if io_match:
                return []
            # Bypassing calls to `print` for now. Need further discussion and
            # decisions to move forward with what we'll do with `print`
            # statements.
            if function_name == "print":
                return []
            # A handler for array <.set_> function
            if ".set_" in function_name:
                array_set = True
                name = function_name.replace(".set_", "")
                """
                if "var" in call["inputs"][0][0]:
                    index = call["inputs"][0][0]["var"]["variable"]
                    # An array name with index holder for later usage
                    # dueing lambda string generation.
                    state.array_assign_name = f"{name}[{index}]"
                else:
                    index = call["inputs"][0][0]["value"]

                array_name = f"{name}_{index}"
                namespace = self._get_namespace(self.fortran_file)
                namespace = self.replace_multiple(namespace, ['$', '-', ':'], '_')
                cur_scope = self.current_scope
                if len(cur_scope) == 0:
                    scope_path = "global"
                container_id_name = f"{namespace}__{cur_scope}__assign_" \
                                    f"_{array_name}__0"
                """
                # DEBUG
                print ("call: ", call)
                arr_index = call["inputs"][0][0]["var"]["variable"]
                variable_spec = self.generate_variable_definition(name, state)
                # DEBUG
                print ("variable_spec: ", variable_spec['name'])
                assign_function = self.generate_function_name("__assign__",
                                                              variable_spec['name'],
                                                              arr_index)
                container_id_name = assign_function["name"]
                function_type = assign_function["type"]

                # DEBUG
                print ("container_id_name: ", container_id_name)
            else:
                container_id_name = self.function_argument_map[function_name][
                    "name"]
                # ty: type
                function_type = "container"

            function = {
                "function": {
                    "name": container_id_name,
                    "type": function_type
                },
                "input": [],
                "output": None,
                "updated": []
            }

            # Array itself needs to be added
            # as an input, so check that it's
            # and array. If yes, then add it manually.
            if array_set:
                function["input"].append(
                        f"@variable::"
                        f"{name}::-1")

            argument_list = []
            array_index = 0
            for arg in call["inputs"]:
                if len(arg) == 1:
                    # TODO: Only variables are represented in function
                    #  arguments. But a function can have strings as
                    #  arguments as well. Do we add that?
                    if "var" in arg[0]:
                        if arg[0]['var']['variable'] not in argument_list:
                            function["input"].append(
                                f"@variable::"
                                f"{arg[0]['var']['variable']}::"
                                f"{arg[0]['var']['index']}")
                        if array_set:
                            argument_list.append(arg[0]['var']['variable'])
                            if array_index > 0:
                                # Generate lambda function for array[index]
                                lambda_string = self._generate_lambda_function(
                                    node,
                                    container_id_name,
                                    True,
                                    True,
                                    argument_list,
                                    state,
                                )
                                state.lambda_strings.append(lambda_string)
                            array_index += 1
                    elif "call" in arg[0]:
                        function = self.generate_array_setter(
                                                        node, function, arg, 
                                                        name, container_id_name,
                                                        state)
                    elif (
                            "type" in arg[0]
                            and array_set
                    ):
                        # Generate lambda function for array[index]
                        lambda_string = self._generate_lambda_function(
                            node,
                            container_id_name,
                            True,
                            True,
                            argument_list,
                            state,
                        )
                        state.lambda_strings.append(lambda_string)
                else:
                    if "call" in arg[0]:
                        if name in self.arrays:
                            # If array type is <float> the argument holder
                            # has a different structure that it does not hold
                            # function info. like when an array is 'int' type
                            # [{'call': {'function': '_type_', 'inputs': [...]]
                            # which causes an error. Thus, the code below fixes
                            # by correctly structuring it.
                            array_type = self.arrays[name]['elem_type']
                            fixed_arg = [{'call': {
                                                    'function': array_type,
                                                    'inputs':[arg]}}]
                            function = self.generate_array_setter(
                                                            node, function, fixed_arg, 
                                                            name, container_id_name,
                                                            state)
                    else:
                        raise For2PyError(
                            "Only 1 input per argument supported right now."
                        )

            # Keep a track of all functions whose `update` might need to be
            # later updated, along with their scope.
            if len(function['input']) > 0:
                # self.update_functions.append(function_name)
                self.update_functions[function_name] = {
                    "scope": self.current_scope,
                    "state": state
                }

            grfn["functions"].append(function)
        # DEBUG
        print ("grfn: ", grfn)
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
            if (
                    isinstance(node.ctx, ast.Store)
                    and state.next_definitions.get(node.id)
                    and call_source == "annassign"
            ):
                last_definition = self._get_next_definition(
                    node.id,
                    state.last_definitions,
                    state.next_definitions,
                    state.last_definition_default,
                )

            # TODO Change this structure. This is not required for the new
            #  spec. It made sense for the old spec but now it is not required.
            return [{"var": {"variable": node.id, "index": last_definition}}]

    def process_annotated_assign(self, node, state, *_):
        """
            This function handles annotated assignment operations i.e.
            ast.AnnAssign. This tag appears when a variable has been assigned
            with an annotation e.g. x: int = 5, y: List[float] = [None], etc.
        """
        # Get the sources and targets of the annotated assignment
        sources = self.gen_grfn(node.value, state, "annassign")
        targets = self.gen_grfn(node.target, state, "annassign")

        # If the source i.e. assigned value is `None` (e.g. day: List[int] =
        # [None]), only update the data type of the targets and populate the
        # `annotated_assigned` map. No further processing will be done.
        if len(sources) == 1 and ('value' in sources[0].keys()) and not \
                sources[0]['value']:
            for target in targets:
                state.variable_types[target["var"]["variable"]] = \
                    self._get_variable_type(node.annotation)
                if target["var"]["variable"] not in self.annotated_assigned:
                    self.annotated_assigned.append(target["var"]["variable"])
            return []

        grfn = {"functions": [], "variables": [], "containers": []}

        # Only a single target appears in the current version. The `for` loop
        # seems unnecessary but will be required when multiple targets start
        # appearing (e.g. a = b = 5).
        for target in targets:
            target_name = target["var"]["variable"]
            # Preprocessing and removing certain Assigns which only pertain
            # to the Python code and do not relate to the FORTRAN code in any
            # way.
            io_match = self._check_io_variables(target_name)
            if io_match:
                self.exclude_list.append(target_name)
                return []
            state.variable_types[target_name] = \
                self._get_variable_type(node.annotation)
            if target_name not in self.annotated_assigned:
                self.annotated_assigned.append(target_name)
            # Update the `next_definition` index of the target since it is
            # not being explicitly done by `process_name`.
            # TODO Change this functionality ground up by modifying
            #  `process_name` and `process_subscript` to make it simpler.
            if not state.next_definitions.get(target_name):
                state.next_definitions[target_name] = target[
                    "var"]["index"] + 1
            # DEBUG
            print ("target_name: ", target_name)
            variable_spec = self.generate_variable_definition(target_name,
                                                              state)
            function_name = self.generate_function_name(
                            "__assign__",
                            variable_spec['name'],
                            None
            )

            # TODO Somewhere around here, the Float32 class problem will have
            #  to be handled.
            fn = self.make_fn_dict(function_name, target, sources)

            if len(sources) > 0:
                lambda_string = self._generate_lambda_function(
                    node,
                    function_name["name"],
                    False,
                    True,
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

            grfn["functions"].append(fn)
            grfn["variables"].append(variable_spec)

        return [grfn]

    def process_assign(self, node, state, *_):
        """
            This function handles an assignment operation (ast.Assign).
        """
        # Get the GrFN element of the RHS side of the assignment which are
        # the variables involved in the assignment operations.
        sources = self.gen_grfn(node.value, state, "assign")

        array_assignment = False
        # If current assignment is for Array declaration,
        # we need to extract information (dimension, index, and type)
        # of the array based on its dimension (single or multi-).
        if (
                "call" in sources[0]
                and sources[0]["call"]["function"] == "Array"
        ):
            array_assignment = True
            array_dimensions = []
            inputs = sources[0]["call"]["inputs"]
            array_type = inputs[0][0]["var"]["variable"]
            self.get_array_dimension(sources, array_dimensions, inputs)

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

        grfn = {"functions": [], "variables": [], "containers": []}
        # Again as above, only a single target appears in current version.
        # The `for` loop seems unnecessary but will be required when multiple
        # targets start appearing.
        for target in targets:
            # Bypass any assigns that have multiple targets.
            # E.g. (i[0], x[0], j[0], y[0],) = ...
            if "list" in target:
                return []
            target_name = target["var"]["variable"]
            # Preprocessing and removing certain Assigns which only pertain
            # to the Python code and do not relate to the FORTRAN code in any
            # way.
            io_match = self._check_io_variables(target_name)
            if io_match:
                self.exclude_list.append(target_name)
                return []

            # If the target is a list of variables, the grfn notation for the
            # target will be a list of variable names i.e. "[a, b, c]"
            # TODO: This does not seem right. Discuss with Clay and Paul
            #  about what a proper notation for this would be
            if target.get("list"):
                targets = ",".join(
                    [x["var"]["variable"] for x in target["list"]]
                )
                target = {"var": {"variable": targets, "index": 1}}

            if array_assignment:
                var_name = target["var"]["variable"]
                array_info = {
                    "index": target["var"]["index"],
                    "dimensions": array_dimensions,
                    "elem_type": array_type,
                    "mutable": True,
                }
                self.arrays[var_name] = array_info
                state.array_types[var_name] = array_type

            # DEBUG
            print ("target_name: ", target_name)
            variable_spec = self.generate_variable_definition(target_name,
                                                              state)

            function_name = self.generate_function_name("__assign__",
                                                        variable_spec['name'],
                                                        None
                                                        )
            # DEBUG
            print ("    **function_name: ", function_name)
            fn = self.make_fn_dict(function_name, target, sources)

            if len(fn) == 0:
                return []
            source_list = self.make_source_list_dict(sources)
            lambda_string = self._generate_lambda_function(
                node, function_name["name"], False, True,
                source_list, state
            )
            state.lambda_strings.append(lambda_string)

            grfn["functions"].append(fn)
            grfn["variables"].append(variable_spec)

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

            # The `function_node` can be a ast.Name (e.g. Format(format_10)
            # where `format_10` will be an ast.Name or it can have another
            # ast.Attribute (e.g. Format(main.file_10.readline())).
            # Currently, only these two nodes have been detected, so test for
            # these will be made.
            if isinstance(function_node.value, ast.Name):
                module = function_node.value.id
                function_name = function_node.attr
                function_name = module + "." + function_name
            elif isinstance(function_node.value, ast.Attribute):
                module = self.gen_grfn(function_node.value, state, "call")
                function_name = function_node.attr
                function_name = module + "." + function_name
            else:
                assert False, f"Invalid expression call {function_node}"
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
        merged_grfn = [self._merge_dictionary(grfn_list)]

        # We fill in the `updated` field of function calls by looking at the
        # `updated` field of their container grfn
        final_grfn = self.load_updated(merged_grfn)
        return final_grfn

    @staticmethod
    def _process_nameconstant(node, *_):
        # TODO Change this format according to the new spec
        return [
            {"type": "literal", "dtype": "string", "value": node.value}
        ]

    def process_attribute(self, node, state, call_source):
        """
            Handle Attributes: This is a fix on `feature_save` branch to
            bypass the SAVE statement feature where a SAVEd variable is
            referenced as <function_name>.<variable_name>. So the code below
            only returns the <variable_name> which is stored under
            `node.attr`. The `node.id` stores the <function_name> which is
            being ignored.
        """
        # If this node appears inside an ast.Call processing, then this is
        # the case where a function call has been saved in the case of IO
        # handling. E.g. format_10_obj.read_line(main.file_10.readline())).
        # Here, main.file_10.readline is
        if call_source == "call":
            module = node.attr
            return module
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

            # TODO Change the format according to the new spec
            return [{"var": {"variable": node.attr, "index": last_definition}}]

    def process_return_value(self, node, state, *_):
        """
        This function handles the return value from a function.
        """
        grfn = {"functions": [], "variables": [], "containers": []}
        if node.value:
            val = self.gen_grfn(node.value, state, "return_value")
        else:
            val = None

        return_dict = {
            "type": "return",
            "value": val
        }
        grfn["functions"].append(return_dict)
        return [grfn]

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

    @staticmethod
    def _get_namespace(original_fortran_file) -> str:
        """
            This function returns the namespace for every identifier in the
            system being analyzed.
            Currently, this function is very barebone and just returns the
            name of the system being evaluated. After more testing with
            modules and imports, the namespace will expand into more than
            just the system file name.
        """
        namespace_path_list = get_path(original_fortran_file, "namespace")
        namespace_path = ".".join(namespace_path_list)

        # TODO Hack: Currently only the last element of the
        #  `namespace_path_list` is being returned as the `namespace_path` in
        #  order to make it consistent with the handwritten SIR-Demo GrFN
        #  JSON. Will need a more generic path for later instances.
        namespace_path = namespace_path_list[-1]

        return namespace_path

    def make_source_list_dict(self, source_dictionary):
        source_list = []
        for src in source_dictionary:
            if "var" in src:
                if src["var"]["variable"] not in ANNOTATE_MAP:
                    source_list.append(src["var"]["variable"])
            elif "call" in src:
                for ip in src["call"]["inputs"]:
                    source_list.extend(self.make_source_list_dict(ip))
            elif "list" in src:
                for ip in src["list"]:
                    if "var" in ip:
                        source_list.append(ip["var"]["variable"])

        # Removing duplicates
        unique_source = []
        [unique_source.append(obj) for obj in source_list if obj not in
         unique_source]
        source_list = unique_source

        return source_list

    def make_fn_dict(self, name, target, sources):
        source = []
        fn = {}
        target_name = target["var"]["variable"]
        
        target_string = f"@variable::{target_name}::{target['var']['index']}"

        for src in sources:
            # Check for a write to a file
            if re.match(r"\d+", target_name) and "list" in src:
                return fn
            if "call" in src:
                # Remove first index of an array function as it's
                # really a type name not the variable for input.
                if src["call"]["function"] is "Array":
                    del src["call"]["inputs"][0]
                # If a RHS of an assignment is an array getter,
                # for example, meani.get_((runs[0])), we only need
                # the array name (meani in this case) and append
                # to source.
                if ".get_" in src["call"]["function"]:
                    get_array_name = src["call"]["function"].replace(".get_", "")
                    name = f"@variable::{get_array_name}::-1"
                    source.append({
                                    "name": get_array_name,
                                    "type": "variable"})

                # Bypassing identifiers who have I/O constructs on their source
                # fields too.
                # Example: (i[0],) = format_10_obj.read_line(file_10.readline())
                # 'i' is bypassed here
                # TODO this is only for PETASCE02.for. Will need to include 'i'
                #  in the long run
                bypass_match_source = self._check_io_variables(
                    src["call"]["function"]
                )
                if bypass_match_source:
                    if "var" in src:
                        self.exclude_list.append(src["var"]["variable"])
                    return fn
                # TODO Finalize the spec for calls here of this form:
                #  "@container::<namespace_path_string>::<scope_path_string>::
                #   <container_base_name>" and add here.
                for source_ins in self.make_call_body_dict(src):
                    source.append(source_ins)

            elif "var" in src:
                source_string = f"@variable::{src['var']['variable']}::" \
                                f"{src['var']['index']}"
                source.append(source_string)
            # The code below is commented out to not include any `literal`
            # values in the input of `function` bodies. The spec does mention
            # including `literals` so if needed, uncomment the code block below

            # elif "type" in src and src["type"] == "literal":
            #     variable_type = self.type_def_map[src["dtype"]]
            #     source_string = f"@literal::{variable_type}::{src['value']}"
            #     source.append(source_string)
            # else:
            #     assert False, f"Unidentified source: {src}"

        # Removing duplicates
        unique_source = []
        [unique_source.append(obj) for obj in source if obj not in
         unique_source]
        source = unique_source

        fn = {
            "function": name,
            "input": source,
            "output": target_string,
            "updated": []
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
            "sin", etc to the source list. The following two lines when
            commented helps us do that. If user-defined functions come up as
            sources, some other approach might be required.
        """
        # TODO Try with user defined functions and see if the below two lines
        #  need to be reworked
        # name = source["call"]["function"]
        # source_list.append({"name": name, "type": "function"})

        source_list = []
        for ip in source["call"]["inputs"]:
            if isinstance(ip, list):
                for item in ip:
                    if "var" in item:
                        source_string = f"@variable::" \
                                        f"{item['var']['variable']}::" \
                                        f"{item['var']['index']}"
                        source_list.append(source_string)
                    elif "call" in item:
                        source_list.extend(self.make_call_body_dict(item))
                    elif "list" in item:
                        # Handles a case where array declaration size
                        # was given with a variable value.
                        for value in item["list"]:
                            if "var" in value:
                                variable = f"@variable::{value['var']['variable']}::0"
                                source_list.append(variable)

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
            into another dictionary in a managed manner. The `dicts` argument is
            a list of form [{}, {}, {}] where each {} dictionary is the grfn
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
    def _get_variables_and_functions(grfn):
        variables = list(chain.from_iterable(stmt["variables"] for stmt in
                                             grfn))
        fns = list(chain.from_iterable(stmt["functions"] for stmt in grfn))
        containers = list(chain.from_iterable(stmt["containers"] for stmt in
                                              grfn))
        return variables, fns, containers

    def generate_gensym(self, tag):
        """
            The gensym is used to uniquely identify any identifier in the
            program. Python's uuid library is used to generate a unique 12 digit
            HEX string. The uuid4() function of 'uuid' focuses on randomness.
            Each and every bit of a UUID v4 is generated randomly and with no
            inherent logic. To every gensym, we add a tag signifying the data
            type it represents. 'v' is for variables and 'h' is for holders.
        """
        return f"{self.gensym_tag_map[tag]}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _generate_lambda_function(node, function_name: str, return_value: bool,
                                  array_assign: bool, inputs, state):
        lambda_for_var = True
        lambda_strings = []
        argument_strings = []
        # Sort the arguments in the function call as it is used in the operation
        input_list = sorted(set(inputs), key=inputs.index)
        # Add type annotations to the function arguments
        for ip in input_list:
            annotation = state.variable_types.get(ip)
            if ip in state.array_types:
                lambda_for_var = False
            if (
                lambda_for_var
                and not annotation
            ):
                # `variable_types` does not contain annotations for variables
                # for indexing such as 'abc_1', etc. Check if the such variables
                # exist and assign appropriate annotations
                key_match = lambda var, dicn: ([i for i in dicn if i in var])
                annotation = state.variable_types[
                    key_match(ip, state.variable_types)[0]
                ]
            # function argument requires annotation only when
            # it's dealing with simple variable (at least for now).
            if lambda_for_var:
                annotation = ANNOTATE_MAP[annotation]
                argument_strings.append(f"{ip}: {annotation}")
            # Currently, this is for array specific else case.
            else:
                argument_strings.append(ip)
                lambda_for_var = True

        lambda_strings.append(
            f"def {function_name}({', '.join(argument_strings)}):\n    "
        )
        # If a `decision` tag comes up, override the call to genCode to manually
        # enter the python script for the lambda file.
        if "__decision__" in function_name:
            code = f"{inputs[1]} if {inputs[2]} else {inputs[0]}"
        else:
            lambda_code_generator = genCode()
            code = lambda_code_generator.generate_code(node,
                                                       PrintState("\n    ")
                                                       )
        if return_value:
            if array_assign:
                lambda_strings.append(f"{state.array_assign_name} = {code}\n")
                lambda_strings.append(f"    return {state.array_assign_name}")
                state.array_assign_name = None
            else:
                lambda_strings.append(f"return {code}")
        else:
            lines = code.split("\n")
            indent = re.search("[^ ]", lines[-1]).start()
            lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
            lambda_strings.append("\n".join(lines))
        lambda_strings.append("\n\n")
        return "".join(lambda_strings)

    def generate_container_id_name(self, namespace_file: str, scope_path: str,
                                   container_basename: str) -> str:
        namespace = self._get_namespace(namespace_file)
        if isinstance(scope_path, list):
            scope_path_string = '.'.join(scope_path)
        elif isinstance(scope_path, str):
            scope_path_string = scope_path
        else:
            assert False, f"Invalid scope_path type {scope_path}"
        container_id = f"@container::{namespace}::{scope_path_string}::" \
            f"{container_basename}"

        return container_id

    def generate_variable_definition(self, variable, state):
        """
            This function generates the GrFN structure for a variable
            definition, of the form:
            variable: {
                        name:
                        source_refs:
                        gensym:
                        domain:
                        domain_constraints:
                        }
        """
        namespace = self._get_namespace(self.fortran_file)
        # DEBUG
        print ("    --state.last_definitions: ", state.last_definitions)
        if variable in state.last_definitions:
            index = state.last_definitions[variable]
        elif variable in self.arrays:
            index = 0

        variable_name = f"@variable::{namespace}::{self.current_scope}::" \
            f"{variable}::{index}"
        variable_gensym = self.generate_gensym("variable")
        domain = self.get_domain_dictionary(variable, state)

        # TODO Change the domain constraint. How do you figure the domain
        #  constraint out?
        domain_constraint = "(and (> v -infty) (< v infty)))"

        variable_definition = {
            "name": variable_name,
            "gensym": variable_gensym,
            "source_refs": [],
            "domain": domain,
            "domain_constraint": domain_constraint,
        }

        return variable_definition

    def get_domain_dictionary(self, variable, state):
        if variable in self.arrays:
            domain_dictionary = self.arrays[variable]
        else:
            variable_type = state.variable_types[variable]
            domain_dictionary = {
                "name": self.type_def_map[variable_type],
                "type": "type"
            }
        return domain_dictionary

    def generate_function_name(self, function_type, variable, arr_index):
        """
            This function generates the name of the function inside the
            container wiring within the body of a container.
        """
        # DEBUG
        print ("variable: ", variable)
        print ("arr_index: ", arr_index)
        variable_spec_regex = r'@.*?::(?P<namescope>.*?::.*?)::(' \
                              r'?P<variable>.*?)::(?P<index>.*)'
        variable_match = re.match(variable_spec_regex, variable)
        if variable_match:
            namespace_scope = variable_match.group("namescope")
            variable_name = variable_match.group("variable")
            if arr_index:
                variable_name = variable_name + f"_{arr_index}"
            variable_index = variable_match.group("index")

            name = namespace_scope + function_type + variable_name + "::" + \
                variable_index
            name = self.replace_multiple(name, ['$', '-', ':'], '_')
            name = name.replace('.', '__')
            if any([x in function_type for x in ["assign", "condition",
                                                 "decision"]]):
                spec_type = "lambda"
            else:
                spec_type = "None"
        else:
            assert False, f"Cannot match regex for variable spec: {variable}"

        return {"name": name, "type": spec_type}

    def load_updated(self, grfn_dict):
        """
            This function parses through the GrFN once and finds the
            container spec of functions whose `updated` fields needs to be
            filled in that functions' function call spec.
        """
        for container in self.function_argument_map:
            if container in self.update_functions:
                for container_grfn in grfn_dict[0]['containers']:
                    for body_function in container_grfn['body']:
                        function_name = body_function['function']['name']
                        if function_name.startswith('@container') and \
                                function_name.split('::')[-1] == container:
                            updated_variable = [body_function['input'][i] for
                                                i in self.function_argument_map[
                                                    container][
                                                    'updated_indices']]
                            for i in range(len(updated_variable)):
                                old_index = int(updated_variable[i].split(
                                    "::")[-1])
                                new_index = old_index + 1
                                updated_var_list = updated_variable[i].split(
                                    "::")[:-1]
                                updated_var_list.append(str(new_index))
                                updated_variable[i] = '::'.join(
                                    updated_var_list)
                                self.current_scope = self.update_functions[
                                    container]['scope']
                                variable_name = updated_var_list[1]
                                variable_spec = \
                                    self.generate_variable_definition(
                                        variable_name, self.update_functions[
                                            container]['state']
                                    )
                                variable_name_list = variable_spec[
                                                         'name'
                                                     ].split("::")[:-1]
                                variable_name_list.append(str(new_index))
                                variable_spec['name'] = "::".join(
                                    variable_name_list
                                )
                                grfn_dict[0]['variables'].append(variable_spec)
                            body_function['updated'] = updated_variable

        return grfn_dict

    def get_array_dimension (self, sources, array_dimensions, inputs):
        """This function is for extracting bounds of an array.

            Args:
                sources (list): A list holding GrFN element of
                array function. For example, Array (int, [[(0, 10)]).
                array_dimensions (list): An empty list that will be
                populated by current function with the dimension info.
                inputs (list): A list that holds inputs dictionary
                extracted from sources.

            Returns:
                None.
        """
        # A multi-dimensional array handler
        if "list" in inputs[1][0]["list"][0]:
            for lists in inputs[1][0]["list"]:
                low_bound = int(lists["list"][0]["value"])
                upper_bound = int(lists["list"][1]["value"])
                array_dimensions.append(upper_bound-low_bound+1)
        # 1-D array handler
        else:
            bounds = inputs[1][0]["list"]
            # Get lower bound of an array
            if "type" in bounds[0]:
                # When an index is a scalar value
                low_bound = bounds[0]["value"]
            else:
                # When an index is a variable
                low_bound = bounds[0]["var"]["variable"]
            # Get upper bound of an array
            if "type" in bounds[1]:
                upper_bound = bounds[1]["value"]
            else:
                upper_bound = bounds[1]["var"]["variable"]

            if isinstance(low_bound, int) and isinstance(upper_bound, int):
                array_dimensions.append(upper_bound-low_bound+1)
            elif isinstance(upper_bound, str):
                assert (
                    isinstance(low_bound, int) and low_bound == 0
                ), "low_bound must be <integer> type and 0 (zero) for now."
                array_dimensions.append(upper_bound)
            else:
                assert False, f"low_bound type: {type(low_bound)} is currently not handled."

    def generate_array_setter (self, node, function, arg, name, container_id_name, state):
        """
            This function is for handling array setter (ex. means.set_(...)).
            
            Args:
                function (list): A list holding the information of the function
                for JSON and lambda function generation.
                arg (list): A list holding the arguments of call['inputs'].
                name (str): A name of the array.
                container_id_name (str): A name of function container. It's an
                array name with other appended info. in this function.

            Returns:
                (list) function: A completed list of function.
        """
        argument_list = []
        input_list = []
        function["output"] = []
        # For array setter value handler
        for var in arg[0]["call"]["inputs"][0]:
            # If an input is a simple variable
            if "var" in var:
                var_name = var['var']['variable']
                if var_name not in argument_list:
                    function["input"].append(
                            f"@variable::"
                            f"{var_name}::-1")
                    argument_list.append(var_name)
                else:
                    # It's not an error, so just pass it.
                    pass
            # If an input is an array (for now).
            elif "call" in var:
                ref_call = var["call"]
                if ".get_" in ref_call["function"]:
                    get_array_name = ref_call["function"].replace(".get_", "")
                    if get_array_name not in argument_list:
                        function["input"].append(
                            f"@variable::"
                            f"{get_array_name}::-1")
                        argument_list.append(get_array_name)
                    else:
                        # It's not an error, so just pass it.
                        pass
        # Generate lambda function for array[index]
        lambda_string = self._generate_lambda_function(
            node,
            container_id_name,
            True,
            True,
            argument_list,
            state,
        )
        state.lambda_strings.append(lambda_string)
        function["output"].append(
                f"@variable::"
                f"{name}::0")

        return function

    @staticmethod
    def replace_multiple(main_string, to_be_replaced, new_string):
        """
            Replace a set of multiple sub strings with a new string in main
            string.
        """
        # Iterate over the strings to be replaced
        for elem in to_be_replaced:
            # Check if string is in the main string
            if elem in main_string:
                # Replace the string
                main_string = main_string.replace(elem, new_string)

        return main_string

    @staticmethod
    def _find_updated(argument_list, body_variable_list):
        """
            This function finds and generates a list of updated identifiers
            in a container.
        """
        # TODO After implementing everything, check if `argument_dict` and
        #  `body_dict` will be the same as `function_state.last_definitions`
        #  before and after getting `body_grfn`. If so, remove the creation
        #  of `argument_dict` and `body_dict` and use the `last_definitions`
        #  instead
        argument_dict = {}
        body_dict = {}
        updated_list = []
        variable_regex = re.compile(r'.*::(?P<variable>.*?)::(?P<index>.*$)')
        # First, get mapping of argument variables and their indexes
        for var in argument_list:
            var_match = variable_regex.match(var["name"])
            if var_match:
                argument_dict[var_match.group('variable')] = var_match.group(
                    'index')
            else:
                assert False, f"Error when parsing argument variable " \
                              f"{var['name']}"
        # Now, get mapping of body variables and their latest indexes
        for var in body_variable_list:
            var_match = variable_regex.match(var["name"])
            if var_match:
                body_dict[var_match.group('variable')] = \
                    var_match.group('index')
            else:
                assert False, f"Error when parsing body variable " \
                              f"{var['name']}"

        # Now loop through every argument variable over the body variable to
        # check if the indices mismatch which would indicate an updated variable
        for argument in argument_dict:
            if argument in body_dict:
                updated_list.append(f"@variable::{argument}::"
                                    f"{body_dict[argument]}")

        return updated_list

    @staticmethod
    def _check_io_variables(variable_name):
        """
            This function scans the variable and checks if it is an io
            variable. It returns the status of this check i.e. True or False.
        """
        io_match = RE_BYPASS_IO.match(variable_name)
        return io_match


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
        grfn["start"] = [grfn["start"][0]]
    else:
        # TODO: The `grfn_spec` mentions this to be null (None) but it looks
        #  like `networks.py` requires a certain function. Finalize after
        #  `networks.py` is completed.
        # grfn["start"] = None
        grfn["start"] = [generator.function_definitions[-1]]

    # Add the placeholder to enter the grounding and link hypothesis information
    grfn["grounding"] = []
    # TODO Add a placeholder for `types`. This will have to be populated when
    #  user defined types start appearing.
    grfn["types"] = []
    # Get the file path of the original Fortran code being analyzed
    source_file = get_path(original_file, "source")

    # TODO Hack: Currently only the file name is being displayed as the
    #  source in order to match the handwritten SIR model GrFN JSON. Since
    #  the directory of the `SIR-Gillespie-SD.f` file is the root, it works
    #  for this case but will need to be generalized for other cases.
    file_path_list = source_file.split("/")
    grfn["source"] = [file_path_list[-1]]

    # Get the source comments from the original Fortran source file.
    source_comments = str(dict(get_comments(original_file)))
    grfn["source_comments"] = source_comments

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
