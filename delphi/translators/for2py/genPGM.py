#!/usr/bin/python3.6

import ast
import sys
import tokenize
import pickle
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
from collections import OrderedDict
from itertools import chain, product
import operator
import uuid
import os.path


# noinspection PyDefaultArgument
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
        array_assign_name: Optional = None,
        string_assign_name: Optional = None,
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
        self.string_assign_name = string_assign_name

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
        string_assign_name: Optional = None,
    ):
        return GrFNState(
            self.lambda_strings if lambda_strings is None else lambda_strings,
            self.last_definitions
            if last_definitions is None
            else last_definitions,
            self.next_definitions
            if next_definitions is None
            else next_definitions,
            self.last_definition_default
            if last_definition_default is None
            else last_definition_default,
            self.function_name if function_name is None else function_name,
            self.variable_types if variable_types is None else variable_types,
            self.start if start is None else start,
            self.scope_path if scope_path is None else scope_path,
            self.arrays if arrays is None else arrays,
            self.array_types if array_types is None else array_types,
            self.array_assign_name
            if array_assign_name is None
            else array_assign_name,
            self.string_assign_name
            if string_assign_name is None
            else string_assign_name,
        )


# noinspection PyDefaultArgument,PyTypeChecker
class GrFNGenerator(object):
    def __init__(self, annotated_assigned=[], function_definitions=[]):
        self.annotated_assigned = annotated_assigned
        self.function_definitions = function_definitions
        self.use_numpy = True
        self.fortran_file = None
        self.exclude_list = []
        self.loop_input = []
        self.comments = {}
        self.update_functions = {}
        self.mode_mapper = {}
        self.name_mapper = {}
        self.variable_map = {}
        self.function_argument_map = {}
        # Holds all declared arrays {symbol:domain}
        self.arrays = {}
        # Holds declared array types {symbol:type}
        self.array_types = {}
        self.array_assign_name = None
        # Holds a list of multi-dimensional array symbols
        self.md_array = []
        # Holds all the string assignments along with their length
        self.strings = {}
        self.outer_count = 0
        self.types = (list, ast.Module, ast.FunctionDef)
        self.current_scope = "global"
        self.loop_index = -1
        self.if_index = -1
        self.parent_loop_state = None
        self.parent_if_state = None
        self.handling_f_args = True
        self.f_array_arg = []
        # {symbol:index}
        self.updated_arrays = {}
        # {symbol: [_list_of_domains_]}
        # This mapping is required as there may be
        # a multiple array passes to the function
        # argument and we do not want to replace one
        # with another. We need to update all function
        # argument domains for arrays
        self.array_arg_domain = {}
        # Holds list of modules that program references
        self.module_variable_types = {}
        # Holds the list of declared subprograms in modules
        self.module_subprograms = []
        # Holds module names
        self.module_names = []
        # List of generated lambda function def. names
        self.generated_lambda_functions = []
        # List of user-defined (derived) types
        self.derived_types = []
        # List of derived type grfns
        self.derived_types_grfn = []
        # List of attributes declared under user-defined types
        self.derived_types_attributes = {}
        # List of variables (objects) declared with user-defined type
        self.derived_type_objects = {}
        # Currently handling derived type object name
        self.current_d_object_name = None
        # Currently handling derived type object's accessing attributes
        self.current_d_object_attributes = []
        self.is_d_object_array_assign = False
        self.module_summary = None
        self.global_scope_variables = {}
        self.exit_candidates = []
        self.global_state = None
        self.imported_module = []
        self.global_grfn = {
            "containers": [{
                "name": None,
                "source_refs": [],
                "gensym": None,
                "repeat": False,
                "arguments": [],
                "updated": [],
                "return_value": [],
                "body": [],
            }],
            "variables": [],
        }

        self.gensym_tag_map = {
            "container": "c",
            "variable": "v",
            "function": "f",
        }
        self.type_def_map = {
            "real": "float",
            "integer": "integer",
            "string": "string",
            "bool": "boolean",
            "Array": "Array",
            "character": "string",
            "String": "string",
        }

        # The binops dictionary holds operators for all the arithmetic and
        # comparative functions
        self.binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Eq: operator.eq,
            ast.LtE: operator.le,
        }

        # The annotate_map dictionary is used to map Python ast data types
        # into data types for the lambdas
        self.annotate_map = {
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
            "file_handle": "fh",
            "Array": "Array",
            "String": "string",
        }

        # Arithmetic operator dictionary
        # TODO: Complete the list
        self.arithmetic_ops = {
            "Add": "+",
            "Sub": "-",
            "Div": "/",
            "Mult": "*",
            "+": "+",
            "-": "-",
            "/": "/",
            "*": "*",
        }

        # The unnecessary_types tuple holds the ast types to ignore
        self.unnecessary_types = (
            ast.Mult,
            ast.Add,
            ast.Sub,
            ast.Pow,
            ast.Div,
            ast.USub,
            ast.Eq,
            ast.LtE,
        )

        # List of helper library types
        self.library_types = ["Array", "String"]

        # Regular expression to match python statements that need to be
        # bypassed in the GrFN and lambda files. Currently contains I/O
        # statements.
        self.bypass_io = (
            r"^format_\d+$|^format_\d+_obj$|^file_\d+$|^file_\w+$"
            r"|^write_list_\d+$|^write_line$|^format_\d+_obj"
            r".*|^Format$|^list_output_formats$|"
            r"^write_list_stream$|^file_\d+\.write$|^file_\w+\.write$"
            r"^output_fmt$"
        )

        self.re_bypass_io = re.compile(self.bypass_io, re.I)

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
            "ast.Break": self.process_break,
            "ast.ClassDef": self.process_class_def,
            "ast.Try": self.process_try,
        }

    def gen_grfn(self, node, state, call_source):
        """
            This function generates the GrFN structure by parsing through the
            python AST
        """
        # Look for code that is not inside any function.
        if state.function_name is None and not any(
            isinstance(node, t) for t in self.types
        ):
            # If the node is of instance ast.Call, it is the starting point
            # of the system.
            node_name = node.__repr__().split()[0][2:]
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Name) and \
                    node.func.id != "String" and \
                    node.func.id not in self.derived_types:
                start_function_name = self.generate_container_id_name(
                    self.fortran_file, ["@global"], node.func.id
                )
                return [{"start": start_function_name}]
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call) and \
                    isinstance(node.value.func, ast.Attribute) and \
                        node.value.func.attr == "set_":
                    return self.process_grfn[node_name](
                        node, state, call_source
                    )
                else:
                    return self.gen_grfn(node.value, state, "start")
            elif isinstance(node, ast.If):
                return self.gen_grfn(node.body, state, "start")
            else:
                if node_name != "ast.Import" and node_name != "ast.ImportFrom":
                    return self.process_grfn[node_name](
                        node, state, call_source
                    )
                else:
                    # Check if this is a module that is imported
                    if node_name == "ast.ImportFrom" and "m_" in \
                            node.module.split(".")[-1]:
                        imported_module = node.module.split(".")[-1][2:]
                        self.imported_module.append(imported_module)
                        self.derived_types.extend(self.module_summary[
                                                      imported_module][
                                                      "derived_type_list"])
                    return []
        elif isinstance(node, list):
            return self.process_list(node, state, call_source)
        elif any(
            isinstance(node, node_type) for node_type in self.unnecessary_types
        ):
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

        # result = []
        # for cur in node:
        #     tmp = self.gen_grfn(cur, state, call_source)
        #     result.append(tmp)
        # return list(chain.from_iterable(result))

        return list(
            chain.from_iterable(
                [self.gen_grfn(cur, state, call_source) for cur in node]
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
        self.module_names.append(node.name)

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
            self.process_decorators(node.decorator_list, function_state)

        # Check if the function contains arguments or not. This determines
        # whether the function is the outermost scope (does not contain
        # arguments) or it is not (contains arguments). For non-outermost
        # scopes, indexing starts from -1 (because all arguments will have
        # an index of -1). For outermost scopes, indexing starts from
        # normally from 0.
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
            assert self.outer_count == 1, (
                "There is more than one function "
                "without arguments in this system. "
                "This is not currently handled."
            )

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

        # Copy the states of all the global variables into the global state
        # holder
        if not self.global_scope_variables and self.current_scope == "global":
            self.global_scope_variables = state.copy(
                last_definitions=local_last_definitions,
                next_definitions=local_next_definitions,
                function_name=node.name,
                variable_types=local_variable_types,
                last_definition_default=0,
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
            "argument_list": "",
        }
        self.function_argument_map[node.name]["argument_list"] = argument_list
        # Update the current scope so that for every identifier inside the
        # body, the scope information is updated
        self.current_scope = node.name
        # Create the variable definition for the arguments
        argument_variable_grfn = []
        self.handling_f_args = True
        for argument in argument_list:
            argument_variable_grfn.append(
                self.generate_variable_definition(
                    [argument], None, False, function_state
                )
            )
        self.handling_f_args = False
        # Generate the `variable_identifier_name` for each container argument.
        # TODO Currently only variables are handled as container arguments.
        #  Create test cases of other containers as container arguments and
        #  extend this functionality.
        argument_list = [
            f"@variable::{x}::{function_state.last_definitions[x]}"
            for x in argument_list
        ]

        # Enter the body of the function and recursively generate the GrFN of
        # the function body
        body_grfn = self.gen_grfn(node.body, function_state, "functiondef")

        # Get the `return_value` from the body. We want to append it separately.
        # TODO There can be multiple return values. `return_value` should be
        #  a list and you should append to it.
        for body in body_grfn:
            for function in body["functions"]:
                if (
                    function.get("function")
                    and function["function"]["type"] == "return"
                ):
                    return_value = function["value"]
                    # Remove the return_value function body from the main
                    # body as we don't need that anymore
                    body["functions"].remove(function)

        # TODO The return value cannot always be a `variable`. It can be
        #  literals as well. Add that functionality here.
        if return_value:
            for value in return_value:
                if "var" in value:
                    return_list.append(
                        f"@variable::{value['var']['variable']}"
                        f"::{value['var']['index']}"
                    )
                elif "call" in value:
                    for _ in value["call"]["inputs"]:
                        if "var" in value:
                            return_list.append(
                                f"@variable::{value['var']['variable']}::"
                                f"{value['var']['index']}"
                            )
            return_list = list(set(return_list))
        else:
            return_list = []

        # Get the function_reference_spec, function_assign_spec and
        # identifier_spec for the function
        (
            function_variable_grfn,
            function_assign_grfn,
            body_container_grfn,
        ) = self._get_variables_and_functions(body_grfn)
        # Combine the variable grfn of the arguments with that of the
        # container body
        container_variables = argument_variable_grfn + function_variable_grfn
        # Find the list of updated identifiers
        if argument_list:
            updated_identifiers = self._find_updated(
                argument_variable_grfn,
                function_variable_grfn,
                self.f_array_arg,
                function_state,
            )
            for array in self.f_array_arg:
                if array in function_state.last_definitions:
                    self.updated_arrays[
                        array
                    ] = function_state.last_definitions[array]
        else:
            updated_identifiers = []
        self.function_argument_map[node.name][
            "updated_list"
        ] = updated_identifiers

        # Get a list of all argument names
        argument_name_list = []
        for item in argument_list:
            argument_name_list.append(item.split("::")[1])

        # Now, find the indices of updated arguments
        for arg in updated_identifiers:
            updated_argument = arg.split("::")[1]
            argument_index = argument_name_list.index(updated_argument)
            self.function_argument_map[node.name]["updated_indices"].append(
                argument_index
            )

        # Create a gensym for the function container
        container_gensym = self.generate_gensym("container")

        container_id_name = self.generate_container_id_name(
            self.fortran_file, scope_path, node.name
        )
        self.function_argument_map[node.name]["name"] = container_id_name
        # Add the function name to the list that stores all the functions
        # defined in the program
        self.function_definitions.append(container_id_name)

        function_container_grfn = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "type": "function",
            "arguments": argument_list,
            "updated": updated_identifiers,
            "return_value": return_list,
            "body": function_assign_grfn,
        }

        function_container_grfn = [
            function_container_grfn
        ] + body_container_grfn

        # function_assign_grfn.append(function_container_grfn)
        pgm = {
            "containers": function_container_grfn,
            "variables": container_variables,
        }
        return [pgm]

    def process_arguments(self, node, state, call_source):
        """
            This function returns a list of arguments defined in a function
            definition. `node.args` is a list of `arg` nodes which are
            iteratively processed to get the argument name.
        """
        return [self.gen_grfn(arg, state, call_source) for arg in node.args]

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
        state.variable_types[node.arg] = self.get_variable_type(
            node.annotation
        )

        # Check if an argument is a string
        annotation_name = node.annotation.__repr__().split()[0][2:]
        if annotation_name == "ast.Name" and node.annotation.id == "String":
            self.strings[node.arg] = {
                "length": "Undefined",
                "annotation": False,
                "annotation_assign": True,
            }

        if state.last_definitions.get(node.arg) is None:
            if call_source == "functiondef":
                if node.arg not in self.updated_arrays:
                    state.last_definitions[node.arg] = -1
                else:
                    state.last_definitions[node.arg] = self.updated_arrays[
                        node.arg
                    ]
            else:
                assert False, (
                    "Call source is not ast.FunctionDef. "
                    "Handle this by setting state.last_definitions["
                    "node.arg] = 0 in place of the assert False. "
                    "But this case should not occur in general."
                )
        else:
            assert False, (
                "The argument variable was already defined "
                "resulting in state.last_definitions containing an "
                "entry to this argument. Resolve this by setting "
                "state.last_definitions[node.arg] += 1. But this "
                "case should not occur in general."
            )

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

    def process_num(self, node, *_):
        """
            This function handles the ast.Num of the ast tree. This node only
            contains a numeric value in its body. For example: Num(n=0),
            Num(n=17.27), etc. So, we return the numeric value in a
            <function_assign_body_literal_spec> form.

        """
        data_type = self.annotate_map.get(type(node.n).__name__)
        if data_type:
            # TODO Change this format. Since the spec has changed,
            #  this format is no longer required. Go for a simpler format.
            return [{"type": "literal", "dtype": data_type, "value": node.n}]
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
        # Currently, this function is encountered only for annotated
        # assignments of the form, a: List[float] = [None], b: List[float] =
        # [100], c: List[float] = [(x[0]*y[0])], etc. If any further cases
        # arise, the following code might need to be rewritten and the
        # following return code will have to be added as well.
        # return elements if len(elements) == 1 else [{"list": elements}]
        element_grfn = []
        for list_element in node.elts:
            element_grfn.append(self.gen_grfn(list_element, state, "List"))

        return element_grfn

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
        return [{"type": "literal", "dtype": "string", "value": node.s}]

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
        container_return_value = []
        container_updated = []
        function_output = []
        function_updated = []
        function_input = []
        loop_condition_inputs = []
        loop_condition_inputs_lambda = []
        loop_variables_grfn = []
        loop_functions_grfn = []

        # Increment the loop index universally across the program
        if self.loop_index > -1:
            self.loop_index += 1
        else:
            self.loop_index = 0

        # First, get the `container_id_name` of the loop container
        container_id_name = self.generate_container_id_name(
            self.fortran_file, self.current_scope, f"loop${self.loop_index}"
        )

        # Update the scope of the loop container so that everything inside
        # the body of the loop will have the below scope
        self.current_scope = f"{self.current_scope}.loop${self.loop_index}"

        index_variable = self.gen_grfn(node.target, state, "for")
        # Check: Currently, only one variable is supported as a loop variable
        if len(index_variable) != 1 or "var" not in index_variable[0]:
            raise For2PyError("Only one index variable is supported.")
        index_name = index_variable[0]["var"]["variable"]

        # Define a new empty state that will be used for mapping the state of
        # the operations within the for-loop container
        loop_last_definition = {}
        loop_state = state.copy(
            last_definitions=loop_last_definition,
            next_definitions={},
            last_definition_default=-1,
        )

        # We want the loop_state to have state information about variables
        # defined one scope above its current parent scope. The below code
        # allows us to do that
        if self.parent_loop_state:
            for var in self.parent_loop_state.last_definitions:
                if var not in state.last_definitions:
                    # state.last_definitions[var] = \
                    #     self.parent_loop_state.last_definitions[var]
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
        loop_condition_inputs_lambda.append(index_name)
        for ip in range_call["inputs"]:
            for var in ip:
                if "var" in var:
                    function_input.append(
                        f"@variable::"
                        f"{var['var']['variable']}::"
                        f"{var['var']['index']}"
                    )
                    container_argument.append(
                        f"@variable::" f"{var['var']['variable']}::-1"
                    )
                    loop_condition_inputs.append(
                        f"@variable::" f"{var['var']['variable']}::-1"
                    )
                    loop_condition_inputs_lambda.append(var["var"]["variable"])
                elif "call" in var:
                    # TODO: Very specifically for arrays. Will probably break
                    #  for other calls
                    self._get_call_inputs(
                        var["call"],
                        function_input,
                        container_argument,
                        loop_condition_inputs,
                        loop_condition_inputs_lambda,
                        state,
                    )
        function_input = self._remove_duplicate_from_list(function_input)
        container_argument = self._remove_duplicate_from_list(
            container_argument
        )
        loop_condition_inputs = self._remove_duplicate_from_list(
            loop_condition_inputs
        )
        loop_condition_inputs_lambda = self._remove_duplicate_from_list(
            loop_condition_inputs_lambda
        )

        # Save the current state of the system so that it can used by a
        # nested loop to get information about the variables declared in its
        # outermost scopes.
        self.parent_loop_state = state

        # Define some condition and break variables in the loop state
        loop_state.last_definitions[index_name] = 0
        loop_state.last_definitions["IF_0"] = 0
        loop_state.last_definitions["EXIT"] = 0
        loop_state.variable_types["IF_0"] = "bool"
        loop_state.variable_types["EXIT"] = "bool"

        # Now, create the `variable` spec, `function name` and `container
        # wiring` for the loop index, check condition and break decisions.
        index_variable_grfn = self.generate_variable_definition(
            [index_name], None, False, loop_state
        )
        index_function_name = self.generate_function_name(
            "__assign__", index_variable_grfn["name"], None
        )
        index_function = {
            "function": index_function_name,
            "input": [],
            "output": [f"@variable::{index_name}::0"],
            "updated": [],
        }

        loop_check_variable = self.generate_variable_definition(
            ["IF_0"], None, False, loop_state
        )

        loop_state.next_definitions["#cond"] = 1
        loop_state.last_definitions["#cond"] = 0

        loop_check_function_name = self.generate_function_name(
            "__condition__", loop_check_variable["name"], None
        )
        loop_condition_function = {
            "function": loop_check_function_name,
            "input": loop_condition_inputs,
            "output": [f"@variable::IF_0::0"],
            "updated": [],
        }
        # Create the lambda function for the index variable initiations and
        # other loop checks. This has to be done through a custom lambda
        # function operation since the structure of genCode does not conform
        # with the way this lambda function will be created.
        # TODO Add code to support a step other than +1 as well
        assert len(range_call["inputs"]) <= 3, (
            f"Only two or three elements in range function supported as of now - {range_call}"
        )
        loop_start = range_call["inputs"][0]
        loop_end = range_call["inputs"][1]
        # TODO: need to decide what we'll do with step.
        if len(range_call["inputs"]) == 3:
            loop_step = range_call["inputs"][2]

        # TODO Add a separate function to get the variables/literals of a
        #  more complex form. The one below is for the basic case.
        if "var" in loop_start[0]:
            loop_start_name = loop_start[0]["var"]["variable"]
        elif "type" in loop_start[0] and loop_start[0]["type"] == "literal":
            loop_start_name = loop_start[0]["value"]
        else:
            assert False, "Error in getting loop start name"

        if "var" in loop_end[0]:
            loop_end_name = loop_end[0]["var"]["variable"]
        elif "type" in loop_end[0] and loop_end[0]["type"] == "literal":
            loop_end_name = loop_end[0]["value"]
        else:
            assert False, "Error in getting loop end name"
        # First, lambda function for loop index initiation
        index_initiation_lambda = self.generate_lambda_function(
            loop_start_name,
            index_function_name["name"],
            True,
            False,
            False,
            False,
            [],
            state,
            True,
        )
        loop_state.lambda_strings.append(index_initiation_lambda)

        # Second, lambda function for IF_0_0 test
        loop_test_string = f"0 <= {index_name} < {loop_end_name}"
        loop_continuation_test_lambda = self.generate_lambda_function(
            loop_test_string,
            loop_check_function_name["name"],
            True,
            False,
            False,
            False,
            loop_condition_inputs_lambda,
            state,
            True,
        )
        loop_state.lambda_strings.append(loop_continuation_test_lambda)

        # Parse through the body of the loop container
        loop = self.gen_grfn(node.body, loop_state, "for")
        # Separate the body grfn into `variables` and `functions` sub parts
        (
            body_variables_grfn,
            body_functions_grfn,
            body_container_grfn,
        ) = self._get_variables_and_functions(loop)

        # Get a list of all variables that were used as inputs within the
        # loop body (nested as well).
        loop_body_inputs = []

        # Get only the dictionaries
        body_functions_grfn = [
            item for item in body_functions_grfn if isinstance(item, dict)
        ]
        for function in body_functions_grfn:
            if function["function"]["type"] == "lambda":
                for ip in function["input"]:
                    (_, input_var, input_index) = ip.split("::")
                    if (
                        _ == "@variable"
                        and int(input_index) == -1
                        and input_var not in loop_body_inputs
                    ):
                        loop_body_inputs.append(input_var)
            elif function["function"]["type"] == "container":
                # The same code as above but separating it out just in case
                # some extra checks are added in the future
                for ip in function["input"]:
                    (_, input_var, input_index) = ip.split("::")
                    if _ == "@variable" and int(input_index) == -1:
                        loop_body_inputs.append(input_var)

        # Remove any duplicates since variables can be used multiple times in
        # various assignments within the body
        loop_body_inputs = self._remove_duplicate_from_list(loop_body_inputs)
        # If the index name is a part of the loop body, remove it since it is
        # not an input to the container
        if index_name in loop_body_inputs:
            loop_body_inputs.remove(index_name)

        # TODO: Not doing this right now. Refine this code and do it then.
        """
        # Now, we remove the variables which were defined inside the loop
        # body itself and not taken as an input from outside the loop body
        filtered_loop_body_inputs = []
        for input_var in loop_body_inputs:
            # We filter out those variables which have -1 index in `state` (
            # which means it did not have a defined value above the loop
            # body) and is not a function argument (since they have an index
            # of -1 as well but have a defined value)
            if not (state.last_definitions[input_var] == -1 and input_var
            not in
                    self.function_argument_map[main_function_name][
                        "argument_list"]
                    ):
                filtered_loop_body_inputs.append(input_var)

        """

        for item in loop_body_inputs:
            # TODO Hack for now, this should be filtered off from the code
            #  block above
            if (
                "IF" not in item
                and state.last_definitions.get(item) is not None
            ):
                function_input.append(
                    f"@variable::{item}::" f"{state.last_definitions[item]}"
                )
                container_argument.append(f"@variable::{item}::-1")

        function_input = self._remove_duplicate_from_list(function_input)
        container_argument = self._remove_duplicate_from_list(
            container_argument
        )

        # Creating variable specs for the inputs to the containers.
        start_definitions = loop_state.last_definitions.copy()
        container_definitions = start_definitions.copy()
        container_input_state = loop_state.copy(
            last_definitions=container_definitions
        )
        for argument in container_argument:
            (_, var, index) = argument.split("::")
            container_input_state.last_definitions[var] = int(index)
            argument_variable = self.generate_variable_definition(
                [var], None, False, container_input_state
            )
            body_variables_grfn.append(argument_variable)

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
        loop_body_outputs = {}
        for function in body_functions_grfn:
            if function["function"]["type"] == "lambda":
                # TODO Currently, we only deal with a single output variable.
                #  Modify the line above to not look at only [0] but loop
                #  through the output to incorporate multiple outputs
                (_, output_var, output_index) = function["output"][0].split(
                    "::"
                )
                loop_body_outputs[output_var] = output_index
            elif function["function"]["type"] == "container":
                for ip in function["updated"]:
                    if isinstance(ip, dict):
                        for x in ip["updates"]:
                            (_, output_var, output_index) = x.split("::")
                            loop_body_outputs[output_var] = output_index
                    else:
                        (_, output_var, output_index) = ip.split("::")
                        loop_body_outputs[output_var] = output_index

        for item in loop_body_outputs:
            # TODO the indexing variables in of function block and container
            #  block will be different. Figure about the differences and
            #  implement them.
            # TODO: Hack, this IF check should not even appear in
            #  loop_body_outputs
            if (
                "IF" not in item
                and "EXIT" not in item
                and state.last_definitions.get(item) is not None
            ):
                if state.last_definitions[item] == -2:
                    updated_index = 0
                else:
                    updated_index = state.last_definitions[item] + 1
                function_updated.append(
                    f"@variable::{item}::" f"{updated_index}"
                )
                state.last_definitions[item] = updated_index
                state.next_definitions[item] = updated_index + 1
                item_id = loop_state.last_definitions.get(
                    item, loop_body_outputs[item]
                )
                container_updated.append(f"@variable::{item}::" f"{item_id}")
                # Create variable spec for updated variables in parent scope.
                # So, temporarily change the current scope to its previous form
                tmp_scope = self.current_scope
                self.current_scope = ".".join(
                    self.current_scope.split(".")[:-1]
                )
                updated_variable = self.generate_variable_definition(
                    [item], None, False, state
                )
                body_variables_grfn.append(updated_variable)
                # Changing it back to its current form
                self.current_scope = tmp_scope

        loop_break_variable = self.generate_variable_definition(
            ["EXIT"], None, False, loop_state
        )
        loop_state.next_definitions["EXIT"] = 1

        loop_break_function_name = self.generate_function_name(
            "__decision__", loop_break_variable["name"], None
        )
        exit_inputs = ["@variable::IF_0::0"]
        exit_inputs.extend([f"@variable::{list(x.keys())[0]}::0" for x in
                            self.exit_candidates])
        lambda_inputs = [f"{x.split('::')[1]}_{x.split('::')[2]}" for x in
                         exit_inputs]

        loop_break_function = {
            "function": loop_break_function_name,
            "input": exit_inputs,
            "output": [f"@variable::EXIT::0"],
            "updated": [],
        }
        loop_exit_test_lambda = self.generate_lambda_function(
            ["EXIT"],
            loop_break_function_name["name"],
            True,
            False,
            False,
            False,
            lambda_inputs,
            state,
            True,
        )
        loop_state.lambda_strings.append(loop_exit_test_lambda)

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

        loop_functions_grfn += body_functions_grfn

        # Finally, add the index increment variable and function grfn to the
        # body grfn
        loop_state.last_definitions[index_name] = 1
        index_increment_grfn = self.generate_variable_definition(
            [index_name], None, False, loop_state
        )
        index_increment_function_name = self.generate_function_name(
            "__assign_", index_increment_grfn["name"], None
        )
        index_increment_function = {
            "function": index_increment_function_name,
            "input": [f"@variable::{index_name}::0"],
            "output": [f"@variable::{index_name}::1"],
            "updated": [],
        }

        # Finally, lambda function for loop index increment
        index_increment_lambda = self.generate_lambda_function(
            f"{index_name} + 1",
            index_increment_function_name["name"],
            True,
            False,
            False,
            False,
            [index_name],
            state,
            True,
        )
        loop_state.lambda_strings.append(index_increment_lambda)

        loop_variables_grfn.append(index_increment_grfn)
        loop_functions_grfn.append(index_increment_function)

        container_gensym = self.generate_gensym("container")

        loop_container = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "type": "loop",
            "arguments": container_argument,
            "updated": container_updated,
            "return_value": container_return_value,
            "body": loop_functions_grfn,
        }
        loop_function = {
            "function": {"name": container_id_name, "type": "container"},
            "input": function_input,
            "output": function_output,
            "updated": function_updated,
        }

        loop_container = [loop_container] + body_container_grfn
        loop_variables = body_variables_grfn + loop_variables_grfn
        grfn = {
            "containers": loop_container,
            "variables": loop_variables,
            "functions": [loop_function],
        }

        # Change the current scope back to its previous form.
        self.current_scope = ".".join(self.current_scope.split(".")[:-1])

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
        container_return_value = []
        container_updated = []
        function_output = []
        function_updated = []
        function_input = []
        loop_condition_inputs = []
        loop_condition_inputs_lambda = []
        loop_variables_grfn = []
        loop_functions_grfn = []

        # Increment the loop index universally across the program
        if self.loop_index > -1:
            self.loop_index += 1
        else:
            self.loop_index = 0

        # First, get the `container_id_name` of the loop container
        container_id_name = self.generate_container_id_name(
            self.fortran_file, self.current_scope, f"loop${self.loop_index}"
        )

        # Update the scope of the loop container so that everything inside
        # the body of the loop will have the below scope
        self.current_scope = f"{self.current_scope}.loop${self.loop_index}"

        loop_test = self.gen_grfn(node.test, state, "while")

        # Define a new empty state that will be used for mapping the state of
        # the operations within the loop container
        loop_last_definition = {}
        loop_state = state.copy(
            last_definitions=loop_last_definition,
            next_definitions={},
            last_definition_default=-1,
        )

        # We want the loop_state to have state information about variables
        # defined one scope above its current parent scope. The below code
        # allows us to do that
        if self.parent_loop_state:
            for var in self.parent_loop_state.last_definitions:
                if var not in state.last_definitions:
                    # state.last_definitions[var] = \
                    #     self.parent_loop_state.last_definitions[var]
                    state.last_definitions[var] = -1

        # Now populate the IF and EXIT functions for the loop by identifying
        # the loop conditionals
        # TODO Add a test to check for loop validity in this area. Need to
        #  test with more types of while loops to finalize on a test condition

        for item in loop_test:
            if not isinstance(item, list):
                item = [item]
            for var in item:
                if "var" in var:
                    function_input.append(
                        f"@variable::"
                        f"{var['var']['variable']}::"
                        f"{var['var']['index']}"
                    )
                    container_argument.append(
                        f"@variable::" f"{var['var']['variable']}::-1"
                    )
                    loop_condition_inputs.append(
                        f"@variable::" f"{var['var']['variable']}::-1"
                    )
                    loop_condition_inputs_lambda.append(var["var"]["variable"])
                elif "call" in var:
                    # TODO: Very specifically for arrays. Will probably break
                    #  for other calls
                    self._get_call_inputs(
                        var["call"],
                        function_input,
                        container_argument,
                        loop_condition_inputs,
                        loop_condition_inputs_lambda,
                        state,
                    )

        function_input = self._remove_duplicate_from_list(function_input)
        container_argument = self._remove_duplicate_from_list(
            container_argument
        )
        loop_condition_inputs = self._remove_duplicate_from_list(
            loop_condition_inputs
        )
        loop_condition_inputs_lambda = self._remove_duplicate_from_list(
            loop_condition_inputs_lambda
        )

        # Save the current state of the system so that it can used by a
        # nested loop to get information about the variables declared in its
        # outermost scopes.
        self.parent_loop_state = state

        # Define some condition and break variables in the loop state
        loop_state.last_definitions["IF_0"] = 0
        loop_state.last_definitions["EXIT"] = 0
        loop_state.variable_types["IF_0"] = "bool"
        loop_state.variable_types["EXIT"] = "bool"

        # Now, create the `variable` spec, `function name` and `container
        # wiring` for the check condition and break decisions.

        loop_check_variable = self.generate_variable_definition(
            ["IF_0"], None, False, loop_state
        )

        loop_state.next_definitions["#cond"] = 1
        loop_state.last_definitions["#cond"] = 0

        loop_check_function_name = self.generate_function_name(
            "__condition__", loop_check_variable["name"], None
        )
        loop_condition_function = {
            "function": loop_check_function_name,
            "input": loop_condition_inputs,
            "output": [f"@variable::IF_0::0"],
            "updated": [],
        }

        # Create the lambda function for the index variable initiations and
        # other loop checks. This has to be done through a custom lambda
        # function operation since the structure of genCode does not conform
        # with the way this lambda function will be created.
        # TODO Add a separate function to get the variables/literals of a
        #  more complex form. The one below is for the basic case.

        # Second, lambda function for IF_0_0 test
        loop_continuation_test_lambda = self.generate_lambda_function(
            node.test,
            loop_check_function_name["name"],
            True,
            False,
            False,
            False,
            loop_condition_inputs_lambda,
            state,
            True,
        )
        loop_state.lambda_strings.append(loop_continuation_test_lambda)

        # Parse through the body of the loop container
        loop = self.gen_grfn(node.body, loop_state, "for")
        # Separate the body grfn into `variables` and `functions` sub parts
        (
            body_variables_grfn,
            body_functions_grfn,
            body_container_grfn,
        ) = self._get_variables_and_functions(loop)

        # Get a list of all variables that were used as inputs within the
        # loop body (nested as well).
        loop_body_inputs = []

        # Get only the dictionaries
        body_functions_grfn = [
            item for item in body_functions_grfn if isinstance(item, dict)
        ]
        for function in body_functions_grfn:
            if function["function"]["type"] == "lambda":
                for ip in function["input"]:
                    (_, input_var, input_index) = ip.split("::")
                    if (
                        int(input_index) == -1
                        and input_var not in loop_body_inputs
                    ):
                        loop_body_inputs.append(input_var)
            elif function["function"]["type"] == "container":
                # The same code as above but separating it out just in case
                # some extra checks are added in the future
                for ip in function["input"]:
                    (_, input_var, input_index) = ip.split("::")
                    # TODO Hack for bypassing `boolean` types. Will be
                    #  removed once the `literal` as an input question is
                    #  answered.
                    if int(input_index) == -1 and input_var != "boolean":
                        loop_body_inputs.append(input_var)

        # Remove any duplicates since variables can be used multiple times in
        # various assignments within the body
        loop_body_inputs = self._remove_duplicate_from_list(loop_body_inputs)

        # TODO: Not doing this right now. Refine this code and do it then.
        """
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

        """

        # for item in filtered_loop_body_inputs:
        for item in loop_body_inputs:
            # TODO Hack for now, this should be filtered off from the code
            #  block above
            if (
                "IF" not in item
                and state.last_definitions.get(item) is not None
            ):
                function_input.append(
                    f"@variable::{item}::" f"{state.last_definitions[item]}"
                )
                container_argument.append(f"@variable::{item}::-1")

        function_input = self._remove_duplicate_from_list(function_input)
        container_argument = self._remove_duplicate_from_list(
            container_argument
        )

        # Creating variable specs for the inputs to the containers.
        start_definitions = loop_state.last_definitions.copy()
        container_definitions = start_definitions.copy()
        container_input_state = loop_state.copy(
            last_definitions=container_definitions
        )
        for argument in container_argument:
            (_, var, index) = argument.split("::")
            container_input_state.last_definitions[var] = int(index)
            argument_variable = self.generate_variable_definition(
                [var], None, False, container_input_state
            )
            body_variables_grfn.append(argument_variable)

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
        loop_body_outputs = {}
        for function in body_functions_grfn:
            if function["function"]["type"] == "lambda":
                # TODO Currently, we only deal with a single output variable.
                #  Modify the line above to not look at only [0] but loop
                #  through the output to incorporate multiple outputs
                (_, output_var, output_index) = function["output"][0].split(
                    "::"
                )
                loop_body_outputs[output_var] = output_index
            elif function["function"]["type"] == "container":
                for ip in function["updated"]:
                    if isinstance(ip, dict):
                        for x in ip["updates"]:
                            (_, output_var, output_index) = x.split("::")
                            loop_body_outputs[output_var] = output_index
                    else:
                        (_, output_var, output_index) = ip.split("::")
                        loop_body_outputs[output_var] = output_index

        for item in loop_body_outputs:
            # TODO the indexing variables in of function block and container
            #  block will be different. Figure about the differences and
            #  implement them.
            # TODO: Hack, this IF check should not even appear in
            #  loop_body_outputs
            if (
                "IF" not in item
                and "EXIT" not in item
                and state.last_definitions.get(item) is not None
            ):
                if (state.last_definitions[item] == -2) or (item == "EXIT"):
                    updated_index = 0
                else:
                    updated_index = state.last_definitions[item] + 1
                function_updated.append(
                    f"@variable::{item}::" f"{updated_index}"
                )
                state.last_definitions[item] = updated_index
                state.next_definitions[item] = updated_index + 1
                item_id = loop_state.last_definitions.get(
                    item, loop_body_outputs[item]
                )
                container_updated.append(f"@variable::{item}::" f"{item_id}")
                # Create variable spec for updated variables in parent scope.
                # So, temporarily change the current scope to its previous form
                tmp_scope = self.current_scope
                self.current_scope = ".".join(
                    self.current_scope.split(".")[:-1]
                )
                updated_variable = self.generate_variable_definition(
                    [item], None, False, state
                )
                body_variables_grfn.append(updated_variable)
                # Changing it back to its current form
                self.current_scope = tmp_scope

        loop_break_variable = self.generate_variable_definition(
            ["EXIT"], None, False, loop_state
        )
        loop_state.next_definitions["EXIT"] = 1

        loop_break_function_name = self.generate_function_name(
            "__decision__", loop_break_variable["name"], None
        )
        exit_inputs = ["@variable::IF_0::0"]
        exit_inputs.extend([f"@variable::{list(x.keys())[0]}::0" for x in
                            self.exit_candidates])
        lambda_inputs = [f"{x.split('::')[1]}_{x.split('::')[2]}" for x in
                         exit_inputs]

        loop_break_function = {
            "function": loop_break_function_name,
            "input": exit_inputs,
            "output": [f"@variable::EXIT::0"],
            "updated": [],
        }
        loop_exit_test_lambda = self.generate_lambda_function(
            ["EXIT"],
            loop_break_function_name["name"],
            True,
            False,
            False,
            False,
            lambda_inputs,
            state,
            True,
        )
        loop_state.lambda_strings.append(loop_exit_test_lambda)

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

        loop_functions_grfn += body_functions_grfn

        container_gensym = self.generate_gensym("container")

        loop_container = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "type": "loop",
            "arguments": container_argument,
            "updated": container_updated,
            "return_value": container_return_value,
            "body": loop_functions_grfn,
        }
        loop_function = {
            "function": {"name": container_id_name, "type": "container"},
            "input": function_input,
            "output": function_output,
            "updated": function_updated,
        }
        loop_container = [loop_container] + body_container_grfn
        loop_variables = body_variables_grfn + loop_variables_grfn
        grfn = {
            "containers": loop_container,
            "variables": loop_variables,
            "functions": [loop_function],
        }
        self.current_scope = ".".join(self.current_scope.split(".")[:-1])

        return [grfn]

    def process_if(self, node, state, call_source):
        """
            This function handles the ast.IF node of the AST. It goes through
            the IF body and generates the `decision` and `condition` type of
            the `<function_assign_def>`.
        """

        # Update the scope path
        scope_path = state.scope_path.copy()
        if len(scope_path) == 0:
            scope_path.append("@global")
        state.scope_path = scope_path

        # First, check whether this is the start of an `if-block` or an
        # `elif` block
        if call_source == "if":
            is_else_if = True
        else:
            is_else_if = False
            # Increment the loop index universally across the program
            if self.if_index > -1:
                self.if_index += 1
            else:
                self.if_index = 0
            # Define a new empty state that will be used for mapping the
            # state of the operations within the for-loop container
            loop_last_definition = {}
            if_state = state.copy(
                last_definitions=loop_last_definition,
                next_definitions={},
                last_definition_default=-1,
            )
            # First, get the `container_id_name` of the if container
            container_id_name = self.generate_container_id_name(
                self.fortran_file, self.current_scope, f"IF_{self.if_index}"
            )
            # Update the scope to include the new `if-block`
            self.current_scope = f"{self.current_scope}.IF_{self.if_index}"

            # We want the if_state to have state information about variables
            # defined one scope above its current parent scope. The below code
            # allows us to do that
            # TODO: How relevant is this for `if-blocks`? Will need to verify
            if self.parent_if_state:
                for var in self.parent_if_state.last_definitions:
                    if var not in state.last_definitions:
                        state.last_definitions[var] = -1

        grfn = {"functions": [], "variables": [], "containers": []}

        # Get the GrFN schema of the test condition of the `IF` command
        # Notice that we'll have two different states depending on whether
        # this is the main `if-block` or the `elif` block
        if is_else_if:
            condition_sources = self.gen_grfn(node.test, state, "if")
            condition_variables = self.get_variables(condition_sources, state)
        else:
            condition_sources = self.gen_grfn(node.test, if_state, "if")
            condition_variables = self.get_variables(condition_sources,
                                                     if_state)

        # When opening files, if a check for a pre-existing file has to be
        # done, this if-block is bypassed
        if isinstance(condition_sources[0], dict) and \
                condition_sources[0].get('call'):
            if condition_sources[0]["call"].get("function") and \
                    condition_sources[0]["call"]["function"] == "path.exists":
                return []

        # The index of the IF_x_x variable will start from 0
        if state.last_definition_default in (-1, 0, -2):
            # default_if_index = state.last_definition_default + 1
            default_if_index = 0
        else:
            assert False, (
                f"Invalid value of last_definition_default:"
                f"{state.last_definition_default}"
            )

        # For every new `if-block`, the IF_X value will increase by one.
        # For every condition check within an `if-block` (i.e. if .. elif ..
        # elif .. else ..), the COND_X index will start from 0 for the `if`
        # check and increment by 1 for every `elif` check
        if not is_else_if:
            condition_number = 0
            if_state.next_definitions["#cond"] = condition_number + 1
            condition_name = f"COND_{condition_number}"
            condition_index = self.get_last_definition(
                condition_name, if_state.last_definitions, 0
            )
            if_state.variable_types[condition_name] = "bool"
            if_state.last_definitions[condition_name] = condition_index
            variable_spec = self.generate_variable_definition(
                [condition_name], None, False, if_state
            )
        else:
            condition_number = self._get_next_definition(
                "#cond", state.last_definitions, state.next_definitions,
                default_if_index-1
            )
            condition_name = f"COND_{condition_number}"
            condition_index = self.get_last_definition(
                condition_name, state.last_definitions, 0
            )
            state.variable_types[condition_name] = "bool"
            state.last_definitions[condition_name] = condition_index
            variable_spec = self.generate_variable_definition(
                [condition_name], None, False, state
            )

        function_name = self.generate_function_name(
            "__condition__", variable_spec["name"], None
        )
        # Getting the output variable
        output_regex = re.compile(r".*::(?P<output>.*?)::(?P<index>.*$)")
        output_match = output_regex.match(variable_spec["name"])
        if output_match:
            output = output_match.group("output")
            index = output_match.group("index")
            output_variable = f"@variable::{output}::{index}"
        else:
            assert False, (
                f"Could not match output variable for "
                f"{variable_spec['name']}"
            )

        # Create the function statement for the COND_X variables
        fn = {
            "function": function_name,
            "input": [
                f"@variable::{src['var']['variable']}::{src['var']['index']}"
                for src in condition_variables
                if "var" in src
            ],
            "output": [output_variable],
            "updated": [],
        }

        # Save the current state of the system so that it can used by a
        # nested loop to get information about the variables declared in its
        # outermost scopes.
        self.parent_if_state = state

        # Update the variable definition with the COND_X variable spec
        grfn["variables"].append(variable_spec)

        # Generate the lambda function code of the COND_X check
        lambda_string = self.generate_lambda_function(
            node.test,
            function_name["name"],
            False,
            False,
            False,
            False,
            [
                src["var"]["variable"]
                for src in condition_variables
                if "var" in src
            ],
            state,
            False,
        )
        if not is_else_if:
            if_state.lambda_strings.append(lambda_string)
        else:
            state.lambda_strings.append(lambda_string)

        if is_else_if:
            if_grfn = self.gen_grfn(node.body, state, "if")
            else_grfn = self.gen_grfn(node.orelse, state, "if")
        else:
            if_grfn = self.gen_grfn(node.body, if_state, "if")
            else_grfn = self.gen_grfn(node.orelse, if_state, "if")

        # Sometimes, some if-else body blocks only contain I/O operations,
        # the GrFN for which will be empty. Check for these and handle
        # accordingly
        if (
            len(else_grfn) > 0
            and len(else_grfn[0]["functions"]) == 0
            and len(else_grfn[0]["variables"]) == 0
            and len(else_grfn[0]["containers"]) == 0
        ):
            else_grfn = []

        # If this is an `elif` block, append the if-body GrFN and else-body
        # GrFN to the functions with their respective structure and simply
        # return
        if is_else_if:
            grfn["functions"] = [{"condition": [fn], "statements": []}]
            for spec in if_grfn:
                grfn["functions"][0]["statements"] += spec["functions"]
                grfn["variables"] += spec["variables"]
                grfn["containers"] += spec["containers"]

            grfn["functions"].append({"condition": None, "statements": []})
            for spec in else_grfn:
                # Check for cases like more than one `elif` where the overall
                # structure has to stay consistent
                if "condition" in spec["functions"][0]:
                    grfn["functions"].pop()
                    grfn["functions"] += spec["functions"]
                else:
                    grfn["functions"][1]["statements"] += spec["functions"]
                grfn["variables"] += spec["variables"]
                grfn["containers"] += spec["containers"]

            return [grfn]

        # All of the operations from this point is only for the main
        # `if-block` if statement

        for spec in if_grfn:
            grfn["variables"] += spec["variables"]
            grfn["containers"] += spec["containers"]
        for spec in else_grfn:
            grfn["variables"] += spec["variables"]
            grfn["containers"] += spec["containers"]

        container_argument = []
        function_input = []
        if_body_inputs = []
        if_body_outputs = {}
        container_updated = []
        function_updated = []

        container_body = [{"condition": [fn], "statements": []}]
        for spec in if_grfn:
            container_body[0]["statements"] += spec["functions"]
        container_body.append({"condition": None, "statements": []})
        for spec in else_grfn:
            if "condition" in spec["functions"][0]:
                container_body.pop()
                container_body += spec["functions"]
            else:
                container_body[1]["statements"] += spec["functions"]

        # If there is no else statement, remove it from the block
        if container_body[-1]["condition"] is None and \
                len(container_body[-1]["statements"]) == 0:
            container_body.pop()

        current_condition = None
        for body_dict in container_body:
            for key in body_dict:
                if body_dict[key]:
                    for function in body_dict[key]:
                        if key == "condition":
                            current_condition = function["output"][
                                0].split("::")[1]
                            if_body_outputs[current_condition] = dict()
                        if function["function"]["type"] == "lambda":
                            for ip in function["input"]:
                                (_, input_var, input_index) = ip.split("::")
                                if (
                                        int(input_index) == -1
                                        and input_var not in if_body_inputs
                                ):
                                    if_body_inputs.append(input_var)
                            (_, output_var, output_index) = function["output"][
                                0].split("::")
                            if_body_outputs[current_condition][output_var] = \
                                output_index
                        elif function["function"]["type"] == "container":
                            # The same code as above but separating it out just
                            # in case some extra checks are added in the future
                            for ip in function["input"]:
                                (_, input_var, input_index) = ip.split("::")
                                if int(input_index) == -1:
                                    if_body_inputs.append(input_var)
                            for ip in function["updated"]:
                                if isinstance(ip, dict):
                                    for x in ip["updates"]:
                                        (_, output_var, output_index) = x.split(
                                            "::")
                                        if_body_outputs[output_var] = \
                                            output_index
                                else:
                                    (_, output_var, output_index) = ip.split(
                                        "::")
                                    if_body_outputs[output_var] = output_index
                else:
                    current_condition = None
                    if_body_outputs[current_condition] = dict()

        # Remove any duplicates since variables can be used multiple times in
        # various assignments within the body
        if_body_inputs = self._remove_duplicate_from_list(if_body_inputs)

        for item in if_body_inputs:
            if (
                "COND" not in item
                and state.last_definitions.get(item) is not None
            ):
                function_input.append(
                    f"@variable::{item}::" f"{state.last_definitions[item]}"
                )
                container_argument.append(f"@variable::{item}::-1")

        function_input = self._remove_duplicate_from_list(function_input)
        container_argument = self._remove_duplicate_from_list(
            container_argument
        )
        # Creating variable specs for the inputs to the containers.
        start_definitions = if_state.last_definitions.copy()
        container_definitions = start_definitions.copy()
        container_input_state = if_state.copy(
            last_definitions=container_definitions
        )
        for argument in container_argument:
            (_, var, index) = argument.split("::")
            container_input_state.last_definitions[var] = int(index)
            argument_variable = self.generate_variable_definition(
                [var], None, False, container_input_state
            )
            grfn["variables"].append(argument_variable)

        updated_vars = [list(if_body_outputs[x].keys()) for x in
                        if_body_outputs]
        updated_vars = list(set([i for x in updated_vars for i in x]))

        for index, item in enumerate(if_body_outputs):
            function_updated.append({
                "condition": item,
                "updates": []
            })
            container_updated.append({
                "condition": item,
                "updates": []
            })
            for var in if_body_outputs[item]:
                if (
                    "COND" not in var
                ):
                    if state.last_definitions.get(var, -2) == -2:
                        updated_index = 0
                    else:
                        if var in updated_vars:
                            updated_index = state.last_definitions[var] + 1
                        else:
                            updated_index = state.last_definitions[var]
                    function_updated[index]["updates"].append(
                        f"@variable::{var}::" f"{updated_index}"
                    )
                    if var in updated_vars:
                        state.last_definitions[var] = updated_index
                        state.next_definitions[var] = updated_index + 1
                        # Create variable spec for updated variables in parent
                        # scope. So, temporarily change the current scope to its
                        # previous form.
                        tmp_scope = self.current_scope
                        self.current_scope = ".".join(
                            self.current_scope.split(".")[:-1]
                        )
                        updated_variable = self.generate_variable_definition(
                            [var], None, False, state
                        )
                        grfn["variables"].append(updated_variable)
                        # Changing it back to its current form
                        self.current_scope = tmp_scope
                        updated_vars.remove(var)

                    item_id = if_body_outputs[item][var]
                    container_updated[index]["updates"].append(f"@variable:"
                                                               f":{var}::" 
                                                               f"{item_id}")

            # If there is no else statement, remove it from the block
            if container_updated[-1]["condition"] is None and \
                    len(container_updated[-1]["updates"]) == 0:
                container_updated.pop()
            if function_updated[-1]["condition"] is None and \
                    len(function_updated[-1]["updates"]) == 0:
                function_updated.pop()

        container_gensym = self.generate_gensym("container")

        if_type = "if-block"
        node_lineno = node.lineno
        for comment in self.comments:
            if comment == "# select-case" and \
                    node_lineno == int(self.comments[comment]) + 1:
                if_type = "select-block"

        if_container = {
            "name": container_id_name,
            "source_refs": [],
            "gensym": container_gensym,
            "type": if_type,
            "arguments": container_argument,
            "updated": container_updated,
            "return_value": [],
            "body": container_body,
        }
        if_function = {
            "function": {"name": container_id_name, "type": "container"},
            "input": function_input,
            "output": [],
            "updated": function_updated,
        }

        grfn["functions"].append(if_function)
        grfn["containers"] = [if_container] + grfn["containers"]

        # Change the current scope back to its previous form.
        self.current_scope = ".".join(self.current_scope.split(".")[:-1])
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
            for op in self.binops:
                if isinstance(node.op, op):
                    val = self.binops[type(node.op)](node.left.n, node.right.n)
                    data_type = self.annotate_map.get(type(val).__name__)
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
            assert False, (
                "Both operands are numbers but no operator "
                "available to handle their computation. Either add "
                "a handler if possible or remove this assert and "
                "allow the code below to handle such cases."
            )

        # If the operands are anything other than numbers (ast.Str,
        # ast.BinOp, etc), call `gen_grfn` on each side so their respective
        # ast handlers will process them and return a [{grfn_spec}, ..] form
        # for each side. Add these two sides together to give a single [{
        # grfn_spec}, ...] form.
        operation_grfn = self.gen_grfn(
            node.left, state, "binop"
        ) + self.gen_grfn(node.right, state, "binop")

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
        grfn = {"functions": [], "variables": [], "containers": []}

        for expr in expressions:
            if "call" not in expr:
                # assert False, f"Unsupported expr: {expr}."
                return []
        for expr in expressions:
            array_set = False
            string_set = False
            container_id_name = None
            input_index = None
            output_index = None
            arr_index = None
            call = expr["call"]
            function_name = call["function"]
            io_match = self.check_io_variables(function_name)
            if io_match:
                return []
            # Bypassing calls to `print` for now. Need further discussion and
            # decisions to move forward with what we'll do with `print`
            # statements.
            if function_name == "print":
                return []

            # A handler for array and string <.set_>/<.set_substr> function
            if ".set_" in function_name:
                split_function = function_name.split(".")
                is_derived_type_array_ref = False
                if (
                    len(split_function) > 2
                    and split_function[0] == self.current_d_object_name
                ):
                    d_object = split_function[0]
                    call_var = split_function[1]
                    method = split_function[2]
                    is_derived_type_array_ref = True
                else:
                    call_var = split_function[0]
                    method = split_function[1]
                # This is an array
                if len(call["inputs"]) > 1 and method != "set_substr":
                    array_set = True
                    function_name = function_name.replace(".set_", "")
                    for idx in range(0, len(split_function) - 1):
                        input_index = self.get_last_definition(
                            split_function[idx],
                            state.last_definitions,
                            state.last_definition_default,
                        )
                        output_index = self._get_next_definition(
                            split_function[idx],
                            state.last_definitions,
                            state.next_definitions,
                            state.last_definition_default,
                        )
                    if is_derived_type_array_ref:
                        function_name = function_name.replace(".", "_")
                        input_index = self.get_last_definition(
                            function_name,
                            state.last_definitions,
                            state.last_definition_default,
                        )
                        output_index = self._get_next_definition(
                            function_name,
                            state.last_definitions,
                            state.next_definitions,
                            state.last_definition_default,
                        )
                        state.variable_types[
                            function_name
                        ] = state.variable_types[call_var]
                        self.arrays[function_name] = self.arrays[call_var]

                    arr_index = self._generate_array_index(node)
                    str_arr_index = ""
                    str_arr_for_varname = ""
                    for idx in arr_index:
                        if function_name not in self.md_array:
                            str_arr_index += str(idx)
                            str_arr_for_varname += str(idx)
                        else:
                            str_arr_index += f"[{idx}]"
                            str_arr_for_varname += f"{idx}"
                    # arr_index = call["inputs"][0][0]["var"]["variable"]
                    # Create a new variable spec for indexed array. Ex.
                    # arr(i) will be arr_i. This will be added as a new
                    # variable in GrFN.
                    variable_spec = self.generate_variable_definition(
                        [function_name], str_arr_for_varname, False, state
                    )
                    grfn["variables"].append(variable_spec)
                    if function_name not in self.md_array:
                        state.array_assign_name = (
                            f"{function_name}[{str_arr_index}]"
                        )
                    else:
                        state.array_assign_name = (
                            f"{function_name}{str_arr_index}"
                        )
                    # We want to have a new variable spec for the original
                    # array (arr(i), for example) and generate the function
                    # name with it.
                    variable_spec = self.generate_variable_definition(
                        [function_name], None, False, state
                    )
                    assign_function = self.generate_function_name(
                        "__assign__", variable_spec["name"], arr_index
                    )
                    container_id_name = assign_function["name"]
                    function_type = assign_function["type"]
                # This is a string
                else:
                    string_set = True
                    function_name = call_var
                    state.string_assign_name = function_name
                    input_index = self._get_next_definition(
                        function_name,
                        state.last_definitions,
                        state.next_definitions,
                        -1,
                    )
                    variable_spec = self.generate_variable_definition(
                        [function_name], None, False, state
                    )
                    grfn["variables"].append(variable_spec)
                    assign_function = self.generate_function_name(
                        "__assign__", variable_spec["name"], None
                    )
                    container_id_name = assign_function["name"]
                    function_type = assign_function["type"]
            else:
                if function_name in self.function_argument_map:
                    container_id_name = self.function_argument_map[
                        function_name
                    ]["name"]
                elif function_name in self.module_subprograms:
                    container_id_name = function_name
                function_type = "container"

            function = {
                "function": {"name": container_id_name, "type": function_type},
                "input": [],
                "output": [],
                "updated": [],
            }

            # Array itself needs to be added as an input, so check that it's
            # and array. If yes, then add it manually.
            if array_set:
                function["input"].append(
                    f"@variable::" f"{function_name}::{input_index}"
                )
                function["output"] = [
                    f"@variable::" f"{function_name}::{output_index}"
                ]

            argument_list = []
            list_index = 0
            for arg in call["inputs"]:
                # We handle inputs to strings differently, so break out of
                # this loop
                if string_set:
                    break
                generate_lambda_for_arr = False
                if len(arg) == 1:
                    # TODO: Only variables are represented in function
                    #  arguments. But a function can have strings as
                    #  arguments as well. Do we add that?
                    if "var" in arg[0]:
                        if arg[0]["var"]["variable"] not in argument_list:
                            function["input"].append(
                                f"@variable::"
                                f"{arg[0]['var']['variable']}::"
                                f"{arg[0]['var']['index']}"
                            )
                        # This is a case where a variable gets assigned to
                        # an array. For example, arr(i) = var.
                        if array_set:
                            argument_list.append(arg[0]["var"]["variable"])
                            # If list_index is 0, it means that the current
                            # loop is dealing with the array index (i in arr(
                            # i)), which we do not wish to generate a lambda
                            # function for. list_index > 0 are the RHS values
                            # for array assignment.
                            if list_index > 0:
                                generate_lambda_for_arr = True
                            list_index += 1
                    # This is a case where either an expression or an array
                    # gets assigned to an array. For example, arr(i) =
                    # __expression__ or arr(i) = arr2(i).
                    elif "call" in arg[0]:
                        # Check if a function argument is a string value
                        # E.g: GET("test", "this", x)
                        if arg[0]["call"].get("function"):
                            if arg[0]["call"]["function"] == "String":
                                # The value can either be a string literal or
                                # a substring of another string. Check for this
                                if "value" in arg[0]["call"]["inputs"][1][0]:
                                    value = arg[0]["call"]["inputs"][1][0][
                                        "value"]
                                    function["input"].append(
                                        f"@literal::" f"string::" f"'{value}'"
                                    )
                                elif "call" in arg[0]["call"]["inputs"][1][0]:
                                    string_variable = arg[0]["call"][
                                        "inputs"][1][0]["call"]["function"]
                                    if ".get_substr" in string_variable:
                                        string_variable = \
                                            string_variable.split('.')[0]
                                        string_index = \
                                            state.last_definitions[
                                                string_variable]
                                        function["input"].append(
                                            f"@variable::{string_variable}::"
                                            f"{string_index}"
                                        )
                                    else:
                                        assert False, "Unidentified " \
                                                      "expression in String."
                                else:
                                    assert False, \
                                        "Unidentified expression in String."
                        else:
                            function = self._generate_array_setter(
                                node, function, arg,
                                function_name, container_id_name,
                                arr_index, state)
                    # This is a case where a literal gets assigned to an array.
                    # For example, arr(i) = 100.
                    elif "type" in arg[0] and array_set:
                        generate_lambda_for_arr = True

                    if generate_lambda_for_arr and state.array_assign_name:
                        argument_list.append(function_name)
                        lambda_string = self.generate_lambda_function(
                            node,
                            container_id_name,
                            True,
                            True,
                            False,
                            False,
                            argument_list,
                            state,
                            False,
                        )
                        state.lambda_strings.append(lambda_string)
                else:
                    if function_name in self.arrays:
                        # If array type is <float> the argument holder
                        # has a different structure that it does not hold
                        # function info. like when an array is 'int' type
                        # [{'call': {'function': '_type_', 'inputs': [...]]
                        # which causes an error. Thus, the code below fixes
                        # by correctly structuring it.
                        array_type = self.arrays[function_name]["elem_type"]
                        fixed_arg = [
                            {"call": {"function": array_type, "inputs": [arg]}}
                        ]
                        function = self._generate_array_setter(
                            node,
                            function,
                            fixed_arg,
                            function_name,
                            container_id_name,
                            arr_index,
                            state,
                        )
                    else:
                        assert (
                            "call" in arg[0]
                        ), "Only 1 input per argument supported right now."

            # Below is a separate loop just for filling in inputs for arrays
            if array_set:
                argument_list = []
                need_lambdas = False
                if "set_" in call["function"]:
                    argument_list.append(call["function"].replace(".set_", ""))
                for arg in call["inputs"]:
                    for ip in arg:
                        if "var" in ip:
                            function["input"].append(
                                f"@variable::"
                                f"{ip['var']['variable']}::"
                                f"{ip['var']['index']}"
                            )
                            if ip["var"]["variable"] not in argument_list:
                                argument_list.append(ip["var"]["variable"])
                        elif "call" in ip:
                            function_call = ip["call"]
                            function_name = function_call["function"]
                            need_lambdas = True
                            if ".get_" in function_name:
                                if ".get_substr" in function_name:
                                    function_name = function_name.replace(
                                        ".get_substr", ""
                                    )
                                else:
                                    function_name = function_name.replace(
                                        ".get_", ""
                                    )
                                # In some cases, the target array itself will
                                # be an input as well. Don't add such arrays
                                # again.
                                if True not in [
                                    function_name in i
                                    for i in function["input"]
                                ]:
                                    function["input"].append(
                                        f"@variable::"
                                        f"{function_name}::"
                                        f"{state.last_definitions[function_name]}"
                                    )
                                if function_name not in argument_list:
                                    argument_list.append(function_name)
                            for call_input in function_call["inputs"]:
                                # TODO: This is of a recursive nature. Make
                                #  this a loop. Works for SIR for now.
                                for var in call_input:
                                    if "var" in var:
                                        function["input"].append(
                                            f"@variable::"
                                            f"{var['var']['variable']}::"
                                            f"{var['var']['index']}"
                                        )
                                        if (
                                            var["var"]["variable"]
                                            not in argument_list
                                        ):
                                            argument_list.append(
                                                var["var"]["variable"]
                                            )

                function["input"] = self._remove_duplicate_from_list(
                    function["input"]
                )
                argument_list = self._remove_duplicate_from_list(argument_list)

                if (
                    need_lambdas
                    and container_id_name
                    not in self.generated_lambda_functions
                ):
                    lambda_string = self.generate_lambda_function(
                        node,
                        container_id_name,
                        True,
                        True,
                        False,
                        False,
                        argument_list,
                        state,
                        False,
                    )
                    state.lambda_strings.append(lambda_string)
                    need_lambdas = False

            # Make an assign function for a string .set_ operation
            if string_set:
                target = {
                    "var": {
                        "variable": function_name,
                        "index": variable_spec["name"].split("::")[-1],
                    }
                }

                source_list = list(chain.from_iterable(call["inputs"]))

                # If the function is `set_substr`, the target string will
                # also be an input.
                if method == "set_substr":
                    source_list.append(
                        {
                            "var": {
                                "variable": function_name,
                                "index": int(
                                    variable_spec["name"].split("::")[-1]
                                )
                                - 1,
                            }
                        }
                    )
                function = self.make_fn_dict(
                    assign_function, target, source_list, state
                )

                argument_list = self.make_source_list_dict(source_list)

                lambda_string = self.generate_lambda_function(
                    node,
                    container_id_name,
                    True,
                    False,
                    True,
                    False,
                    argument_list,
                    state,
                    False,
                )
                state.lambda_strings.append(lambda_string)

            # This is sort of a hack for SIR to get the updated fields filled
            # in beforehand. For a generalized approach, look at
            # `process_module`.
            if function["function"]["type"] == "container":
                for functions in self.function_argument_map:
                    if (
                        self.function_argument_map[functions]["name"]
                        == function["function"]["name"]
                    ):

                        for var in function["input"]:
                            input_var = var.rsplit("::")
                            index = input_var[2]
                            if (
                                input_var[1] in self.f_array_arg
                                or var
                                in self.function_argument_map[functions][
                                    "updated_list"
                                ]
                            ):
                                variable_name = input_var[1]
                                function["updated"].append(
                                    f"@variable::{variable_name}::{int(index)+1}"
                                )

                                state.last_definitions[variable_name] += 1
                                state.next_definitions[variable_name] = (
                                    state.last_definitions[variable_name] + 1
                                )

                                variable_spec = \
                                    self.generate_variable_definition(
                                     [variable_name], None, False, state
                                    )
                                grfn["variables"].append(variable_spec)
            # Keep a track of all functions whose `update` might need to be
            # later updated, along with their scope.
            if len(function["input"]) > 0:
                # self.update_functions.append(function_name)
                self.update_functions[function_name] = {
                    "scope": self.current_scope,
                    "state": state,
                }
            grfn["functions"].append(function)
        return [grfn]

    def process_compare(self, node, state, *_):
        """
            This function handles ast.Compare i.e. the comparator tag which
            appears on logical comparison i.e. ==, <, >, <=, etc. This
            generally occurs within an `if` statement but can occur elsewhere
            as well.
        """
        return self.gen_grfn(node.left, state, "compare") + self.gen_grfn(
            node.comparators, state, "compare"
        )

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
            # raise For2PyError("can't handle arrays right now.")
            pass

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
        if not re.match(r"i_g_n_o_r_e___m_e__.*", node.id):
            last_definition = self.get_last_definition(
                node.id, state.last_definitions, state.last_definition_default
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
        if (
            len(sources) == 1
            and ("value" in sources[0][0].keys())
            and sources[0][0]["value"] is None
        ):
            for target in targets:
                state.variable_types[
                    target["var"]["variable"]
                ] = self.get_variable_type(node.annotation)
                if target["var"]["variable"] not in self.annotated_assigned:
                    self.annotated_assigned.append(target["var"]["variable"])

                # Check if these variables are io variables. We don't maintain
                # states for io variables. So, don't add them on the
                # last_definitions and next_definitions dict.
                io_match = self.check_io_variables(target["var"]["variable"])
                if io_match:
                    self.exclude_list.append(target["var"]["variable"])
                # When a variable is AnnAssigned with [None] it is being
                # declared but not defined. So, it should not be assigned an
                # index. Having the index as -2 indicates that this variable
                # has only been declared but never defined. The next
                # definition of this variable should start with an index of 0
                # though.
                if not io_match:
                    target["var"]["index"] = -2
                    state.last_definitions[target["var"]["variable"]] = -2
                    state.next_definitions[target["var"]["variable"]] = 0
            return []

        grfn = {"functions": [], "variables": [], "containers": []}

        # Only a single target appears in the current version. The `for` loop
        # seems unnecessary but will be required when multiple targets start
        # appearing (e.g. a = b = 5).
        for target in targets:
            target_name = target["var"]["variable"]
            # Because we use the `last_definition_default` to be -1 for
            # functions with arguments, in these functions the annotated
            # assigns at the top of the function will get the -1 index which
            # is incorrect.
            if target["var"]["index"] == -1:
                target["var"]["index"] = 0
                state.last_definitions[target_name] = 0
            # Preprocessing and removing certain Assigns which only pertain
            # to the Python code and do not relate to the FORTRAN code in any
            # way.
            io_match = self.check_io_variables(target_name)
            if io_match:
                self.exclude_list.append(target_name)
                return []
            state.variable_types[target_name] = self.get_variable_type(
                node.annotation
            )
            if target_name not in self.annotated_assigned:
                self.annotated_assigned.append(target_name)
            # Update the `next_definition` index of the target since it is
            # not being explicitly done by `process_name`.
            # TODO Change this functionality ground up by modifying
            #  `process_name` and `process_subscript` to make it simpler.
            if not state.next_definitions.get(target_name):
                state.next_definitions[target_name] = (
                    target["var"]["index"] + 1
                )
            variable_spec = self.generate_variable_definition(
                [target_name], None, False, state
            )
            function_name = self.generate_function_name(
                "__assign__", variable_spec["name"], None
            )

            # If the source is a list inside of a list, remove the outer list
            if len(sources) == 1 and isinstance(sources[0], list):
                sources = sources[0]

            # TODO Somewhere around here, the Float32 class problem will have
            #  to be handled.
            fn = self.make_fn_dict(function_name, target, sources, state)

            if len(sources) > 0:
                lambda_string = self.generate_lambda_function(
                    node,
                    function_name["name"],
                    False,
                    True,
                    False,
                    False,
                    [
                        src["var"]["variable"]
                        for src in sources
                        if "var" in src
                    ],
                    state,
                    False,
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
        io_source = False
        is_function_call = False
        maybe_d_type_object_assign = False
        d_type_object_name = None
        # Get the GrFN element of the RHS side of the assignment which are
        # the variables involved in the assignment operations.
        sources = self.gen_grfn(node.value, state, "assign")
        node_name = node.targets[0].__repr__().split()[0][2:]
        if node_name == "ast.Attribute":
            node_value = node.targets[0].value
            attrib_ast = node_value.__repr__().split()[0][2:]
            if (
                attrib_ast == "ast.Name"
                and node_value.id in self.derived_type_objects
            ):
                maybe_d_type_object_assign = True
                d_type_object_name = node_value.id
                object_type = self.derived_type_objects[d_type_object_name]
            elif (
                attrib_ast == "ast.Attribute"
                and node_value.value.id in self.derived_type_objects
            ):
                maybe_d_type_object_assign = True
                d_type_object_name = node_value.value.id
                object_type = self.derived_type_objects[d_type_object_name]

        array_assignment = False
        is_d_type_obj_declaration = False
        # Detect assigns which are string initializations of the
        # following form: String(10). String initialization of the form
        # String(10, "abcdef") are valid assignments where the index of the
        # variables will be incremented but for the former case the index
        # will not be incremented and neither will its variable spec be
        # generated
        is_string_assign = False
        is_string_annotation = False
        if len(sources) > 0 and "call" in sources[0]:
            type_name = sources[0]["call"]["function"]
            if type_name == "String":
                is_string_assign = True
                # Check if it just an object initialization or initialization
                # with value assignment
                if len(sources[0]["call"]["inputs"]) == 1:
                    # This is just an object initialization e.g. String(10)
                    is_string_annotation = True
            elif type_name == "Array":
                array_assignment = True
                array_dimensions = []
                inputs = sources[0]["call"]["inputs"]

                # If the array type is string, the structure of inputs will
                # be a bit different than when it is int of float
                if "call" in inputs[0][0]:
                    if inputs[0][0]["call"]["function"] == "String":
                        array_type = "string"
                else:
                    array_type = inputs[0][0]["var"]["variable"]
                self._get_array_dimension(sources, array_dimensions, inputs)
            elif type_name in self.derived_types:
                is_d_type_obj_declaration = True
                if isinstance(node.targets[0], ast.Name):
                    variable_name = node.targets[0].id
                    if variable_name not in self.module_variable_types:
                        for program in self.mode_mapper["public_objects"]:
                            if (
                                variable_name
                                in self.mode_mapper["public_objects"][program]
                            ):
                                self.module_variable_types[variable_name] = [
                                    program,
                                    type_name,
                                ]
            else:
                pass
        else:
            pass

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
        target_names = []
        object_attr_num = 1
        for target in targets:
            # Bypass any assigns that have multiple targets.
            # E.g. (i[0], x[0], j[0], y[0],) = ...
            if "list" in target:
                return []
            target_names.append(target["var"]["variable"])
            # Fill some data structures if this is a string
            # assignment/initialization
            if is_string_assign:
                state.variable_types[target_names[0]] = "string"
                state.string_assign_name = target_names[0]
                self.strings[target_names[0]] = {
                    "length": sources[0]["call"]["inputs"][0][0]["value"]
                }
                if is_string_annotation:
                    # If this is just a string initialization,
                    # last_definition should not contain this string's index.
                    # This happens only during assignments.
                    del state.last_definitions[target_names[0]]
                    self.strings[target_names[0]]["annotation"] = True
                    self.strings[target_names[0]]["annotation_assign"] = False
                    return []
                else:
                    self.strings[target_names[0]]["annotation"] = False
                    self.strings[target_names[0]]["annotation_assign"] = True

            # Pre-processing and removing certain Assigns which only pertain
            # to the Python code and do not relate to the FORTRAN code in any
            # way.
            io_match = self.check_io_variables(target_names[0])
            if io_match:
                self.exclude_list.append(target_names[0])
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
                state.array_assign_name = var_name
                # Just like the same reason as the variables
                # declared with annotation within function (not
                # function arguments) need to have index of zero.
                # Thus, these 3 lines of code fixes the index to
                # correct value from -1 to 0.
                if target["var"]["index"] == -1:
                    target["var"]["index"] = 0
                    state.last_definitions[target_names[0]] = 0
                is_mutable = False
                array_info = {
                    "index": target["var"]["index"],
                    "dimensions": array_dimensions,
                    "elem_type": array_type,
                    "mutable": is_mutable,
                }
                self.arrays[var_name] = array_info
                state.array_types[var_name] = array_type
                if array_type == "string":
                    length = inputs[0][0]["call"]["inputs"][0][0]["value"]
                    self.strings[var_name] = {
                        "length": length,
                        "annotation": False,
                        "annotation_assign": True,
                    }

            if (
                maybe_d_type_object_assign
                and object_type
                and object_type in self.derived_types_attributes
                and target_names[0]
                in self.derived_types_attributes[object_type]
            ):
                self.current_d_object_name = d_type_object_name
                is_d_type_object_assignment = True

                # If targets holds more than 1 variable information and
                # it's greater than the object attribute number, then
                # the derived type object is referencing more than
                # 1 attribute (i.e. x.k.v).
                if len(targets) > 1 and len(targets) > object_attr_num:
                    object_attr_num += 1
                    # Therefore, we do not want to go any further before
                    # collecting all the information of the attribute
                    # information, so we need to simply return back to the
                    # beginning of loop and restart the process
                    continue
            else:
                is_d_type_object_assignment = False

            variable_spec = self.generate_variable_definition(
                target_names,
                d_type_object_name,
                is_d_type_object_assignment,
                state,
            )

            # Do not add the variable spec if this is a string annotation
            # since this can collide with the variable spec of the first
            # string assignment.
            if not is_string_annotation:
                grfn["variables"].append(variable_spec)

            # Since a Python class (derived type) object declaration has syntax
            # is __object_name__ = __class_name__, it's considered as an
            # assignment that will create __assign__ function GrFN,
            # which should not. Thus, simply return the [grfn] here to avoid
            # generating __assign__ function.
            if is_d_type_obj_declaration:
                return [grfn]

            # TODO Hack to not print lambda function for IO assigns. Need a
            #  proper method to handle IO moving on
            for src in sources:
                if "call" in src:
                    if self.check_io_variables(src["call"]["function"]):
                        io_source = True
                    function = src["call"]["function"]
                    # Check if the source is a function call by comparing its
                    # value with the list of functions in our program (
                    # obtained from the mode mapper)
                    for program_functions in self.mode_mapper["subprograms"]:
                        if (
                            function
                            in self.mode_mapper["subprograms"][
                                program_functions
                            ]
                        ):
                            is_function_call = True

            if is_function_call:
                container_name = self.generate_container_id_name(
                    self.fortran_file, ["@global"], function
                )
                function_name = {"name": container_name, "type": "container"}
            else:
                function_name = self.generate_function_name(
                    "__assign__", variable_spec["name"], None
                )
            # If current assignment process is for a derived type object (i.e
            # x.k), then
            if is_d_type_object_assignment:
                # (1) we need to add derived type object as function input.
                if state.last_definitions.get(d_type_object_name) is not None:
                    index = state.last_definitions[d_type_object_name]
                elif self.global_scope_variables and \
                        self.global_scope_variables.last_definitions.get(
                            d_type_object_name) is not None:
                    index = 0
                else:
                    assert False, f"{d_type_object_name} not defined in the " \
                                  f"the current scope: {self.current_scope} " \
                                  f"or globally."
                src = [
                    {
                        "var": {
                            "variable": d_type_object_name,
                            "index": index,
                        }
                    }
                ]
                sources.extend(src)

                # (2) Generate the object name + attributes variable name
                new_var_name = d_type_object_name
                for target_name in target_names:
                    new_var_name += f"__{target_name}"
                    self.current_d_object_attributes.append(target_name)

                # (3) we need to modify thee target to be "objectName_attribute"
                # For example, variable: x_k and index: __index_of_x_y__.
                target["var"] = {
                    "variable": new_var_name,
                    "index": state.last_definitions[new_var_name],
                }

            fn = self.make_fn_dict(function_name, target, sources, state)
            if len(fn) == 0:
                return []

            source_list = self.make_source_list_dict(sources)

            if not io_source and not is_function_call:
                lambda_string = self.generate_lambda_function(
                    node,
                    function_name["name"],
                    True,
                    array_assignment,
                    is_string_assign,
                    is_d_type_object_assignment,
                    source_list,
                    state,
                    False,
                )
                state.lambda_strings.append(lambda_string)

            grfn["functions"].append(fn)
            # We need to cleanup the object attribute tracking list.
            self.current_d_object_attributes = []
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
            if (
                isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
            ):
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
                function_name = ""
                func_name = function_node.attr
                if self.is_d_object_array_assign:
                    function_name = self.current_d_object_name + "."
                    self.is_d_object_array_assign = False
                function_name += module + "." + func_name
            elif isinstance(function_node.value, ast.Call):
                return self.gen_grfn(function_node.value, state, "call")
            elif (
                    isinstance(function_node.value, ast.Str)
                    and hasattr(function_node, "attr")
            ):
                module = function_node.value.s
                function_name = module + function_node.attr
            else:
                assert False, f"Invalid expression call" \
                              f" {dump_ast(function_node)}"
        else:
            function_name = node.func.id

        inputs = []
        for arg in node.args:
            argument = self.gen_grfn(arg, state, _)
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
            node_name = cur.__repr__().split()[0][2:]
            grfn = self.gen_grfn(cur, state, "module")
            if grfn and "name" in grfn[0] and "@type" in grfn[0]["name"]:
                self.derived_types_grfn.append(grfn[0])
            elif self.current_scope == "global" and  \
                    node_name in ["ast.AnnAssign", "ast.Assign", "ast.Expr"]:
                if not self.global_grfn["containers"][0]["name"]:
                    namespace = self._get_namespace(self.fortran_file)
                    self.global_grfn["containers"][0]["name"] = \
                        f"@container::{namespace}::@global"
                    self.global_grfn["containers"][0]["gensym"] = \
                        self.generate_gensym("container")
                if len(grfn) > 0:
                    self.global_grfn["containers"][0]["body"] += grfn[0][
                        "functions"]
                    self.global_grfn["variables"] += grfn[0]["variables"]
            else:
                grfn_list += grfn
        if self.global_grfn["containers"][0]["name"]:
            grfn_list += [self.global_grfn]
        merged_grfn = [self._merge_dictionary(grfn_list)]
        return merged_grfn
        # TODO Implement this. This needs to be done for generality
        # We fill in the `updated` field of function calls by looking at the
        # `updated` field of their container grfn
        final_grfn = self.load_updated(merged_grfn)
        return final_grfn

    @staticmethod
    def _process_nameconstant(node, *_):
        # TODO Change this format according to the new spec
        if isinstance(node.value, bool):
            dtype = "boolean"
        else:
            dtype = "string"
        return [{"type": "literal", "dtype": dtype, "value": node.value}]

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
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == self.current_d_object_name
            ):
                self.is_d_object_array_assign = True
            module = node.attr
            return module
        # When a computations float value is extracted using the Float32
        # class's _val method, an ast.Attribute will be present
        if node.attr == "_val":
            return self.gen_grfn(node.value, state, call_source)
        else:
            node_value = node.__repr__().split()[0][2:]
            if node_value != "ast.Attribute":
                # TODO: This section of the code should be the same as
                #  `process_name`. Verify this.
                last_definition = self.get_last_definition(
                    node.attr,
                    state.last_definitions,
                    state.last_definition_default,
                )
                # TODO Change the format according to the new spec
                return [
                    {"var": {"variable": node.attr, "index": last_definition}}
                ]
            else:
                (
                    attributes,
                    last_definitions,
                ) = self.get_derived_type_attributes(node, state)
                attribs = []
                for attr in attributes:
                    variable_info = {
                        "var": {
                            "variable": attr,
                            "index": last_definitions[attr],
                        }
                    }
                    attribs.append(variable_info)

                return attribs

    def process_return_value(self, node, state, *_):
        """
        This function handles the return value from a function.
        """
        grfn = {"functions": [], "variables": [], "containers": []}
        if node.value:
            val = self.gen_grfn(node.value, state, "return_value")
        else:
            val = None

        namespace = self._get_namespace(self.fortran_file)
        function_name = f"{namespace}__{self.current_scope}__return"
        function_name = self.replace_multiple(
            function_name, ["$", "-", ":"], "_"
        )
        function_name = function_name.replace(".", "__")
        return_dict = {
            "function": {"name": function_name, "type": "return"},
            "value": val,
        }
        grfn["functions"].append(return_dict)
        return [grfn]

    def process_class_def(self, node, state, *_):
        """This function handles user defined type (class) by populating
        types grfn attribute.
        """
        grfn = {"name": "", "type": "type", "attributes": []}
        namespace = self._get_namespace(self.fortran_file)
        type_name = f"@type::{namespace}::@global::{node.name}"
        grfn["name"] = type_name

        # Keep a track of declared user-defined types
        self.derived_types.append(node.name.lower())
        self.derived_types_attributes[node.name] = []

        attributes = node.body[0].body
        # Populate class member variables into attributes array.
        for attrib in attributes:
            attrib_is_array = False
            attrib_ast = attrib.__repr__().split()[0][2:]
            if attrib_ast == "ast.AnnAssign":
                attrib_name = attrib.target.attr
                if attrib.annotation.id in self.annotate_map:
                    attrib_type = self.annotate_map[attrib.annotation.id]
                elif attrib.annotation.id in self.derived_types:
                    attrib_type = attrib.annotation.id
            elif attrib_ast == "ast.Assign":
                attrib_name = attrib.targets[0].attr
                attrib_type = attrib.value.func.id
                assert (
                    attrib_type in self.derived_types
                    or attrib_type in self.library_types
                ), f"User-defined type [{attrib_type}] does not exist."

                if attrib_type == "Array":
                    attrib_is_array = True

            if attrib_is_array:
                elem_type = attrib.value.args[0].id
                # TODO: Currently, derived type array attributes are assumed
                # to be a single dimensional array with integer type. It maybe
                # appropriate to handle a multi-dimensional with variable used
                # as a dimension size.
                dimension_info = attrib.value.args[1]
                is_literal = False
                is_name = False

                single_dimension = False
                dimension_list = []
                if isinstance(dimension_info.elts[0], ast.Tuple):
                    lower_bound = int(dimension_info.elts[0].elts[0].n)
                    single_dimension = True

                    # Retrieve upper bound of an array.
                    if isinstance(dimension_info.elts[0].elts[1], ast.Num):
                        upper_bound = int(dimension_info.elts[0].elts[1].n)
                        is_literal = True
                    elif isinstance(dimension_info.elts[0].elts[1], ast.Name):
                        upper_bound = dimension_info.elts[0].elts[1].id
                        is_name = True
                    else:
                        assert False, (
                            f"Currently, ast type "
                            f"[{type(dimension_info.elts[0].elts[1])}] is not "
                            f"supported."
                        )

                    if is_literal:
                        dimension = (upper_bound - lower_bound) + 1
                    elif is_name:
                        dimension = upper_bound
                    else:
                        pass

                    dimension_list.append(dimension)

                elif isinstance(dimension_info.elts[0], ast.Call):
                    lower_bound = int(dimension_info.elts[0].func.elts[0].n)
                    if isinstance(
                        dimension_info.elts[0].func.elts[1], ast.Num
                    ):
                        upper_bound = int(
                            dimension_info.elts[0].func.elts[1].n
                        )
                        is_literal = True
                    elif isinstance(
                        dimension_info.elts[0].func.elts[1], ast.Name
                    ):
                        upper_bound = dimension_info.elts[0].func.elts[1].id
                        is_name = True

                    if is_literal:
                        first_dimension = (upper_bound - lower_bound) + 1
                    elif is_name:
                        first_dimension = upper_bound

                    dimension_list.append(first_dimension)

                    lower_bound = int(dimension_info.elts[0].args[0].n)

                    if isinstance(dimension_info.elts[0].args[1], ast.Num):
                        upper_bound = int(dimension_info.elts[0].args[1].n)
                        is_literal = True
                    elif isinstance(dimension_info.elts[0].args[1], ast.Name):
                        upper_bound = dimension_info.elts[0].args[1].id
                        is_name = True

                    if is_literal:
                        second_dimension = (upper_bound - lower_bound) + 1
                    elif is_name:
                        second_dimension = upper_bound

                    dimension_list.append(second_dimension)

                dimensions = dimension_list

                grfn["attributes"].append(
                    {
                        "name": attrib_name,
                        "type": attrib_type,
                        "elem_type": elem_type,
                        "dimensions": dimensions,
                    }
                )
                # Here index is not needed for derived type attributes,
                # but simply adding it as a placeholder to make a constant
                # structure with other arrays.
                self.arrays[attrib_name] = {
                    "index": 0,
                    "dimensions": dimensions,
                    "elem_type": elem_type,
                    "mutable": True,
                }
            else:
                grfn["attributes"].append(
                    {"name": attrib_name, "type": attrib_type}
                )
                pass
            self.derived_types_attributes[node.name].append(attrib_name)

            state.variable_types[attrib_name] = attrib_type

        return [grfn]

    def process_break(self, node, state, *_):
        """
            Process the breaks in the file, adding an EXIT node
        """
        # Get all the IF_X identifiers and pick the one with the largest
        # index because that is the current one
        if_ids = [int(x[-1]) for x in state.last_definitions.keys() if "IF_"
                  in x]
        current_if = f"IF_{max(if_ids)}"
        self.exit_candidates.append({
            current_if: "break"
        })
        grfn = {
            "functions": ["insert_break"],
            "variables": [],
            "containers": [],
        }
        return [grfn]

    @staticmethod
    def process_ast(node, *_):
        sys.stderr.write(
            f"No handler for AST.{node.__class__.__name__} in gen_grfn, "
            f"fields: {node._fields}\n"
        )

    def process_load(self, node, state, call_source):
        raise For2PyError(
            f"Found ast.Load, which should not happen. "
            f"From source: {call_source}"
        )

    def process_store(self, node, state, call_source):
        raise For2PyError(
            f"Found ast.Store, which should not happen. "
            f"From source: {call_source}"
        )

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

        # If the source is a list inside of a list, remove the outer list
        if len(source_dictionary) == 1 and isinstance(
            source_dictionary[0], list
        ):
            source_dictionary = source_dictionary[0]

        for src in source_dictionary:
            if "var" in src:
                if src["var"]["variable"] not in self.annotate_map:
                    source_list.append(src["var"]["variable"])
            elif "call" in src:
                for ip in src["call"]["inputs"]:
                    source_list.extend(self.make_source_list_dict(ip))
                if (
                    "f_index" in src["call"]["function"]
                    or "get_substr" in src["call"]["function"]
                ):
                    string_var = src["call"]["function"].split(".")[0]
                    source_list.append(string_var)
            elif "list" in src:
                for ip in src["list"]:
                    if "var" in ip:
                        source_list.append(ip["var"]["variable"])

        # Removing duplicates
        unique_source = []
        [
            unique_source.append(obj)
            for obj in source_list
            if obj not in unique_source
        ]
        source_list = unique_source

        return source_list

    def make_fn_dict(self, name, target, sources, state):
        source = []
        fn = {}
        io_source = False
        target_name = target["var"]["variable"]
        target_string = f"@variable::{target_name}::{target['var']['index']}"

        # If the source is a list inside of a list, remove the outer list
        if len(sources) == 1 and isinstance(sources[0], list):
            sources = sources[0]

        for src in sources:
            # Check for a write to a file
            if re.match(r"\d+", target_name) and "list" in src:
                return fn
            if "call" in src:
                function_name = src["call"]["function"]
                method_var = function_name.split(".")
                if len(method_var) > 1:
                    method_name = method_var[1]
                else:
                    method_name = function_name
                # Remove first index of an array function as it's
                # really a type name not the variable for input.
                if function_name is "Array":
                    del src["call"]["inputs"][0]
                # If a RHS of an assignment is an array getter,
                # for example, meani.get_((runs[0])), we only need
                # the array name (meani in this case) and append
                # to source.
                # if ".get_" in src["call"]["function"]:
                if method_name == "get_":
                    get_array_name = src["call"]["function"].replace(
                        ".get_", ""
                    )
                    var_arr_name = f"@variable::{get_array_name}::-1"
                    source.append(var_arr_name)

                # Bypassing identifiers who have I/O constructs on their source
                # fields too.
                # Example: (i[0],) = format_10_obj.read_line(file_10.readline())
                # 'i' is bypassed here
                # TODO this is only for PETASCE02.for. Will need to include 'i'
                #  in the long run
                bypass_match_source = self.check_io_variables(function_name)
                if bypass_match_source:
                    if "var" in src:
                        self.exclude_list.append(src["var"]["variable"])
                    # TODO This is a hack for SIR's retval to be included as
                    #  an assign function (will not have an input). But,
                    #  moving on a proper method for handling IO is required.
                    #  Uncomment the line below to revert to old form.
                    # return fn
                    io_source = True
                # TODO Finalize the spec for calls here of this form:
                #  "@container::<namespace_path_string>::<scope_path_string>::
                #   <container_base_name>" and add here.
                if not io_source:
                    for source_ins in self.make_call_body_dict(src):
                        source.append(source_ins)
                # Check for cases of string index operations
                if method_name in ["f_index", "get_substr"]:
                    string_var = src["call"]["function"].split(".")[0]
                    index = state.last_definitions[string_var]
                    source.append(f"@variable::{string_var}::{index}")

            elif "var" in src:
                source_string = (
                    f"@variable::{src['var']['variable']}::"
                    f"{src['var']['index']}"
                )
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
        [
            unique_source.append(obj)
            for obj in source
            if obj not in unique_source
        ]
        source = unique_source

        fn = {
            "function": name,
            "input": source,
            "output": [target_string],
            "updated": [],
        }

        return fn

    @staticmethod
    def _remove_io_variables(variable_list):
        """
            This function scans each variable from a list of currently defined
            variables and removes those which are related to I/O such as format
            variables, file handles, write lists and write_lines.
        """
        io_regex = re.compile(
            r"(format_\d+_obj)|(file_\d+)|(write_list_\d+)|" r"(write_line)"
        )
        io_match_list = [io_regex.match(var) for var in variable_list]

        return [
            var
            for var in variable_list
            if io_match_list[variable_list.index(var)] is None
        ]

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
                        source_string = (
                            f"@variable::"
                            f"{item['var']['variable']}::"
                            f"{item['var']['index']}"
                        )
                        source_list.append(source_string)
                    # TODO Adding boolean literals as an input to an assign
                    #  function but not integer literals?
                    elif (
                        "type" in item
                        and item["type"] == "literal"
                        and item["dtype"] == "boolean"
                    ):
                        source_string = (
                            f"@literal::"
                            f"{item['dtype']}::"
                            f"{item['value']}"
                        )
                        source_list.append(source_string)
                    elif "call" in item:
                        source_list.extend(self.make_call_body_dict(item))
                    elif "list" in item:
                        # Handles a case where array declaration size
                        # was given with a variable value.
                        for value in item["list"]:
                            if "var" in value:
                                variable = (
                                    f"@variable:"
                                    f":{value['var']['variable']}::0"
                                )
                                source_list.append(variable)

        return source_list

    def process_decorators(self, node, state):
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

                    if "String" in variable_type:
                        length_regex = re.compile(r"String\((\d+)\)", re.I)
                        match = length_regex.match(variable_type)
                        if match:
                            length = match.group(1)
                        elif (
                                hasattr(arg.values[1], "func")
                                and hasattr(arg.values[1], "args")
                                and hasattr(arg.values[1].args[0], "args")
                        ):
                            length = arg.values[1].args[0].args[0].n
                        else:
                            assert False, "Could not identify valid String type"
                        self.strings[variable] = {
                            "length": length,
                            "annotation": False,
                            "annotation_assign": False
                        }
                        state.variable_types[variable] = self.annotate_map[
                            "String"
                        ]
                    else:
                        state.variable_types[variable] = self.annotate_map[
                            variable_type
                        ]

    @staticmethod
    def process_try(node, state, call_source):
        return []

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

    def get_last_definition(
        self, var, last_definitions, last_definition_default
    ):
        """
            This function returns the last (current) definition (index) of a
            variable.
        """
        index = last_definition_default

        # Pre-processing and removing certain Assigns which only pertain to the
        # Python code and do not relate to the FORTRAN code in any way.
        bypass_match = self.re_bypass_io.match(var)

        if not bypass_match:
            if var in last_definitions:
                index = last_definitions[var]
            else:
                last_definitions[var] = index
            return index
        else:
            return 0

    @staticmethod
    def _get_next_definition(
        var, last_definitions, next_definitions, last_definition_default
    ):
        """
            This function returns the next definition i.e. index of a variable.
        """
        # The dictionary `next_definitions` holds the next index of all current
        # variables in scope. If the variable is not found (happens when it is
        # assigned for the first time in a scope), its index will be one greater
        # than the last definition default.

        index = next_definitions.get(var, last_definition_default + 1)
        # Update the next definition index of this variable by incrementing
        # it by 1. This will be used the next time when this variable is
        # referenced on the LHS side of an assignment.
        next_definitions[var] = index + 1
        # Also update the `last_definitions` dictionary which holds the current
        # index of all variables in scope.
        last_definitions[var] = index
        return index

    def get_variable_type(self, annotation_node):
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
        if self.annotate_map.get(data_type):
            return self.annotate_map[data_type]
        elif data_type in self.derived_types:
            return data_type
        else:
            assert False, "Unsupported type (only float, int, list, real, " \
                          "bool and str supported as of now).\n"

    @staticmethod
    def _get_variables_and_functions(grfn):
        variables = list(
            chain.from_iterable(stmt["variables"] for stmt in grfn)
        )
        fns = list(chain.from_iterable(stmt["functions"] for stmt in grfn))
        containers = list(
            chain.from_iterable(stmt["containers"] for stmt in grfn)
        )
        return variables, fns, containers

    def generate_gensym(self, tag):
        """
            The gensym is used to uniquely identify any identifier in the
            program. Python's uuid library is used to generate a unique 12 digit
            HEX string. The uuid4() function of 'uuid' focuses on randomness.
            Each and every bit of a UUID v4 is generated randomly and with no
            inherent logic. To every gensym, we add a tag signifying the data
            type it represents.
            'v': variables
            'c': containers
            'f': functions
        """
        return f"{self.gensym_tag_map[tag]}_{uuid.uuid4().hex[:12]}"

    def generate_lambda_function(
        self,
        node,
        function_name: str,
        return_value: bool,
        array_assign: bool,
        string_assign: bool,
        d_type_assign: bool,
        inputs,
        state,
        is_custom: bool,
    ):
        self.generated_lambda_functions.append(function_name)
        lambda_for_var = True
        lambda_strings = []
        argument_strings = []

        # We need to remove the attribute (class member var) from
        # the source_list as we do not need it in the lambda function
        # argument. Also, form an __object.attribute__ string.
        if d_type_assign:
            d_type = state.variable_types[self.current_d_object_name]
            target_name = self.current_d_object_name
            for attr in self.current_d_object_attributes:
                if attr in self.derived_types_attributes[d_type]:
                    target_name += f".{attr}"
                    # Since the next attribute that will be seen must be
                    # dependent on the current attribute type, here it's
                    # updating the d_type.
                    d_type = state.variable_types[attr]

        # If a custom lambda function is encountered, create its function
        # instead
        if is_custom:
            lambda_strings.append(
                f"def {function_name}({', '.join(inputs)}):\n    "
            )
            if return_value:
                if isinstance(node, str):
                    lambda_strings.append(f"return {node}")
                elif isinstance(node, int):
                    lambda_strings.append(f"return {node}")
                elif isinstance(node, list) and node[0] == "EXIT":
                    exit_string = f"(not {inputs[0]})"
                    for ip in inputs[1:]:
                        exit_string += f" or (not {ip})"
                    lambda_strings.append(f"return {exit_string}")
                else:
                    lambda_code_generator = genCode(self.use_numpy)
                    code = lambda_code_generator.generate_code(
                        node, PrintState("\n    ")
                    )
                    lambda_strings.append(f"return {code}")
            else:
                assert False, f"Should always return"
            lambda_strings.append("\n\n")
            return "".join(lambda_strings)

        # Sort the arguments in the function call as it is used in the operation
        input_list = sorted(set(inputs), key=inputs.index)
        # Add type annotations to the function arguments
        for ip in input_list:
            annotation = state.variable_types.get(ip)
            if ip in state.array_types:
                lambda_for_var = False
            if lambda_for_var and not annotation:
                # `variable_types` does not contain annotations for variables
                # for indexing such as 'abc_1', etc. Check if the such variables
                # exist and assign appropriate annotations
                key_match = lambda var, dicn: ([i for i in dicn if i in var])
                if len(key_match(ip, state.variable_types)) > 0:
                    annotation = state.variable_types[
                        key_match(ip, state.variable_types)[0]
                    ]
            # function argument requires annotation only when
            # it's dealing with simple variable (at least for now).
            # TODO String assignments of all kinds are class/method related
            #  operations and will not involve annotations. Discuss this.
            if lambda_for_var and annotation != "string":
                if annotation in self.annotate_map:
                    annotation = self.annotate_map[annotation]
                # else:
                #    assert annotation in self.derived_types, (
                #        f"Annotation must be a regular type or user defined "
                #        f"type. Annotation: {annotation}"
                #    )
                if annotation:
                    argument_strings.append(f"{ip}: {annotation}")
                else:
                    argument_strings.append(f"{ip}")
            # Currently, this is for array specific else case.
            else:
                argument_strings.append(ip)
                lambda_for_var = True

        lambda_strings.append(
            f"def {function_name}({', '.join(argument_strings)}):\n    "
        )
        # A case where calculating the sum of array.
        # In this case, we do not have to invoke genCode function(s).
        if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and "attr" in node.value.func._fields
                and node.value.func.attr == "get_sum"
        ):
            arr_name = node.value.func.value.id
            # TODO: Currently, only handles 1D-list to satisfy cases
            # that onnly appears in the Min-SPAM files.
            lambda_strings.append(f"return sum({arr_name})\n")

            return "".join(lambda_strings)

        # If a `decision` tag comes up, override the call to genCode to manually
        # enter the python script for the lambda file.
        if "__decision__" in function_name:
            if self.use_numpy:
                code = f"np.where({inputs[2]}, {inputs[1]}, {inputs[0]})"
            else:
                code = f"{inputs[1]} if {inputs[2]} else {inputs[0]}"
        elif not string_assign:
            array_name = None
            if state.array_assign_name:
                array_name = state.array_assign_name.split("[")[0]
            if array_name in self.strings:
                lambda_code_generator = genCode(
                    self.use_numpy, self.strings[array_name]["length"]
                )
            else:
                lambda_code_generator = genCode(self.use_numpy)
            code = lambda_code_generator.generate_code(
                node, PrintState("\n    ")
            )

        if return_value:
            if array_assign:
                if "_" in state.array_assign_name:
                    names = state.array_assign_name.split("_")
                    if names[0] == self.current_d_object_name:
                        state.array_assign_name = state.array_assign_name.replace(
                            "_", "."
                        )
                lambda_strings.append(f"{state.array_assign_name} = {code}\n")
                # lambda_strings.append(f"    return {state.array_assign_name}")
                lambda_strings.append(f"    return {array_name}")
                state.array_assign_name = None
            elif string_assign:
                lambda_code_generator = genCode(
                    self.use_numpy,
                    self.strings[state.string_assign_name]["length"],
                )
                code = lambda_code_generator.generate_code(
                    node, PrintState("\n    "),
                )

                if self.strings[state.string_assign_name]["annotation"]:
                    self.strings[state.string_assign_name][
                        "annotation"
                    ] = False
                if self.strings[state.string_assign_name]["annotation_assign"]:
                    self.strings[state.string_assign_name][
                        "annotation_assign"
                    ] = False
                lambda_strings.append(f"return {code}")
                state.string_assign_name = None
            elif d_type_assign:
                lambda_strings.append(f"{target_name} = {code}\n")
                lambda_strings.append(f"    return {target_name}")
            else:
                lambda_strings.append(f"return {code}")
        else:
            lines = code.split("\n")
            indent = re.search("[^ ]", lines[-1]).start()
            lines[-1] = lines[-1][:indent] + "return " + lines[-1][indent:]
            lambda_strings.append("\n".join(lines))
        lambda_strings.append("\n\n")
        return "".join(lambda_strings)

    def generate_container_id_name(
        self, namespace_file: str, scope_path, container_basename: str
    ) -> str:
        namespace = self._get_namespace(namespace_file)
        if isinstance(scope_path, list):
            scope_path_string = ".".join(scope_path)
        elif isinstance(scope_path, str):
            scope_path_string = scope_path
        else:
            assert False, f"Invalid scope_path type {scope_path}"
        container_id = (
            f"@container::{namespace}::{scope_path_string}::"
            f"{container_basename}"
        )

        return container_id

    def generate_variable_definition(
        self, variables, reference, d_type_object_assign, state
    ):
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
            Args:
                variables (list): List of variables.
                reference (str): Either array's indexing variable (i.e. i
                for array[i]) or derived type object's referencing class
                member variable (i.e. k for x.k)

            Returns:
                list : Generated GrFN.
        """
        namespace = self._get_namespace(self.fortran_file)
        index = []
        domains = []
        for variable in variables:
            if variable in state.last_definitions:
                index.append(state.last_definitions[variable])
            elif (
                variable in self.strings
                and self.strings[variable]["annotation"]
            ):
                # If this is a string initialization without assignment,
                # the index will be 0 by default
                index.append(0)
            elif variable in self.arrays:
                index.append(0)
            domains.append(self.get_domain_dictionary(variable, state))

        for domain in domains:
            # Since we need to update the domain of arrays that
            # were passed to a function once the program actually
            # finds about it, we need to temporarily hold the domain
            # information in the dictionary of domain list.
            if "name" in domain:
                if domain["name"] == "array":
                    if variable in self.array_arg_domain:
                        self.array_arg_domain[variable].append(domain)
                    else:
                        self.array_arg_domain[variable] = [domain]
                elif domain["name"] in self.derived_types:
                    self.derived_type_objects[variable] = domain["name"]

            # Only array variables hold dimensions in their domain
            # when they get declared, we identify the array variable
            # declaration by simply checking the existence of the dimensions
            # key in the domain. Also, the array was previously passed
            # to functions.
            if "dimensions" in domain and variable in self.array_arg_domain:
                # Since we can't simply do "dom = domain"
                # as this will do a replacement of the dict element
                # not the actual domain object of the original function
                # argument, we need to clean off the existing contents
                # first and then add the array domain spec one-by-one.
                for dom in self.array_arg_domain[variable]:
                    if "name" in dom:
                        del dom["name"]
                    if "type" in dom:
                        del dom["type"]
                    dom["index"] = domain["index"]
                    dom["dimensions"] = domain["dimensions"]
                    dom["elem_type"] = domain["elem_type"]
                    dom["mutable"] = domain["mutable"]

        variable_gensym = self.generate_gensym("variable")

        if reference is not None:
            if d_type_object_assign:
                variable = reference
                for var in variables:
                    variable += f"__{var}"
            # else:
            #     variable = f"{variable}_{reference}"
            # TODO: The code above has been commented for now but not removed.
            #  Remove this when everything works and the line below doesn't
            #  give any issues
            state.last_definitions[variable] = index[0]

        variable_name = (
            f"@variable::{namespace}::{self.current_scope}::"
            f"{variable}::{index[0]}"
        )
        # TODO Change the domain constraint. How do you figure out the domain
        #  constraint?
        domain_constraint = "(and (> v -infty) (< v infty))"

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
            type_name = None
            if (
                variable in state.variable_types
                and state.variable_types[variable]
            ):
                variable_type = state.variable_types[variable]
            elif variable in self.module_variable_types:
                variable_type = self.module_variable_types[variable][1]
            # Is this a derived type variable
            elif "__" in variable:
                variable_tail = variable.split("__")[-1]
                if variable_tail in state.variable_types and \
                        state.variable_types[variable_tail]:
                    variable_type = state.variable_types[variable_tail]
                else:
                    assert False, f"unrecognized variable: {variable}"
            else:
                variable_type = None
            #     assert False, f"unrecognized variable: {variable}"

            # Mark if a variable is mutable or not.
            if (
                variable in self.variable_map
                and self.variable_map[variable]["parameter"]
            ):
                is_mutable = True
            else:
                is_mutable = False

            # Array as a function argument handler.
            if self.handling_f_args and variable_type == "array":
                self.f_array_arg.append(variable)
            # Retrieve variable type name (i.e. integer, float,
            # __derived_type__)
            if variable_type in self.type_def_map:
                type_name = self.type_def_map[variable_type]
            elif variable_type in self.derived_types:
                type_name = variable_type
                # Since derived type variables are not a regular type variable,
                # we need to needs to them manually here into variable_types
                # dictionary to be referenced later in the stream.
                state.variable_types[variable] = type_name
            elif variable_type == "Real":
                type_name = "float"
            elif len(self.imported_module) > 0:
                for mod in self.imported_module:
                    if (
                            mod in self.module_summary
                            and variable in self.module_summary[mod]["symbol_types"]
                    ):
                        type_name = self.module_summary[mod]["symbol_types"]

            else:
                type_found = False
                if len(self.module_names) > 1:
                    for mod in self.module_names:
                        if (
                            mod != "main"
                            and mod in self.module_summary
                            and variable_type
                            in self.module_summary[mod]["derived_type_list"]
                        ):
                            type_found = True
                            type_name = variable_type
                            state.variable_types[variable] = type_name

            domain_dictionary = {
                "name": type_name,
                "type": "type",
                "mutable": is_mutable,
            }
        return domain_dictionary

    def generate_function_name(self, function_type, variable, arr_index):
        """
            This function generates the name of the function inside the
            container wiring within the body of a container.
        """
        variable_spec_regex = (
            r"@.*?::(?P<namescope>.*?::.*?)::("
            r"?P<variable>.*?)::(?P<index>.*)"
        )
        variable_match = re.match(variable_spec_regex, variable)
        if variable_match:
            namespace_scope = variable_match.group("namescope")
            variable_name = variable_match.group("variable")
            if arr_index:
                variable_name += "_"
                for index in arr_index:
                    variable_name = variable_name + f"{index}"
            variable_index = variable_match.group("index")

            name = (
                namespace_scope
                + function_type
                + variable_name
                + "::"
                + variable_index
            )
            name = self.replace_multiple(name, ["$", "-", ":"], "_")
            name = name.replace(".", "__")
            if any(
                [
                    x in function_type
                    for x in ["assign", "condition", "decision"]
                ]
            ):
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
                for container_grfn in grfn_dict[0]["containers"]:
                    for body_function in container_grfn["body"]:
                        function_name = body_function["function"]["name"]
                        if (
                            function_name.startswith("@container")
                            and function_name.split("::")[-1] == container
                        ):
                            updated_variable = [
                                body_function["input"][i]
                                for i in self.function_argument_map[container][
                                    "updated_indices"
                                ]
                            ]
                            for i in range(len(updated_variable)):
                                old_index = int(
                                    updated_variable[i].split("::")[-1]
                                )
                                new_index = old_index + 1
                                updated_var_list = updated_variable[i].split(
                                    "::"
                                )[:-1]
                                updated_var_list.append(str(new_index))
                                updated_variable[i] = "::".join(
                                    updated_var_list
                                )
                                self.current_scope = self.update_functions[
                                    container
                                ]["scope"]
                                variable_name = updated_var_list[1]
                                variable_spec = self.generate_variable_definition(
                                    variable_name,
                                    None,
                                    False,
                                    self.update_functions[container]["state"],
                                )
                                variable_name_list = variable_spec[
                                    "name"
                                ].split("::")[:-1]
                                variable_name_list.append(str(new_index))
                                variable_spec["name"] = "::".join(
                                    variable_name_list
                                )
                                grfn_dict[0]["variables"].append(variable_spec)
                            body_function["updated"] = updated_variable
        return grfn_dict

    @staticmethod
    def _get_array_dimension(sources, array_dimensions, inputs):
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
        if len(inputs[1]) > 1:
            for lst in inputs[1]:
                low_bound = int(lst[0]["list"][0]["value"])
                upper_bound = int(lst[0]["list"][1]["value"])
                array_dimensions.append(upper_bound - low_bound + 1)
        # 1-D array handler
        else:
            bounds = inputs[1][0][0]["list"]
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
                array_dimensions.append(upper_bound - low_bound + 1)
            elif isinstance(upper_bound, str):
                assert (
                    isinstance(low_bound, int) and low_bound == 0
                ), "low_bound must be <integer> type and 0 (zero) for now."
                array_dimensions.append(upper_bound)
            else:
                assert False, (
                    f"low_bound type: {type(low_bound)} is "
                    f"currently not handled."
                )

    def _generate_array_setter(
        self, node, function, arg, name, container_id_name, arr_index, state
    ):
        """
            This function is for handling array setter (ex. means.set_(...)).

            Args:
                node: The node referring to the array
                function (list): A list holding the information of the function
                for JSON and lambda function generation.
                arg (list): A list holding the arguments of call['inputs'].
                name (str): A name of the array.
                container_id_name (str): A name of function container. It's an
                array name with other appended info. in this function.
                arr_index (str): Index of a target array.
                state: The current state of the system

            Returns:
                (list) function: A completed list of function.
        """
        if "_" in name:
            names = name.split("_")
            if names[0] == self.current_d_object_name:
                argument_list = [names[0]]
            else:
                argument_list = [name]
        else:
            argument_list = [name]
        # Array index is always one of
        # the lambda function argument
        for idx in arr_index:
            if idx not in self.arithmetic_ops and isinstance(idx, str):
                argument_list.append(idx)
        # For array setter value handler
        for var in arg[0]["call"]["inputs"][0]:
            # If an input is a simple variable.
            if "var" in var:
                var_name = var["var"]["variable"]
                if var_name not in argument_list:
                    input_index = self.get_last_definition(
                        var_name,
                        state.last_definitions,
                        state.last_definition_default,
                    )
                    function["input"].append(
                        f"@variable::" f"{var_name}::" f"{input_index}"
                    )
                    argument_list.append(var_name)
                else:
                    # It's not an error, so just pass it.
                    pass
            # If an input is an array.
            elif "call" in var:
                ref_call = var["call"]
                if ".get_" in ref_call["function"]:
                    get_array_name = ref_call["function"].replace(".get_", "")
                    if get_array_name not in argument_list:
                        argument_list.append(get_array_name)
                        if get_array_name != name:
                            ip_index = self.get_last_definition(
                                get_array_name,
                                state.last_definitions,
                                state.last_definition_default,
                            )
                            function["input"].append(
                                f"@variable::"
                                f"{get_array_name}::"
                                f"{ip_index}"
                            )
                    else:
                        # It's not an error, so just pass it.
                        pass

        # Generate lambda function for array[index]
        lambda_string = self.generate_lambda_function(
            node,
            container_id_name,
            True,
            True,
            False,
            False,
            argument_list,
            state,
            False,
        )
        state.lambda_strings.append(lambda_string)

        return function

    def _generate_array_index(self, node):
        """This function is for generating array index grfn
        handling both single and multi-dimensional arrays.

        Args:
            node: The node referring to the array.

        Returns:
            (list) index: Formed array index.
        """
        args = node.value.args[0]
        args_name = args.__repr__().split()[0][2:]
        # Case 1: Single dimensional array
        if args_name == "ast.Subscript":
            if hasattr(args.value, "value"):
                return [args.value.value.id]
            else:
                return [args.value.id]
        elif args_name == "ast.Num":
            return [int(args.n) - 1]
        # Case 1.1: Single dimensional array with arithmetic
        # operation as setter index
        elif args_name == "ast.BinOp":
            left_ast = args.left.__repr__().split()[0][2:]
            right_ast = args.right.__repr__().split()[0][2:]
            # Get the operator's left side value
            if left_ast == "ast.Subscript":
                left = args.left.value.id
            elif left_ast == "ast.Num":
                left = args.left.n
            # Get the arithmetic operator
            op = self.arithmetic_ops[args.op.__repr__().split()[0][6:]]
            # Get the operator's right side value
            if right_ast == "ast.Subscript":
                right = args.right.value.id
            elif right_ast == "ast.Num":
                right = args.right.n
            return [left, op, right]
        # Case 2: Multi-dimensional array
        elif args_name == "ast.Tuple":
            if hasattr(node.value.func.value, "id"):
                md_array_name = node.value.func.value.id
            elif hasattr(node.value.func.value, "value"):
                md_array_name = node.value.func.value.value.id

            if md_array_name not in self.md_array:
                self.md_array.append(md_array_name)
            dimensions = args.elts
            dimension_list = []
            for dimension in dimensions:
                ast_name = dimension.__repr__().split()[0][2:]
                if ast_name == "ast.Subscript":
                    if hasattr(dimension.value, "id"):
                        dimension_list.append(dimension.value.id)
                    elif hasattr(dimension.value, "value"):
                        dimension_list.append(dimension.value.value.id)
                else:
                    assert ast_name == "ast.Num", (
                        f"Unable to handle {ast_name} for multi-dimensional "
                        f"array - {dump_ast(dimension)}"
                    )
            return dimension_list
        else:
            assert False, f"Unable to handle {args_name}"

    def get_derived_type_attributes(self, node, state):
        """ This function retrieves the derived type attributes
        from the ast and return the updated last definition dict
        and populated attribute list"""

        attributes = []

        node_value = node.value.__repr__().split()[0][2:]
        if node_value == "ast.Attribute" and node.value.attr:
            attributes.append(node.value.attr)

        if node.attr:
            attributes.append(node.attr)

        last_definitions = {}
        for attrib in attributes:
            last_definitions[attrib] = self.get_last_definition(
                attrib, state.last_definitions, state.last_definition_default
            )
        return attributes, last_definitions

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
    def _find_updated(argument_list, body_variable_list, f_array_arg, state):
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
        variable_regex = re.compile(r".*::(?P<variable>.*?)::(?P<index>.*$)")
        # First, get mapping of argument variables and their indexes
        for var in argument_list:
            var_match = variable_regex.match(var["name"])
            if var_match:
                argument_dict[var_match.group("variable")] = var_match.group(
                    "index"
                )
            else:
                assert False, (
                    f"Error when parsing argument variable " f"{var['name']}"
                )
        # Now, get mapping of body variables and their latest indexes
        for var in body_variable_list:
            var_match = variable_regex.match(var["name"])
            if var_match:
                body_dict[var_match.group("variable")] = var_match.group(
                    "index"
                )
            else:
                assert False, (
                    f"Error when parsing body variable " f"{var['name']}"
                )
        # Now loop through every argument variable over the body variable to
        # check if the indices mismatch which would indicate an updated variable
        for argument in argument_dict:
            if argument in body_dict and int(body_dict[argument]) > int(
                argument_dict[argument]
            ):
                updated_list.append(
                    f"@variable::{argument}::" f"{body_dict[argument]}"
                )
            # If argument is an array type, get the current index and
            # update it. Then, append to the function's updated list
            elif argument in f_array_arg:
                updated_idx = state.last_definitions[argument]
                updated_list.append(
                    f"@variable::{argument}::" f"{updated_idx}"
                )

        return updated_list

    def check_io_variables(self, variable_name):
        """
            This function scans the variable and checks if it is an io
            variable. It returns the status of this check i.e. True or False.
        """
        io_match = self.re_bypass_io.match(variable_name)
        return io_match

    @staticmethod
    def _get_call_inputs(
        call_function,
        function_input,
        container_argument,
        loop_condition_inputs,
        loop_condition_inputs_lambda,
        state,
    ):
        """
            This function parses a call function (such as when reading an
            array) and loads all respective input variables from it.
        """
        # First check if the call is from an array call. We only update the
        # lists if it is an array operation (such as samples.get_((x[0]))
        if ".get_" in call_function["function"]:
            array_name = call_function["function"].replace(".get_", "")
            array_index = state.last_definitions.get(
                array_name, state.last_definition_default
            )
            function_input.append(
                f"@variable::" f"{array_name}::" f"{array_index}"
            )
            container_argument.append(f"@variable::" f"{array_name}::-1")
            loop_condition_inputs.append(f"@variable::" f"{array_name}::-1")
            loop_condition_inputs_lambda.append(array_name)
            for inputs in call_function["inputs"]:
                if not isinstance(inputs, list):
                    inputs = [inputs]
                for var in inputs:
                    if "var" in var:
                        function_input.append(
                            f"@variable::"
                            f"{var['var']['variable']}::"
                            f"{var['var']['index']}"
                        )
                        container_argument.append(
                            f"@variable::{var['var']['variable']}::-1"
                        )
                        loop_condition_inputs.append(
                            f"@variable::" f"{var['var']['variable']}::-1"
                        )
                        loop_condition_inputs_lambda.append(
                            var["var"]["variable"]
                        )
        else:
            pass

    @staticmethod
    def _remove_duplicate_from_list(input_list):
        """
            This helper function removes any duplicates from a list
        """
        # return list(set(input_list))
        return list(OrderedDict.fromkeys(input_list))

    def get_variables(self, condition_sources, state):
        variable_list = list()
        for item in condition_sources:
            if isinstance(item, dict) and "var" in item:
                variable_list.append(item)
            elif isinstance(item, list):
                variable_list += self.get_variables(item, state)
            elif "call" in item:
                function_dict = item["call"]
                if "__str__" in function_dict["function"]:
                    var_name = function_dict["function"].split(".")[0]
                    var_node = [
                        {
                            "var": {
                                "variable": var_name,
                                "index": state.last_definitions[var_name],
                            }
                        }
                    ]
                    variable_list += var_node
                # TODO: Will have to add other if cases for other string
                #  types here

        # Remove any duplicate dictionaries from the list. This is done to
        # preserve ordering.
        # Reference: https://www.geeksforgeeks.org/
        # python-removing-duplicate-dicts-in-list/
        variable_list = [
            i
            for n, i in enumerate(variable_list)
            if i not in variable_list[n + 1 :]
        ]

        return variable_list


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
        source_match = re.match(r"[./]*(.*)", file_name)
        assert source_match, (
            f"Original Fortran source file for {file_name} " f"not found."
        )
        return source_match.group(1)
    elif instance == "namespace":
        source_match = re.match(r"[./]*(.*)\.", file_name)
        assert source_match, f"Namespace path for {file_name} not found."
        return source_match.group(1).split("/")
    else:
        assert False, f"Error when trying to get the path of file {file_name}."


def dump_ast(
    node, annotate_fields=True, include_attributes=False, indent="  "
):
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
            fields = [
                (a, _format(b, level)) for a, b in ast.iter_fields(ast_node)
            ]
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


def process_comments(source_comment_dict, generator_object):
    """
        This function replaces the keys in the source comments that are
        function names in the source files into their container id name.
    """
    grfn_argument_map = generator_object.function_argument_map
    for key in source_comment_dict:
        if key in grfn_argument_map:
            source_comment_dict[
                grfn_argument_map[key]["name"]
            ] = source_comment_dict.pop(key)

    return source_comment_dict


# noinspection PyDefaultArgument
def create_grfn_dict(
    lambda_file: str,
    asts: List,
    file_name: str,
    mode_mapper_dict: list,
    original_file: str,
    mod_log_file_path: str,
    comments: dict,
    module_file_exist=False,
    module_import_paths={},
) -> Dict:
    """ Create a Python dict representing the GrFN, with additional metadata
    for JSON output. """

    lambda_string_list = [
        "from numbers import Real\n",
        "from random import random\n",
        "from delphi.translators.for2py.strings import *\n",
        "import numpy as np\n",
        "from delphi.translators.for2py import intrinsics\n",
        "import delphi.translators.for2py.math_ext as math\n\n",
    ]

    state = GrFNState(lambda_string_list)
    generator = GrFNGenerator()
    generator.mode_mapper = mode_mapper_dict[0]
    # Populate list of modules that the program imports
    for mod in generator.mode_mapper["modules"]:
        if mod != "main":
            generator.module_names.append(mod)
    generator.fortran_file = original_file

    generator.comments = comments
    # Currently, we are specifying the module file with
    # a prefix "m_", this may be changed in the future.
    # If it requires a change, simply modify this below prefix.
    module_file_prefix = "m_"

    with open(mod_log_file_path) as json_f:
        module_logs = json.load(json_f)
        # Load module summary on memory for later use
        generator.module_summary = module_logs["mod_info"]

    try:
        filename_regex = re.compile(r"(?P<path>.*/)(?P<filename>.*).py")
        file_match = re.match(filename_regex, file_name)
        assert file_match, f"Can't match filename to any format: {file_name}"

        path = file_match.group("path")
        filename = file_match.group("filename")

        # Since we do not have separate variable pickle file
        # for m_*.py, we need to use the original program pickle
        # file that module resides.
        module_name = None
        if module_file_exist:
            module_file_path = file_name
            # Ignoring the module file prefix
            module_name = filename[len(module_file_prefix) :]
            org_file = get_original_file_name(original_file)
            file_name = path + org_file
        else:
            file_name = path + filename

        with open(f"{file_name}_variable_map.pkl", "rb") as f:
            variable_map = pickle.load(f)
        generator.variable_map = variable_map
    except IOError:
        raise For2PyError(f"Unable to read file {file_name}.")
    # Extract variables with type that are declared in module
    generator.module_variable_types = mode_mapper_dict[0]["symbol_types"]
    # Extract functions (and subroutines) declared in module
    for module in mode_mapper_dict[0]["subprograms"]:
        for subp in mode_mapper_dict[0]["subprograms"][module]:
            generator.module_subprograms.append(subp)
            if module in module_logs["mod_info"]:
                module_logs["mod_info"][module]["symbol_types"][subp] = "func"

    for module in mode_mapper_dict[0]["imports"]:
        for subm in mode_mapper_dict[0]["imports"][module]:
            import_module_name = list(subm.keys())[0]
            import_function_list = subm[import_module_name]
            if (
                not import_function_list
                and import_module_name in module_logs["mod_info"]
            ):
                symbols = module_logs["mod_info"][import_module_name][
                    "symbol_types"
                ]
                for key, value in symbols.items():
                    if value == "func":
                        generator.module_subprograms.append(key)

            generator.module_subprograms.extend(import_function_list)

    # Generate GrFN with an AST generated from Python IR.
    grfn = generator.gen_grfn(asts, state, "")[0]

    if len(generator.mode_mapper["use_mapping"]) > 0:
        for user, module in generator.mode_mapper["use_mapping"].items():
            if (user in generator.module_names) or (
                module_name and module_name == user
            ):
                module_paths = []
                for import_mods in module:
                    for mod_name, target in import_mods.items():
                        module_path = (
                            path + module_file_prefix + mod_name + "_AIR.json"
                        )
                        module_paths.append(module_path)
                module_import_paths[user] = module_paths

    # If the GrFN has a `start` node, it will refer to the name of the
    # PROGRAM module which will be the entry point of the GrFN.
    if grfn.get("start"):
        grfn["start"] = [grfn["start"][0]]
    elif generator.function_definitions:
        # TODO: The `grfn_spec` mentions this to be null (None) but it looks
        #  like `networks.py` requires a certain function. Finalize after
        #  `networks.py` is completed.
        # grfn["start"] = None
        grfn["start"] = [generator.function_definitions[-1]]
    else:
        grfn["start"] = None

    # Add the placeholder to enter the grounding and link hypothesis information
    grfn["grounding"] = []
    # TODO Add a placeholder for `types`. This will have to be populated when
    #  user defined types start appearing.
    grfn["types"] = generator.derived_types_grfn
    # Get the file path of the original Fortran code being analyzed
    source_file = get_path(original_file, "source")

    # TODO Hack: Currently only the file name is being displayed as the
    #  source in order to match the handwritten SIR model GrFN JSON. Since
    #  the directory of the `SIR-Gillespie-SD_inline.f` file is the root,
    #  it works for this case but will need to be generalized for other cases.
    file_path_list = source_file.split("/")
    grfn["source"] = [file_path_list[-1]]

    # Get the source comments from the original Fortran source file.
    source_file_comments = get_comments(original_file)
    comment_dict = process_comments(dict(source_file_comments), generator)
    source_comments = comment_dict
    grfn["source_comments"] = source_comments

    # dateCreated stores the date and time on which the lambda and GrFN files
    # were created. It is stored in the YYYMMDD format
    grfn["date_created"] = f"{datetime.utcnow().isoformat('T')}Z"

    with open(lambda_file, "w") as lambda_fh:
        lambda_fh.write("".join(lambda_string_list))

    with open(mod_log_file_path, "w+") as json_f:
        json_f.write(json.dumps(module_logs, indent=2))

    del state
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
            system_name_match = re.match(r".*/(.*)\.py", file)
            assert system_name_match, f"System name for file {file} not found."
            system_name = system_name_match.group(1)

            path_match = re.match(r"(.*)/.*", file)
            assert path_match, "Target path not found"
            path = path_match.group(1)

    if not (system_name or path):
        assert False, (
            f"Error when trying to find the system name of the "
            f"analyzed program."
        )

    return system_name, path


def generate_system_def(
    python_list: List[str],
    module_grfn_list: List[str],
    import_grfn_paths: List[str],
    module_logs: Dict,
    original_file_path: str,
):
    """ This function generates the system definition for the system under
    analysis and writes this to the main system file.  """
    (system_name, path) = get_system_name(python_list)
    system_filepath = f"{path}/system.json"
    module_name_regex = re.compile(
        r"(?P<path>.*/)m_(" r"?P<module_name>.*)_AIR.json"
    )

    grfn_components = []
    for module_grfn in module_grfn_list:
        code_sources = []
        module_match = re.match(module_name_regex, module_grfn)
        if module_match:
            module_name = module_match.group("module_name")
            if module_name in module_logs["mod_to_file"]:
                for path in module_logs["mod_to_file"][module_name]:
                    code_sources.append(path)

        if not code_sources:
            code_sources.append(original_file_path)
        grfn_components.append(
            {
                "grfn_source": module_grfn,
                "code_source": code_sources,
                "imports": [],
            }
        )
    for grfn in import_grfn_paths:
        for path in import_grfn_paths[grfn]:
            grfn_components[0]["imports"].append(path)
    system_def = {"name": system_name, "components": grfn_components}
    if os.path.isfile(system_filepath):
        with open(system_filepath, "r") as f:
            systems_def = json.load(f)
            systems_def["systems"].append(system_def)
    else:
        systems_def = {"systems": [system_def]}

    return system_def


def process_files(
    python_list: List[str],
    grfn_tail: str,
    lambda_tail: str,
    original_file_path: str,
    print_ast_flag=False,
):
    """ This function takes in the list of python files to convert into GrFN
    and generates each file's AST along with starting the GrFN generation
    process. """

    module_file_exist = False

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

        # Ignore all Python files of modules created by `pyTranslate.py`
        # since these module files do not contain a corresponding XML file.
        if not filename.startswith("m_"):
            xml_file = f"{path}rectified_{filename}.xml"
            # Calling the `get_index` function in `mod_index_generator.py` to
            # map all variables and objects in the various files
        else:
            module_file_exist = True
            file_name = get_original_file_name(original_file_path)
            xml_file = f"{path}rectified_{file_name}.xml"
        # Calling the `get_index` function in `mod_index_generator.py` to
        # map all variables and objects in the various files
        module_mapper = get_index(xml_file)

    module_import_paths = {}
    for index, ast_string in enumerate(ast_list):
        lambda_file = python_list[index][:-3] + "_" + lambda_tail
        grfn_file = python_list[index][:-3] + "_" + grfn_tail
        grfn_dict = create_grfn_dict(
            lambda_file,
            [ast_string],
            python_list[index],
            module_mapper,
            original_file_path,
            module_file_exist,
            module_import_paths,
        )
        if module_file_exist:
            main_python_file = path + file_name + ".py"
            python_list[index] = main_python_file
        grfn_filepath_list.append(grfn_file)
        # Write each GrFN JSON into a file
        with open(grfn_file, "w") as file_handle:
            file_handle.write(json.dumps(grfn_dict, sort_keys=True, indent=2))


def get_original_file_name(original_file_path):
    original_file = original_file_path.split("/")
    return original_file[-1].split(".")[0]


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

    process_files(
        python_file_list, grfn_suffix, lambda_suffix, fortran_file, print_ast
    )
