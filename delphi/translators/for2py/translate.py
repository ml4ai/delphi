"""
This script converts the XML version of AST of the Fortran
file into a JSON representation of the AST along with other
non-source code information. The output is a pickled file
which contains this information in a parsable data structure.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        python translate.py -f <ast_file> -g <pickle_file> -i <f_src_file>

    where f_src_file is the Fortran source file for ast_file.

ast_file: The XML representation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

pickle_file: The file which will contain the pickled version of JSON AST and
supporting information. """


import sys
import argparse
import pickle
import copy
import xml.etree.ElementTree as ET
from typing import List, Dict
from collections import OrderedDict
from delphi.translators.for2py.get_comments import get_comments
from delphi.translators.for2py.loop_handle import RefactorConstructs


class ParseState(object):
    """This class defines the state of the XML tree parsing
    at any given root. For any level of the tree, it stores
    the subroutine under which it resides along with the
    subroutines arguments."""

    def __init__(self, subroutine=None):
        self.subroutine = subroutine if subroutine is not None else {}
        self.args = []
        if "args" in self.subroutine:
            for arg in self.subroutine["args"]:
                if "name" in arg:
                    self.args.append(arg["name"])

    def copy(self, subroutine=None):
        return ParseState(
            self.subroutine if subroutine is None else subroutine
        )


class XML_to_JSON_translator(object):
    def __init__(self):
        self.libRtns = ["read", "open", "close", "format", "print", "write"]
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
            "len",
            "adjustl",
            "adjustr",
        ]
        self.handled_tags = [
            "access-spec",
            "argument",
            "assignment",
            "call",
            "close",
            "component-decl",
            "declaration",
            "dimension",
            "dimensions",
            "exit",
            "explicit-shape-spec-list__begin",
            "format",
            "format-item",
            "function",
            "if",
            "index-variable",
            "io-control-spec",
            "keyword-argument",
            "literal",
            "loop",
            "module",
            "name",
            "open",
            "operation",
            "program",
            "range",
            "read",
            "return",
            "stop",
            "subroutine",
            "type",
            "use",
            "variable",
            "variables",
            "write",
            "save-stmt",
            "saved-entity",
            "constants",
            "interface",
            "names",
        ]
        self.handled_tags += self.libRtns

        self.ast_tag_handlers = {
            "argument": self.process_argument,
            "assignment": self.process_assignment,
            "call": self.process_call,
            "close": self.process_direct_map,
            "declaration": self.process_declaration,
            "dimension": self.process_dimension,
            "exit": self.process_terminal,
            "format-item": self.process_format_item,
            "format": self.process_format,
            "function": self.process_function,
            "if": self.process_if,
            "index-variable": self.process_index_variable,
            "io-controls": self.process_io_control,
            "keyword-argument": self.process_keyword_argument,
            "literal": self.process_literal,
            "loop": self.process_loop,
            "module": self.process_subroutine_or_program_module,
            "name": self.process_name,
            "open": self.process_direct_map,
            "operation": self.process_operation,
            "program": self.process_subroutine_or_program_module,
            "range": self.process_range,
            "read": self.process_direct_map,
            "return": self.process_terminal,
            "stop": self.process_terminal,
            "subroutine": self.process_subroutine_or_program_module,
            "type": self.process_type,
            "use": self.process_use,
            "variables": self.process_variables,
            "variable": self.process_variable,
            "constants": self.process_constants,
            "constant": self.process_constant,
            "write": self.process_direct_map,
            "derived-types": self.process_derived_types,
            "length": self.process_length,
            "save-stmt": self.process_save,
            "cycle": self.process_continue,
            "select": self.process_select,
            "case": self.process_case,
            "value-range": self.process_value_range,
            "interface": self.process_interface,
            "argument-types": self.process_argument_types,
        }

        self.unhandled_tags = set()  # unhandled xml tags in the current input
        self.summaries = {}
        self.asts = {}
        self.functionList = {}
        self.subroutineList = []
        self.entryPoint = []
        # Dictionary to map all the variables defined in each function
        self.variable_list = {}
        # Dictionary to map the arguments to their functions
        self.argument_list = {}
        # String that holds the current function under context
        self.current_module = None
        # Flag that specifies whether a SAVE statement has been encountered
        # in the subroutine/function or not
        self.is_save = False
        # Variable to hold the node of the SAVE statement to process at the
        # end of the subroutine/function
        self.saved_node = None
        # This list holds the nodes of the file handles that needs to be
        # SAVEd in the python translated code.
        self.saved_filehandle = []
        # Dictionary to hold the different loop constructs present with a loop
        self.loop_constructs = {}
        self.loop_index = 0
        self.break_index = 0
        self.cycle_index = 0
        self.return_index = 0
        self.loop_active = False
        self.derived_type_list = []

    def process_subroutine_or_program_module(self, root, state):
        """ This function should be the very first function to be called """
        subroutine = {"tag": root.tag, "name": root.attrib["name"].lower()}
        self.current_module = root.attrib["name"].lower()
        self.summaries[root.attrib["name"]] = None
        if root.tag not in self.subroutineList:
            self.entryPoint.append(root.attrib["name"])
        for node in root:
            if node.tag == "header":
                subroutine["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                sub_state = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, sub_state)
            elif node.tag == "members":
                subroutine["body"] += self.parseTree(node, sub_state)

        # Check if this subroutine had a save statement and if so, process
        # the saved node to add it to the ast
        if self.is_save:
            subroutine["body"] += self.process_save(self.saved_node, state)
            self.is_save = False
        elif self.saved_filehandle:
            subroutine["body"] += [{"tag": "save", "scope":
                                    self.current_module, "var_list":
                                    self.saved_filehandle}]
            self.saved_filehandle = []

        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_call(self, root, state) -> List[Dict]:
        """ This function handles <call> tag and its subelement <name>. """
        assert (
            root.tag == "call"
        ), f"The root must be <call>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        call = {"tag": "call"}
        for node in root:
            if node.tag == "name":
                call["name"] = node.attrib["id"].lower()
                call["args"] = []
                for arg in node:
                    call["args"] += self.parseTree(arg, state)
        return [call]

    def process_argument(self, root, state) -> List[Dict]:
        """ This function handles <argument> tag. It simply create a new AST
        list and copy the values (tag and attributes) to it.  """
        assert root.tag == "argument", "The root must be <argument>"
        var_name = root.attrib["name"].lower()
        if (
                "type" in root.attrib
                and root.attrib["type"] in self.derived_type_list
        ):
            is_derived_type = "true"
        else:
            is_derived_type = "false"

        array_status = root.attrib["is_array"]
        # If the root does not have any children, this argument tag is a
        # function argument variable. Otherwise, this argument is a named
        # argument to a function (E.g.: index(back = ".true."))
        if len(root) > 0:
            value = []
            for node in root:
                value += self.parseTree(node, state)
            return [
                {"tag": "arg",
                 "name": var_name,
                 "is_array": array_status,
                 "value": value,
                 "is_derived_type": is_derived_type
                 }
            ]
        else:
            # Store each argument respective to the function it is defined in
            self.argument_list.setdefault(self.current_module, []).append(
                 var_name)
            return [{"tag": "arg", "name": var_name, "is_array":
                    array_status, "is_derived_type": is_derived_type}]

    def process_declaration(self, root, state) -> List[Dict]:
        """ This function handles <declaration> tag and its sub-elements by
        recursively calling the appropriate functions for the target tag. """

        declared_type = []
        declared_variable = []
        assert (
            root.tag == "declaration"
        ), f"The root must be <declaration>. Current tag is {root.tag} with "\
            f"{root.attrib} attributes."

        # Check if this is a parameter declaration under which case,
        # the declaration would be turned into an assignment operation
        if root.attrib.get("type") == "parameter":
            parameter_assignment = []
            for node in root:
                parameter_assignment += self.parseTree(node, state)
            return parameter_assignment
        elif root.attrib.get("type") == "data":
            return self.handle_data_statements(root, state)
        for node in root:
            if node.tag not in self.handled_tags:
                self.unhandled_tags.add(node.tag)
            elif node.tag == "type":  # Get the variable type
                if root.attrib["type"] == "variable":
                    declared_type += self.parseTree(node, state)
                else:
                    # If the current node is for declaring a derived type,
                    # every step from type declaration to variable (including
                    # array) declaration will be done in the
                    # "process_derived_types" function and return the completed
                    # AST list object back.  Thus, simply insert the received
                    # AST list object into the declared_variable object. No
                    # other work is done in the current function.
                    declared_variable += self.parseTree(node, state)
                    self.derived_type_list.append(declared_variable[0]["type"])
            elif node.tag == "dimensions":
                num_of_dimensions = int(node.attrib["count"])
                dimensions = {
                    "count": num_of_dimensions,
                    "dimensions": self.parseTree(node, state),
                }
                # Since we always want to access the last element of the list
                # that was added most recently (that is a currently handling
                # variable), add [-1] index to access it.
                if len(declared_type) > 0:
                    declared_type[-1].update(dimensions)
                else:
                    declared_type.append(dimensions)
            elif node.tag == "variables":
                variables = self.parseTree(node, state)
                # Declare variables based on the counts to handle the case
                # where a multiple variables declared under a single type
                # for index in range(int(node.attrib["count"])):
                for index in range(len(variables)):
                    if len(declared_type) > 0:
                        combined = declared_type[-1].copy()
                        combined.update(variables[index])
                        declared_variable.append(combined.copy())
                        if (
                            state.subroutine["name"].lower() in list(
                             self.functionList.keys())
                            and declared_variable[-1]["name"] in state.args
                        ):
                            state.subroutine["args"][
                                state.args.index(
                                    declared_variable[index]["name"]
                                )
                            ]["type"] = declared_variable[index]["type"]
                        if declared_variable[-1]["name"] in state.args:
                            state.subroutine["args"][
                                state.args.index(
                                    declared_variable[index]["name"]
                                )
                            ]["type"] = declared_variable[index]["type"]
            elif (
                    node.tag == "save-stmt"
                    or node.tag == "interface"
                    or node.tag == "names"
            ):
                declared_variable = self.parseTree(node, state)

        # Create an exclusion list of all variables which are arguments
        # to the function/subroutine in context and to
        # function/subroutine names themselves
        exclusion_list = list(self.functionList.keys()) + self.subroutineList
        if self.argument_list.get(self.current_module):
            exclusion_list += self.argument_list[self.current_module]
        exclusion_list = list(set([x.lower() for x in exclusion_list]))

        # Map each variable declaration to this parent
        # function/subroutine to keep a track of local variables
        if len(declared_variable) > 0:
            for var in declared_variable:
                if (var.get("tag") in ["variable", "array"] and
                        var.get("name") not in exclusion_list) or \
                    (var.get("is_derived_type") is True and var.get("type")
                     not in exclusion_list):
                    self.variable_list.setdefault(self.current_module,
                                                  []).append(var)
        return declared_variable

    def process_type(self, root, state) -> List[Dict]:
        """ This function handles <type> declaration.

        There may be two different cases of <type>.
            (1) Simple variable type declaration
            (2) Derived type declaration
        """

        assert (
            root.tag == "type"
        ), f"The root must be <type>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        derived_type = []
        if (
            root.text
        ):  # Check if the <type> has sub-elements, which is the case of (2)
            for node in root:
                if node.tag == "type":
                    derived_type += self.parseTree(node, state)
                elif node.tag == "length":
                    if root.attrib["name"].lower() == "character":
                        string_length = self.parseTree(node, state)
                        declared_type = {
                            "type": root.attrib["name"].lower(),
                            "length": string_length[0]["value"],
                            "is_derived_type": root.attrib[
                                "is_derived_type"].lower(),
                            "is_string": "true",
                            "keyword2": root.attrib["keyword2"],
                        }
                        return [declared_type]
                    else:
                        is_derived_type = False
                        if "is_derived_type" in root.attrib:
                            is_derived_type = root.attrib[
                                "is_derived_type"
                            ].lower()
                        keyword2 = "none"
                        if "keyword2" in root.attrib:
                            keyword2 = root.attrib["keyword2"]
                        declared_type = {
                            "type": root.attrib["name"],
                            "is_derived_type": is_derived_type,
                            "keyword2": keyword2,
                            }
                        declared_type["value"] = self.parseTree(node, state)
                        return [declared_type]
                elif node.tag == "derived-types":
                    derived_type[-1].update(self.parseTree(node, state))
            return derived_type
        else:
            if root.attrib["name"].lower() == "character":
                # Check if this is a string
                declared_type = {
                    "type": root.attrib["name"],
                    "length": root.attrib["string_length"],
                    "is_derived_type": root.attrib["is_derived_type"].lower(),
                    "is_string": "true",
                    "keyword2": root.attrib["keyword2"],
                }
            else:
                # Else, this represents an empty element, which is the case
                # of (1)
                declared_type = {
                    "type": root.attrib["name"],
                    "is_derived_type": root.attrib["is_derived_type"].lower(),
                    "keyword2": root.attrib["keyword2"],
                    "is_string": "false",
                }
            return [declared_type]

    def process_length(self, root, state) -> List[Dict]:
        """ This function handles <length> tag.  """
        assert (
            root.tag == "length"
        ), f"The root must be <length>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        length = {}
        for node in root:
            if node.tag == "literal":
                length.update(self.parseTree(node, state)[-1])
            elif node.tag == "type-param-value" and node.attrib[
                 "hasAsterisk"] == "true":
                length["value"] = "*"
            else:
                self.unhandled_tags.add(node.tag)
        return [length]

    def process_variables(self, root, state) -> List[Dict]:
        """ This function handles <variables> element, which its duty is to
        call <variable> tag processor. """
        try:
            variables = []
            assert (
                root.tag == "variables"
            ), f"The root must be <variables>. Current tag is {root.tag} " \
                f"with {root.attrib} attributes."
            for node in root:
                variables += self.parseTree(node, state)
            return variables
        except:
            return []

    def process_variable(self, root, state) -> List[Dict]:
        """
        This function will get called from the process_variables function, and
        it will construct the variable AST list, then return it back to the
        called function.
        """
        assert (
            root.tag == "variable"
        ), f"The root must be <variable>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        try:
            # First check if the variables are actually function names
            if root.attrib["name"].lower() in list(self.functionList.keys()):
                return []
            var_name = root.attrib["name"].lower()
            is_array = root.attrib["is_array"].lower()

            variable = {"name": var_name, "is_array": is_array}
            if is_array == "true":
                variable["tag"] = "array"
            else:
                variable["tag"] = "variable"

            if root.text:
                for node in root:
                    if node.tag == "initial-value":
                        value = self.parseTree(node, state)
                        variable["value"] = value
                    elif node.tag == "length":
                        variable["length"] = self.parseTree(node, state)[0][
                            "value"]
            return [variable]
        except:
            return []

    def process_constants(self, root, state) -> List[Dict]:
        """ This function handles <constants> element, which its duty is to
        call <constant> tag processor"""
        try:
            constants = []
            assert (
                root.tag == "constants"
            ), f"The root must be <constants>. Current tag is {root.tag}" \
                f"with {root.attrib} attributes."
            for node in root:
                constants += self.parseTree(node, state)
            return constants
        except:
            return []

    def process_constant(self, root, state) -> List[Dict]:
        """
        This function will get called from the process_constants function, and
        it will construct the constant AST list, then return it back to the
        called function.
        """

        assert (
                root.tag == "constant"
        ), f"The root must be <constant>. Current tag is {root.tag} with " \
           f"{root.attrib} attributes."
        assign = {"tag": "assignment"}

        # Populate the target field of the parameter assignment
        target = {
            "tag": "ref",  # Default for a normal variable
            "is_array": root.attrib["is_array"],
            "name": root.attrib["name"].lower(),
            "numPartRef": "1",  # Default value of 1
            "is_arg": "false",
            "hasSubscripts": "false",  # Default of false
            "is_derived_type_ref": "false",  # Default of false
            "is_parameter": "true",
        }

        assign["target"] = [target]

        for node in root:
            assign["value"] = self.parseTree(node, state)

        return [assign]

    def process_derived_types(self, root, state) -> List[Dict]:
        """ This function handles <derived-types> tag nested in the <type> tag.
        Depends on the nested sub-elements of the tag, it will recursively call
        other tag processors.

        (1) Main type declaration
        (2) Single variable declaration (with initial values)
        (3) Array declaration
        """

        assert (
            root.tag == "derived-types"
        ), f"The root must be <derived-type>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        derived_types = {"derived-types": []}
        declared_type = []
        for node in root:
            if node.tag not in self.handled_tags:
                self.unhandled_tags.add(node.tag)
            elif node.tag == "type":  # Get the variable type
                declared_type += self.parseTree(node, state)
            elif node.tag == "dimensions":
                dimensions = {
                    "count": node.attrib["count"],
                    "dimensions": [],
                }
                dims = self.parseTree(node, state)
                for dim in dims:
                    dim_info = {
                            "tag": "dimension",
                            "range": dim["range"]
                    }
                    dimensions["dimensions"].append(dim_info)
                declared_type[-1].update(dimensions)
            elif node.tag == "variables":
                variables = self.parseTree(node, state)
                # Declare variables based on the counts to handle the case
                # where a multiple vars declared under a single type
                for index in range(int(node.attrib["count"])):
                    combined = declared_type[-1]
                    combined.update(variables[index])
                    derived_types["derived-types"].append(combined.copy())
        return derived_types

    def process_loop(self, root, state) -> List[Dict]:
        """ This function handles <loop type=""> tag.  The type attribute
        indicates the current loop is either "do" or "do-while" loop. """
        assert (
            root.tag == "loop"
        ), f"The root must be <loop>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        self.loop_active = True
        if root.attrib["type"] == "do":
            self.loop_index += 1
            do = {"tag": "do"}
            for node in root:
                if node.tag == "header":
                    do["header"] = self.parseTree(node, state)
                elif node.tag == "body":
                    do["body"] = self.parseTree(node, state)
                else:
                    assert (
                        False
                    ), f"Unrecognized tag in the process_loop for 'do' type." \
                        f"{node.tag}"
            self.loop_active = False
            return [do]
        elif root.attrib["type"] == "do-while":
            self.loop_index += 1
            do_while = {"tag": "do-while"}
            for node in root:
                if node.tag == "header":
                    do_while["header"] = self.parseTree(node, state)
                elif node.tag == "body":
                    do_while["body"] = self.parseTree(node, state)
            self.loop_active = False
            return [do_while]
        else:
            self.unhandled_tags.add(root.attrib["type"])
            return []

    def process_index_variable(self, root, state) -> List[Dict]:
        """ This function handles <index-variable> tag. This tag represents
        index ranges of loops or arrays. """

        assert (
            root.tag == "index-variable"
        ), f"The root must be <index-variable>. Current tag is {root.tag} " \
            f"with {root.attrib} attributes."
        ind = {"tag": "index", "name": root.attrib["name"].lower()}
        for bounds in root:
            if bounds.tag == "lower-bound":
                ind["low"] = self.parseTree(bounds, state)
            elif bounds.tag == "upper-bound":
                ind["high"] = self.parseTree(bounds, state)
            elif bounds.tag == "step":
                ind["step"] = self.parseTree(bounds, state)
        return [ind]

    def process_if(self, root, state) -> List[Dict]:
        """ This function handles <if> tag. Else and else if are nested under
        this tag. """
        assert (
            root.tag == "if"
        ), f"The root must be <if>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        ifs = []
        curIf = None
        for node in root:
            if node.tag == "header":
                if "type" not in node.attrib:
                    curIf = {"tag": "if"}
                    curIf["header"] = self.parseTree(node, state)
                    ifs.append(curIf)
                elif node.attrib["type"] == "else-if":
                    newIf = {"tag": "if"}
                    curIf["else"] = [newIf]
                    curIf = newIf
                    curIf["header"] = self.parseTree(node, state)
            elif node.tag == "body" and (
                "type" not in node.attrib or node.attrib["type"] != "else"
            ):
                curIf["body"] = self.parseTree(node, state)
            elif node.tag == "body" and node.attrib["type"] == "else":
                curIf["else"] = self.parseTree(node, state)
        return ifs

    def process_operation(self, root, state) -> List[Dict]:
        """ This function handles <operation> tag. The nested elements should
        either be "operand" or "operator". """

        assert (
            root.tag == "operation"
        ), f"The root must be <operation>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        op = {"tag": "op"}
        for node in root:
            if node.tag == "operand":
                if "left" in op:
                    op["right"] = self.parseTree(node, state)
                else:
                    op["left"] = self.parseTree(node, state)
            elif node.tag == "operator":
                if "operator" in op:
                    newOp = {
                        "tag": "op",
                        "operator": node.attrib["operator"],
                        "left": [op],
                    }
                    op = newOp
                else:
                    op["operator"] = node.attrib["operator"]
        return [op]

    def process_literal(self, root, _) -> List[Dict]:
        """ This function handles <literal> tag """
        assert (
            root.tag == "literal"
        ), f"The root must be <literal>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        for info in root:
            if info.tag == "pause-stmt":
                return [{"tag": "pause", "msg": root.attrib["value"]}]
            elif info.tag == "stop":
                text = root.attrib["value"]
                return [{"tag": "stop", "value": text}]
        return [
            {
                "tag": "literal",
                "type": root.attrib["type"],
                "value": root.attrib["value"],
            }
        ]

    def process_io_control(self, root, state) -> List[Dict]:
        """ This function checks for an asterisk in the argument of a
        read/write statement and stores it if found.  An asterisk in the first
        argument specifies a input through or output to console.  An asterisk
        in the second argument specifies a read/write without a format
        (implicit read/writes).  """

        assert (
            root.tag == "io-controls"
        ), f"The root must be <io-controls>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        io_control = []
        for node in root:
            if node.attrib["hasExpression"] == "true":
                assert (
                    "hasExpression" in node.attrib
                    and node.attrib["hasExpression"] == "true"
                ), "hasExpression is false. Something is wrong."
                io_control += self.parseTree(node, state)
            else:
                assert (
                    node.attrib["hasAsterisk"] == "true"
                ), "hasAsterisk is false. Something is wrong."
                io_control += [
                    {"tag": "literal", "type": "char", "value": "*"}
                ]
        return io_control

    def process_name(self, root, state) -> List[Dict]:
        """ This function handles <name> tag. The name tag will be added to the
        new AST for the pyTranslate.py with "ref" tag.  """

        assert (
            root.tag == "name"
        ), f"The root must be <name>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        if root.attrib["id"].lower() in self.libFns:
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        elif (
            root.attrib["id"].lower() in list(self.functionList.keys())
            # and state.subroutine["tag"] != "function"
        ):
            fn = {"tag": "call", "name": root.attrib["id"].lower(), "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        else:
            # numPartRef represents the number of references in the name.
            # Default = 1
            numPartRef = "1"
            # For example, numPartRef of x is 1 while numPartRef of
            # x.y is 2, etc.
            if "numPartRef" in root.attrib:
                numPartRef = root.attrib["numPartRef"]

            is_array = "false"
            if "is_array" in root.attrib:
                is_array = root.attrib["is_array"]

            ref = {
                "tag": "ref",
                "name": root.attrib["id"].lower(),
                "numPartRef": str(numPartRef),
                "hasSubscripts": root.attrib["hasSubscripts"],
                "is_array": is_array,
                "is_arg": "false",
                "is_parameter": "false",
                "is_interface_func": "false",
                "func_arg_types": []
            }
            # Check whether the passed element is for derived type reference
            if "is_derived_type_ref" in root.attrib:
                ref["is_derived_type_ref"] = "true"
            else:
                ref["is_derived_type_ref"] = "false"
            # Handling derived type references
            if int(numPartRef) > 1:
                for node in root:
                    if node.tag == "name":
                        nextRef = self.parseTree(node, state)
                        ref.update({"ref": nextRef})

            # Handling arrays
            if root.attrib["hasSubscripts"] == "true":
                for node in root:
                    if node.tag == "subscripts":
                        ref["subscripts"] = self.parseTree(node, state)

            for node in root:
                if node.tag == "argument-types":
                    ref["is_interface_func"] = "true"
                    ref["func_arg_types"] = self.parseTree(node, state)

            return [ref]

    def process_argument_types(self, root, _) -> List[Dict]:
        """This function handles <argument-types> tag that only appears
        under the interface function names. It will extract the argument
        types and add to the list, then return the list"""
        argument_types = []
        for node in root:
            argument_types.append(node.attrib["type"])
        return argument_types

    def process_assignment(self, root, state) -> List[Dict]:
        """ This function handles <assignment> tag that nested elements of
        <target> and <value>. """

        assert (
            root.tag == "assignment"
        ), f"The root must be <assignment>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        assign = {"tag": "assignment"}
        for node in root:
            if node.tag == "target":
                assign["target"] = self.parseTree(node, state)
            elif node.tag == "value":
                assign["value"] = self.parseTree(node, state)

        search_for_functions = False
        if len(assign["value"]) == 1 and assign["value"][0]["tag"] in ["op",
                                                                       "call"]:
            search_for_functions = True
        extra_tags = []

        # If the assignment is to the function/subroutine name,
        # then this is a return value. So, create a dummy variable and assign
        # the value to it. Then, return this dummy variable
        if (
            assign["target"][0]["name"]
            in list(self.functionList.keys())
        ) and (
            assign["target"][0]["name"] == state.subroutine["name"].lower()
        ):
            # Create the dummy variable name
            dummy_variable = f'{assign["target"][0]["name"]}_return'
            # We need to make sure that this dummy variable is not already
            # present in this module scope. If it is, throw and error for now
            check_list = []
            if self.variable_list.get(self.current_module):
                check_list = [x['name'] for x in self.variable_list[
                    self.current_module]]
            if self.argument_list.get(self.current_module):
                check_list += self.argument_list[self.current_module]
            if dummy_variable in check_list:
                assert False, "Return variable name is already present, " \
                              "choose a different name."
            else:
                return_type = self.functionList[assign["target"][0]["name"]][
                    'type']
                if return_type == "CHARACTER":
                    is_string = "true"
                else:
                    is_string = "false"
                # If the dummy variable is not present, then create a new
                # variable and then add it
                variable_spec = {
                    "type": return_type,
                    "is_derived_type": "false",
                    "keyword2": "none",
                    "is_string": is_string,
                    "name": dummy_variable,
                    "is_array": "false",
                    "tag": "variable"
                }
                assign["target"][0] = {
                    "tag": "ref",
                    "name": dummy_variable,
                    "numPartRef": "1",
                    "hasSubscripts": "false",
                    "is_array": "false",
                    "is_arg": "false",
                    "is_parameter": "false",
                    "is_interface_func": "false",
                    "func_arg_types": [],
                    "is_derived_type_ref": "false"
                }
                return_spec = {
                    "tag": "ret",
                    "name": dummy_variable,
                    "numPartRef": "1",
                    "hasSubscripts": "false",
                    "is_array": "false",
                    "is_arg": "false",
                    "is_parameter": "false",
                    "is_interface_func": "false",
                    "func_arg_types": [],
                    "is_derived_type_ref": "false"
                }
                if search_for_functions:
                    extra_tags = self.check_function_call(assign["value"])
            return extra_tags + [variable_spec, assign, return_spec]
        else:
            if search_for_functions:
                extra_tags = self.check_function_call(assign["value"])
            return extra_tags + [assign]

    def process_function(self, root, state) -> List[Dict]:
        """ This function handles <function> tag.  """
        assert (
            root.tag == "function"
        ), f"The root must be <function>. Current tag is {root.tag} with" \
            f"{root.attrib} attributes."
        subroutine = {"tag": root.tag, "name": root.attrib["name"].lower()}
        self.current_module = root.attrib["name"].lower()
        self.summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                args = self.parseTree(node, state)
                for arg in args:
                    arg["is_arg"] = "true"
                subroutine["args"] = args
            elif node.tag == "body":
                sub_state = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, sub_state)

        # Check if this subroutine had a save statement and if so, process
        # the saved node to add it to the ast
        if self.is_save:
            subroutine["body"] += self.process_save(self.saved_node, state)
            self.is_save = False
        elif self.saved_filehandle:
            subroutine["body"] += [{
                "tag": "save",
                "scope": self.current_module,
                "var_list": self.saved_filehandle
            }]
            self.saved_filehandle = []

        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_dimension(self, root, state) -> List[Dict]:
        """ This function handles <dimension> tag. This is a tag that holds
        information about the array, such as the range and values. """

        assert (
            root.tag == "dimension"
        ), f"The root must be <dimension>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        dimension = {}
        for node in root:
            if node.tag == "range":
                dimension["range"] = self.parseTree(node, state)
            if node.tag == "literal":
                dimension["literal"] = self.parseTree(node, state)
            if node.tag == "name":
                dimension_info = self.parseTree(node, state)
                dimension = dimension_info[0]
        dimension["tag"] = "dimension"
        return [dimension]

    def process_range(self, root, state) -> List[Dict]:
        """ This function handles <range> tag.  """

        assert (
            root.tag == "range"
        ), f"The root must be <range>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        ran = {}
        for node in root:
            if node.tag == "lower-bound":
                ran["low"] = self.parseTree(node, state)
            if node.tag == "upper-bound":
                ran["high"] = self.parseTree(node, state)
        return [ran]

    def process_keyword_argument(self, root, state) -> List[Dict]:
        """ This function handles <keyword-argument> tag. """
        assert (
            root.tag == "keyword-argument"
        ), f"The root must be <keyword-argument>. Current tag is {root.tag} " \
            f"with {root.attrib} attributes."
        x = []
        if root.attrib and root.attrib["argument-name"] != "":
            x = [{"arg_name": root.attrib["argument-name"]}]
        for node in root:
            x += self.parseTree(node, state)
        return x

    def process_libRtn(self, root, state) -> List[Dict]:
        fn = {"tag": "call", "name": root.tag, "args": []}
        for node in root:
            fn["args"] += self.parseTree(node, state)
        return [fn]

    def process_direct_map(self, root, state) -> List[Dict]:
        """Handles tags that are mapped directly from xml to IR with no
        additional processing other than recursive translation of any child
        nodes."""

        val = {"tag": root.tag, "args": []}
        for node in root:
            val["args"] += self.parseTree(node, state)

        # If the node is a file OPEN node, save it so that it can later be
        # added to the SAVE node in the ast
        if root.tag == "open":
            self.saved_filehandle += [val]
        return [val]

    def process_terminal(self, root, _) -> List[Dict]:
        """Handles tags that terminate the computation of a
        program unit, namely, "return", "stop", and "exit" """
        index = 0
        if root.tag == 'exit':
            self.break_index += 1
            index = self.break_index
            if self.loop_active:
                self.loop_constructs.setdefault(
                    f"loop", []).append(f"break_{self.break_index}")
        elif root.tag == "stop":
            self.return_index += 1
            index = self.return_index
            if self.loop_active:
                self.loop_constructs.setdefault(
                    f"loop", []).append(f"return_{self.return_index}")
        return [{"tag": root.tag, "index": index}]

    def process_format(self, root, state) -> List[Dict]:
        """ This function handles <format> tag. """

        assert (
            root.tag == "format"
        ), f"The root must be <format>. Current tag is {root.tag} with " \
            f"{root.attrib} attributes."
        format_spec = {"tag": "format", "args": []}
        for node in root:
            if node.tag == "label":
                format_spec["label"] = node.attrib["lbl"]
            format_spec["args"] += self.parseTree(node, state)
        return [format_spec]

    def process_format_item(self, root, _) -> List[Dict]:
        """ This function handles <format-item> tag. """

        assert root.tag == "format-item", "The root must be <format-item>"
        variable_spec = {
            "tag": "literal",
            "type": "char",
            "value": root.attrib["descOrDigit"],
        }
        return [variable_spec]

    def process_use(self, root, _) -> List[Dict]:
        """
            This function adds the tag for use statements
            In case of "USE .. ONLY .." statements, the symbols to be included
            are stored in the "include" field of the "use" block
        """

        tag_spec = {"tag": "use", "arg": root.attrib["name"]}
        for node in root:
            if node.tag == "only":
                tag_spec["include"] = []
                for item in node:
                    if item.tag == "name":
                        tag_spec["include"] += [item.attrib["id"]]

        return [tag_spec]
    
    def process_private_variable(self, root, _) -> List[Dict]:
        """ This function adds the tag for private symbols. Any
        variable/function being initialized as private is added in this tag.
        """
        for node in root:
            if node.tag == "name":
                return [{"tag": "private", "name": node.attrib["id"].lower()}]

        return []

    def process_save(self, root, _) -> List[Dict]:
        """
        This function parses the XML tag for the Fortran save statement and
        adds the tag that holds the function under which SAVE has been
        defined along with the variables that are saved by this statement.
        """

        # If is_save is False, the SAVE statement has been encountered for
        # the first time in the particular subroutine/function in context.
        # Here, change the flag value and save the SAVE node.
        if not self.is_save:
            self.is_save = True
            self.saved_node = root
            return []
        else:
            # This block will be entered when a SAVE statement is present
            # and its corresponding ast node has to be added at the end of
            # the subroutine/function body. Here the saved SAVE node
            # is processed as root.
            if root.attrib["hasSavedEntityList"] == "true":
                var_list = []
                for node in root:
                    for var in self.variable_list[self.current_module]:
                        if node.attrib["id"] == var["name"]:
                            var_list.append(var)
            else:
                var_list = self.variable_list[self.current_module]

            if self.saved_filehandle:
                var_list += self.saved_filehandle
            return [{"tag": "save", "scope": self.current_module, "var_list":
                    var_list}]

    def process_continue(self, root, _) -> List[Dict]:
        """This function handles cycle (continue in Python)
           tag."""
        self.cycle_index += 1
        if self.loop_active:
            self.loop_constructs.setdefault(
                f"loop", []).append(f"cycle_{self.cycle_index}")
        return [{"tag": root.tag, "index": self.cycle_index}]

    def process_select(self, root, state) -> List[Dict]:
        """This function handles select statements tag."""
        select_spec = {"tag": "select"}
        for node in root:
            if node.tag == "header":
                select_spec["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                select_spec["body"] = self.parseTree(node, state)

        return [select_spec]

    def process_case(self, root, state) -> List[Dict]:
        """This function handles the CASE statement in Fortran. This should
        be modeled as an if-else statement in languages like Python
        """
        case_spec = {"tag": "case"}
        for node in root:
            if node.tag == "header":
                for child in node:
                    if child.tag == "value-ranges":
                        case_spec["args"] = self.parseTree(child, state)
                    else:
                        assert False, f"Unhandled type {child.tag} in case"
            elif node.tag == "body":
                case_spec["body"] = self.parseTree(node, state)

        return [case_spec]

    def process_value_range(self, root, state) -> List[Dict]:
        """This function handles the range of values inside CASE statements"""
        value_range_spec = {"tag": "case_range", "args": []}
        for node in root:
            if node.tag == "value":
                value_range_spec["args"] += self.parseTree(node, state)

        return [value_range_spec]

    def process_interface(self, root, state) -> List[Dict]:
        """This function handles interface"""
        interface = {"tag": root.tag, "name": "", "functions": {},
                     "max_argument": root.attrib["max_arg"]}
        for node in root:
            if node.tag == "header":
                header = self.parseTree(node, state)
                interface["name"] = header[0]["name"]
            elif node.tag == "body":
                body = self.parseTree(node, state)
                for elem in body:
                    interface["functions"][elem["name"]] = elem[
                        "func_arg_types"]
            else:
                pass
        return [interface]

    def handle_data_statements(self, root, state):
        """
        This function handles the data statements that occurs in the
        declaration tag
        """
        # The main list of assignment inside a single data statement
        assignment_list = []
        tmp_assign = []
        current_var_count = None
        # Iterate over each node in the data statement
        for node in root:
            # The 'variable' tag must always come first, followed by the
            # `value` tag and then possible other `variable-value` pairs
            if node.tag == "variables":
                # Transfer everything from the previous `variable-value`
                # assignment into the main list
                if len(tmp_assign) > 0:
                    for item in tmp_assign:
                        assignment_list.append(item)
                    tmp_assign = []
                # Get the number of variables being assigned
                current_var_count = int(node.attrib["count"])
                # For every variable, create an assignment ast and fill it up
                # with the `tag` and `target` information
                for var in node:
                    assign = dict()
                    assign["tag"] = "assignment"
                    assign["target"] = self.parseTree(var, state)
                    tmp_assign.append(assign)
            # The `values` tag will come after the `variables` tag and assign
            # values to the respective variables
            elif node.tag == "values":
                # Get the number of values present
                current_value_count = int(node.attrib["count"])
                # If for every variable, there is a value assignment i.e.
                # one-to-one E.g. data x,y,z /1,2,3*2/ (z is an array)
                # TODO: Not handled -> data x(1) /2/ where x is an array of
                #  dimension > 1
                if current_value_count == current_var_count:
                    index = 0
                    for var in node:
                        target = tmp_assign[index]["target"][0]
                        # Check if this value assignment is for an array
                        if target["is_array"] == "true":
                            # Check if only one value is assigned or if it is
                            # a range of values using the '*' operator
                            if len(var) == 0:
                                if not tmp_assign[index]["target"][0].get(
                                        "subscripts"):
                                    tmp_assign[index]["target"][0][
                                        "subscripts"] = [
                                        {
                                            "tag": "literal",
                                            "type": "int",
                                            "value": "1"
                                        }]
                                    tmp_assign[index]["target"][0][
                                        "hasSubscripts"] = "true"
                                tmp_assign[index]["value"] = \
                                    self.parseTree(var, state)
                            else:
                                # If a single array is assigned multiple same
                                # values using an '*' operator, create a
                                # do-while loop to assign each index
                                variable_name = target["name"]
                                for var_name in self.variable_list[
                                                self.current_module]:
                                    if var_name["name"] == variable_name:
                                        if len(var_name["dimensions"]) == 1:
                                            dimension = (int(var_name[
                                                            "dimensions"][0][
                                                            "literal"][0][
                                                            "value"]))
                                        else:
                                            dimension = (int(var_name[
                                                            "dimensions"][0][
                                                            "literal"][0][
                                                            "value"]),
                                                         int(var_name[
                                                            "dimensions"][1][
                                                            "literal"][0][
                                                            "value"]))
                                if len(dimension) == 1:
                                    array_ast = self.create_1d_array_ast(
                                        var,
                                        tmp_assign[index],
                                        state)
                                else:
                                    array_ast = self.create_2d_array_ast(
                                        var,
                                        tmp_assign[index],
                                        dimension,
                                        state)
                                if len(array_ast) == 1:
                                    tmp_assign[index] = array_ast[0]
                                else:
                                    tmp_assign = tmp_assign[:index] \
                                                 + array_ast \
                                                 + tmp_assign[index+1:]
                                    index += 1
                        else:
                            # For every respective variable, assign the `value`
                            # information into the AST
                            tmp_assign[index]["value"] = \
                                self.parseTree(var, state)
                        index += 1
                else:
                    # If the number of values is more than the number of
                    # variables, the variable assignment includes an array
                    # assignment of the form: DATA X /1,2,3,4/ where X has a
                    # dimension of 4
                    value_index = 0
                    loop_limit = 0
                    array_assign = []
                    for variable in tmp_assign:
                        variable_name = variable["target"][0]["name"]
                        is_array = variable["target"][0]["is_array"]
                        if is_array == "true":
                            for var in self.variable_list[self.current_module]:
                                if var["name"] == variable_name:
                                    # This is very hard-coded. What other
                                    # kinds of dimensions are present other
                                    # than in literal forms?
                                    if len(var["dimensions"]) == 1:
                                        dimension = [int(var["dimensions"][0][
                                                             "literal"][0][
                                                             "value"])]
                                    else:
                                        dimension = [int(var["dimensions"][0][
                                                             "literal"][0][
                                                             "value"]),
                                                     int(var["dimensions"][1][
                                                             "literal"][0][
                                                             "value"])]
                            arr_index = 0
                            if len(dimension) > 1:
                                two_dim_arr = True
                                row_count = dimension[0]
                                column_count = dimension[1]
                                current_row = 1
                                current_column = 1
                            else:
                                two_dim_arr = False
                            while True:
                                if two_dim_arr:
                                    if arr_index >= row_count * column_count:
                                        break
                                else:
                                    if arr_index >= dimension[0]:
                                        break
                                arr_target = copy.deepcopy(variable)
                                if two_dim_arr:
                                    arr_target["target"][0]["subscripts"] = [
                                        {
                                            "tag": "literal",
                                            "type": "int",
                                            "value": str(current_row)
                                        },
                                        {
                                            "tag": "literal",
                                            "type": "int",
                                            "value": str(current_column)
                                        }
                                    ]
                                else:
                                    arr_target["target"][0]["subscripts"] = [
                                        {
                                            "tag": "literal",
                                            "type": "int",
                                            "value": str(arr_index + 1)
                                        }]
                                arr_target["target"][0]["hasSubscripts"] = \
                                    "true"
                                if len(node[value_index]) == 0:
                                    arr_target["value"] = \
                                        self.parseTree(node[value_index], state)
                                    array_assign.append(arr_target)
                                    value_index += 1
                                else:
                                    if loop_limit == 0:
                                        loop_limit = \
                                            int(node[value_index].attrib[
                                                    "value"])
                                    arr_target["value"] = \
                                        self.parseTree(node[value_index][0],
                                                       state)
                                    array_assign.append(arr_target)
                                    loop_limit -= 1
                                    if loop_limit == 0:
                                        value_index += 1
                                if two_dim_arr:
                                    if current_row == row_count:
                                        current_row = 1
                                        current_column += 1
                                    else:
                                        current_row += 1
                                arr_index += 1
                        else:
                            if len(node[value_index]) == 0:
                                variable["value"] = \
                                    self.parseTree(node[value_index], state)
                                array_assign.append(variable)
                                value_index += 1
                            else:
                                if loop_limit == 0:
                                    loop_limit = int(node[value_index].attrib[
                                                         "value"])
                                variable["value"] = \
                                    self.parseTree(node[value_index][0], state)
                                array_assign.append(variable)
                                loop_limit -= 1
                                if loop_limit == 0:
                                    value_index += 1
                    tmp_assign = array_assign

        for item in tmp_assign:
            assignment_list.append(item)

        return assignment_list

    def create_1d_array_ast(self, root, assign_ast, state):
        """
        This function creates the do-while loop ast which assigns values to
        a one-dimensional array according to the data statement operation
        """
        # First, we need a variable for the iteration. Check if an integer
        # variable 'iterator' has already been defined. If yes, use it,
        # else define it
        array_ast = []
        iterator_ast = {
            "type": "integer",
            "is_derived_type": "false",
            "keyword2": "none",
            "is_string": "false",
            "name": "iterator",
            "is_array": "false",
            "tag": "variable"
        }
        if iterator_ast not in self.variable_list[self.current_module]:
            array_ast.append(iterator_ast)

        # Now, define the do-while loop
        do_ast = dict()
        do_ast["tag"] = "do"
        do_ast["header"] = [{
            "tag": "index",
            "name": "iterator",
            "low": [{
                "tag": "literal",
                "type": "int",
                "value": "1"
            }],
            "high": [{
                "tag": "literal",
                "type": "int",
                "value": root.attrib["value"]
            }]
            }
        ]
        if not assign_ast["target"][0].get("subscripts"):
            assign_ast["target"][0]["subscripts"] = [
                {
                    "tag": "ref",
                    "name": "iterator",
                    "numPartRef": "1",
                    "hasSubscripts": "false",
                    "is_array": "false",
                    "is_arg": "false",
                    "is_parameter": "false",
                    "is_interface_func": "false",
                    "func_arg_types": [],
                    "is_derived_type_ref": "false"
                }]
            assign_ast["target"][0]["hasSubscripts"] = "true"
        assign_ast["value"] = self.parseTree(root[0], state)
        do_ast["body"] = [assign_ast]

        array_ast.append(do_ast)
        return array_ast

    def create_2d_array_ast(self, root, assign_ast, dimension, state):
        """
        This function creates the do-while loop ast which assigns values to a
        two-dimensional array according to the data statement operation
        """
        # First, we need a variable for the iteration. Check if an integer
        # variable 'i_iterator' has already been defined. If yes, use it,
        # else define it. Do the same for 'j_iterator'
        array_ast = []
        i_iterator_ast = {
            "type": "integer",
            "is_derived_type": "false",
            "keyword2": "none",
            "is_string": "false",
            "name": "i_iterator",
            "is_array": "false",
            "tag": "variable"
        }
        if i_iterator_ast not in self.variable_list[self.current_module]:
            array_ast.append(i_iterator_ast)

        j_iterator_ast = {
            "type": "integer",
            "is_derived_type": "false",
            "keyword2": "none",
            "is_string": "false",
            "name": "j_iterator",
            "is_array": "false",
            "tag": "variable"
        }
        if j_iterator_ast not in self.variable_list[self.current_module]:
            array_ast.append(j_iterator_ast)

        # Now, define the inner do-while loop first
        inner_do_ast = dict()
        inner_do_ast["tag"] = "do"
        inner_do_ast["header"] = [{
            "tag": "index",
            "name": "j_iterator",
            "low": [{
                "tag": "literal",
                "type": "int",
                "value": "1"
            }],
            "high": [{
                "tag": "literal",
                "type": "int",
                "value": str(dimension[1])
            }]
        }
        ]
        if not assign_ast["target"][0].get("subscripts"):
            assign_ast["target"][0]["subscripts"] = [
                {
                    "tag": "ref",
                    "name": "i_iterator",
                    "numPartRef": "1",
                    "hasSubscripts": "false",
                    "is_array": "false",
                    "is_arg": "false",
                    "is_parameter": "false",
                    "is_interface_func": "false",
                    "func_arg_types": [],
                    "is_derived_type_ref": "false"
                },
                {
                    "tag": "ref",
                    "name": "j_iterator",
                    "numPartRef": "1",
                    "hasSubscripts": "false",
                    "is_array": "false",
                    "is_arg": "false",
                    "is_parameter": "false",
                    "is_interface_func": "false",
                    "func_arg_types": [],
                    "is_derived_type_ref": "false"
                }
            ]
            assign_ast["target"][0]["hasSubscripts"] = "true"
        assign_ast["value"] = self.parseTree(root[0], state)
        inner_do_ast["body"] = [assign_ast]

        # Now the outer do-while loop
        outer_do_ast = dict()
        outer_do_ast["tag"] = "do"
        outer_do_ast["header"] = [{
            "tag": "index",
            "name": "i_iterator",
            "low": [{
                "tag": "literal",
                "type": "int",
                "value": "1"
            }],
            "high": [{
                "tag": "literal",
                "type": "int",
                "value": str(dimension[0])
            }]
        }
        ]
        outer_do_ast["body"] = [inner_do_ast]
        array_ast.append(outer_do_ast)
        return array_ast

    def check_function_call(self, value):
        """
            This function checks whether there is a function call in the
            value of an assignment. If there is one, remove the function
            call into a separate assignment
        """
        extra_tags = []
        if value[0]["tag"] == "op":
            if value[0]["left"][0]["tag"] == "op":
                extra_tags += self.check_function_call(value[0]["left"])
            elif value[0]["left"][0]["tag"] == "call":
                extra_tags += self.initiate_function_replacement(value[0][
                                                                     "left"])[0]

            if value[0]["right"][0]["tag"] == "op":
                extra_tags += self.check_function_call(value[0]["right"])
            elif value[0]["right"][0]["tag"] == "call":
                extra_tags += self.initiate_function_replacement(value[0][
                                                                     "right"])[0]
        elif value[0]["tag"] == "call":
            extra_tags += self.initiate_function_replacement(value[0][
                                                                 "args"])[0]

        return extra_tags

    def initiate_function_replacement(self, function_tag):
        tags = []
        function_name = function_tag[0]["name"]
        if function_name.lower() in self.functionList:
            function_arguments = function_tag[0]["args"]
            for index, arg in enumerate(function_arguments):
                if arg["tag"] == "call":
                    results = self.initiate_function_replacement([arg])
                    tags += results[0]
                    if isinstance(results[1], dict):
                        function_tag[0]["args"][index] = results[1]
                    else:
                        function_tag[0]["args"][index] = results[1][0]
            tags += self.replace_function_call(function_tag, function_name)
        return tags, function_tag

    def replace_function_call(self, tag, function_name):
        call_spec = copy.deepcopy(tag[0])
        self.functionList[function_name.lower()]['call_count'] += 1
        call_count = self.functionList[function_name.lower()][
            'call_count']
        return_type = self.functionList[function_name.lower()]['type']
        if return_type == "CHARACTER":
            is_string = "true"
        else:
            is_string = "false"
        call_var = {
            "type": return_type,
            "is_derived_type": "false",
            "keyword2": "none",
            "is_string": is_string,
            "name": f"{function_name}__{call_count}",
            "is_array": "false",
            "tag": "variable"
        }
        target_var = {
            "tag": "ref",
            "name": f"{function_name}__{call_count}",
            "numPartRef": "1",
            "hasSubscripts": "false",
            "is_array": "false",
            "is_arg": "false",
            "is_parameter": "false",
            "is_interface_func": "false",
            "func_arg_types": [],
            "is_derived_type_ref": "false"
        }
        assignment_tag = {
            "tag": "assignment",
            "target": [target_var],
            "value": [call_spec],
        }
        tag[0] = target_var
        extra_tags = [call_var, assignment_tag]

        return extra_tags

    def parseTree(self, root, state: ParseState) -> List[Dict]:
        """
        Parses the XML ast tree recursively to generate a JSON AST
        which can be ingested by other scripts to generate Python
        scripts.

        Args:
            root: The current root of the tree.
            state: The current state of the tree defined by an object of the
                ParseState class.

        Returns:
                ast: A JSON ast that defines the structure of the Fortran file.
        """
        if root.tag in self.ast_tag_handlers:
            return self.ast_tag_handlers[root.tag](root, state)

        elif root.tag in self.libRtns:
            return self.process_libRtn(root, state)

        else:
            prog = []
            for node in root:
                prog += self.parseTree(node, state)
            return prog

    def loadFunction(self, root):
        """
        Loads a list with all the functions in the Fortran File

        Args:
            root: The root of the XML ast tree.

        Returns:
            None

        Does not return anything but populates two lists (self.functionList
        and self.subroutineList) that contains all the functions and
        subroutines in the Fortran File respectively.
        """
        return_type = None
        for element in root.iter():
            if element.tag == "declaration" and element[0].tag == "type":
                return_type = element[0].attrib["name"]
            if element.tag == "function":
                self.functionList[element.attrib["name"].lower()] = {
                    'type': return_type,
                    'call_count': 0
                }
            elif element.tag == "subroutine":
                self.subroutineList.append(element.attrib["name"])

    def analyze(self, trees: List[ET.ElementTree]) -> Dict:
        outputDict = {}
        ast = []

        # Parse through the ast once to identify and grab all the functions
        # present in the Fortran file.
        for tree in trees:
            self.loadFunction(tree)

        # Parse through the ast tree a second time to convert the XML ast
        # format to a format that can be used to generate Python statements.
        for tree in trees:
            ast += self.parseTree(tree, ParseState())

        # print(ast)

        """
        Find the entry point for the Fortran file.
        The entry point for a conventional Fortran file is always the PROGRAM
        section. This 'if' statement checks for the presence of a PROGRAM
        segment.

        If not found, the entry point can be any of the functions or
        subroutines in the file. So, all the functions and subroutines of the
        program are listed and included as the possible entry point.
        """
        if self.entryPoint:
            entry = {"program": self.entryPoint[0]}
        else:
            entry = {}
            if self.functionList:
                entry["function"] = list(self.functionList.keys())
            if self.subroutineList:
                entry["subroutine"] = self.subroutineList

        # Load the functions list and Fortran ast to a single data structure
        # which can be pickled and hence is portable across various scripts and
        # usages.
        outputDict["ast"] = ast
        outputDict["functionList"] = list(self.functionList.keys())
        return outputDict

    def print_unhandled_tags(self):
        if self.unhandled_tags != set():
            sys.stderr.write(
                "WARNING: input contains the following unhandled tags:\n"
            )
            for tag in self.unhandled_tags:
                sys.stderr.write(f"    {tag}\n")


def get_trees(files: List[str]) -> List[ET.ElementTree]:
    return [ET.parse(f).getroot() for f in files]


def xml_to_py(trees, fortran_file):
    translator = XML_to_JSON_translator()
    output_dict = translator.analyze(trees)

    # Only go through with the handling of breaks and returns if they are
    # actually there
    if len(translator.loop_constructs) > 0:
        refactor_breaks = RefactorConstructs()
        output_dict = refactor_breaks.refactor(output_dict,
                                               translator.loop_constructs)

    comments = get_comments(fortran_file)
    output_dict["comments"] = comments

    # print_unhandled_tags() was originally intended to alert us to program
    # constructs we were not handling.  It isn't clear we actually use this
    # so I'm commenting out this call for now.  Eventually this code (and all 
    # the code that keeps track of unhandled tags) should go away.
    # --SKD 06/2019
    # translator.print_unhandled_tags()

    return output_dict


def parse_args():
    """ Parse the arguments passed to the script.  Returns a tuple 
        (fortran_file, pickle_file, args) where fortran_file is the
        file containing the input Fortran code, and pickle_file is
        the output pickle file.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        nargs="*",
        help="Pickled version of routines for which dependency graphs should "
             "be generated",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="A list of AST files in XML format to analyze",
    )
    parser.add_argument(
        "-i", "--input", nargs="*", help="Original Fortran Source code file."
    )

    args = parser.parse_args(sys.argv[1:])
    fortran_file = args.input[0]
    pickle_file = args.gen[0]

    return fortran_file, pickle_file, args


def gen_pickle_file(output_dictionary, pickle_filename):
    with open(pickle_filename, "wb") as f:
        pickle.dump(output_dictionary, f)


if __name__ == "__main__":
    (fortran_file, pickle_file, args) = parse_args()
    trees = get_trees(args.files)

    output_dict = xml_to_py(trees, fortran_file)

    gen_pickle_file(output_dict, pickle_file)
