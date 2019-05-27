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

ast_file: The XML represenatation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

pickle_file: The file which will contain the pickled version of JSON AST and
supporting information. """


import sys
import argparse
import pickle
import xml.etree.ElementTree as ET
from typing import List, Dict
from collections import OrderedDict
from delphi.translators.for2py.get_comments import get_comments


class ParseState(object):
    """This class defines the state of the XML tree parsing
    at any given root. For any level of the tree, it stores
    the subroutine under which it resides along with the
    subroutines arguments."""

    def __init__(self, subroutine=None):
        self.subroutine = subroutine if subroutine is not None else {}
        self.args = (
            [arg["name"] for arg in self.subroutine["args"]]
            if "args" in self.subroutine
            else []
        )

    def copy(self, subroutine=None):
        return ParseState(
            self.subroutine if subroutine is None else subroutine
        )


class XMLToJSONTranslator(object):
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
            "write": self.process_direct_map,
            "derived-types": self.process_derived_types,
            "length": self.process_length,
        }

        self.unhandled_tags = set()  # unhandled xml tags in the current input
        self.summaries = {}
        self.asts = {}
        self.functionList = []
        self.subroutineList = []
        self.entryPoint = []

    def process_subroutine_or_program_module(self, root, state):
        """ This function should be the very first function to be called """
        subroutine = {"tag": root.tag, "name": root.attrib["name"].lower()}
        self.summaries[root.attrib["name"]] = None
        if root.tag == "subroutine":
            self.subroutineList.append(root.attrib["name"])
        else:
            self.entryPoint.append(root.attrib["name"])
        for node in root:
            if node.tag == "header":
                subroutine["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, subState)
            elif node.tag == "members":
                subroutine["body"] += self.parseTree(node, subState)
        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_call(self, root, state) -> List[Dict]:
        """ This function handles <call> tag and its subelement <name>. """
        assert (
            root.tag == "call"
        ), f"The root must be <call>. Current tag is {root.tag} with {root.attrib} attributes."
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
        return [{"tag": "arg", "name": root.attrib["name"].lower()}]

    def process_declaration(self, root, state) -> List[Dict]:
        """ This function handles <declaration> tag and its sub-elements by
        recursively calling the appropriate functions for the target tag. """

        declared_type = []
        declared_variable = []
        assert (
            root.tag == "declaration"
        ), f"The root must be <declaration>. Current tag is {root.tag} with {root.attrib} attributes."
        for node in root:
            if node.tag not in self.handled_tags:
                self.unhandled_tags.add(node.tag)
            elif node.tag == "type":  # Get the variable type
                if root.attrib["type"] == "variable":
                    declared_type += self.parseTree(node, state)
                else:
                    # If the current node is for declaring a derived type,
                    # every step from type declaration to variable (including
                    # array) declration will be done in the
                    # "process_derived_types" function and return the completed
                    # AST list object back.  Thus, simply insert the received
                    # AST list object into the declared_variable object. No
                    # other work is done in the current function.
                    declared_variable += self.parseTree(node, state)
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
                for index in range(int(node.attrib["count"])):
                    if len(declared_type) > 0:
                        combined = declared_type[-1]
                        combined.update(variables[index])
                        declared_variable.append(combined.copy())
                        if (
                            state.subroutine["name"] in self.functionList
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
        return declared_variable

    def process_type(self, root, state) -> List[Dict]:
        """ This function handles <type> declaration.
        There may be two different cases of <type>.
            (1) Simple variable type declaration
            (2) Derived type declaration
        """

        assert (
            root.tag == "type"
        ), f"The root must be <type>. Current tag is {root.tag} with {root.attrib} attributes."
        declared_type = {}
        derived_type = []
        if (
            root.text
        ):  # Check if the <type> has sub-elements, which is the case of (2)
            for node in root:
                if node.tag == "type":
                    derived_type += self.parseTree(node, state)
                elif node.tag == "length":
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
        else:  # Else, this represents an empty element, which is the case of (1).
            declared_type = {
                "type": root.attrib["name"],
                "is_derived_type": root.attrib["is_derived_type"].lower(),
                "keyword2": root.attrib["keyword2"],
            }
            return [declared_type]

    def process_length(self, root, state) -> List[Dict]:
        """ This function handles <length> tag.  """
        assert (
            root.tag == "length"
        ), f"The root must be <length>. Current tag is {root.tag} with {root.attrib} attributes."
        length = {}
        for node in root:
            if node.tag == "literal":
                length.update(self.parseTree(node, state)[-1])
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
            ), f"The root must be <variables>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <variable>. Current tag is {root.tag} with {root.attrib} attributes."
        try:
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
            return [variable]
        except:
            return []

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
        ), f"The root must be <derived-type>. Current tag is {root.tag} with {root.attrib} attributes."
        derived_types = {"derived-types": []}
        declared_type = []
        declared_variable = []
        for node in root:
            if node.tag not in self.handled_tags:
                self.unhandled_tags.add(node.tag)
            elif node.tag == "type":  # Get the variable type
                declared_type += self.parseTree(node, state)
            elif node.tag == "dimensions":
                dimensions = {
                    "count": node.attrib["count"],
                    "dimensions": [{"tag": "dimension"}],
                }
                dimensions["dimensions"][0].update(
                    self.parseTree(node, state)[-1]
                )
                declared_type[-1].update(dimensions)
            elif node.tag == "variables":
                variables = self.parseTree(node, state)
                # declare variables based on the counts to handle the case where a multiple vars declared under a single type
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
        ), f"The root must be <loop>. Current tag is {root.tag} with {root.attrib} attributes."
        if root.attrib["type"] == "do":
            do = {"tag": "do"}
            for node in root:
                if node.tag == "header":
                    do["header"] = self.parseTree(node, state)
                elif node.tag == "body":
                    do["body"] = self.parseTree(node, state)
                else:
                    assert (
                        False
                    ), f"Unrecognized tag in the process_loop for 'do' type. {node.tag}"
            return [do]
        elif root.attrib["type"] == "do-while":
            doWhile = {"tag": "do-while"}
            for node in root:
                if node.tag == "header":
                    doWhile["header"] = self.parseTree(node, state)
                elif node.tag == "body":
                    doWhile["body"] = self.parseTree(node, state)
            return [doWhile]
        else:
            self.unhandled_tags.add(root.attrib["type"])
            return []

    def process_index_variable(self, root, state) -> List[Dict]:
        """ This function handles <index-variable> tag. This tag represents
        index ranges of loops or arrays. """

        assert (
            root.tag == "index-variable"
        ), f"The root must be <index-variable>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <if>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <operation>. Current tag is {root.tag} with {root.attrib} attributes."
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

    def process_literal(self, root, state) -> List[Dict]:
        """ This function handles <literal> tag """
        assert (
            root.tag == "literal"
        ), f"The root must be <literal>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <io-controls>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <name>. Current tag is {root.tag} with {root.attrib} attributes."
        if root.attrib["id"].lower() in self.libFns:
            fn = {"tag": "call", "name": root.attrib["id"], "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        elif (
            root.attrib["id"] in self.functionList
            and state.subroutine["tag"] != "function"
        ):
            fn = {"tag": "call", "name": root.attrib["id"].lower(), "args": []}
            for node in root:
                fn["args"] += self.parseTree(node, state)
            return [fn]
        else:
            # numPartRef represents the number of references in the name. Default = 1
            numPartRef = "1"
            # For example, numPartRef of x is 1 while numPartRef of x.y is 2, etc.
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

            return [ref]

    def process_assignment(self, root, state) -> List[Dict]:
        """ This function handles <assignment> tag that nested elements of
        <target> and <value>. """

        assert (
            root.tag == "assignment"
        ), f"The root must be <assignment>. Current tag is {root.tag} with {root.attrib} attributes."
        assign = {"tag": "assignment"}
        devTypeAssignment = False
        for node in root:
            if node.tag == "target":
                assign["target"] = self.parseTree(node, state)
            elif node.tag == "value":
                assign["value"] = self.parseTree(node, state)

        if (
            assign["target"][0]["name"]
            in [x.lower() for x in self.functionList]
        ) and (
            assign["target"][0]["name"] == state.subroutine["name"].lower()
        ):
            assign["value"][0]["tag"] = "ret"
            return assign["value"]
        else:
            return [assign]

    def process_function(self, root, state) -> List[Dict]:
        """ This function handles <function> tag.  """
        assert (
            root.tag == "function"
        ), f"The root must be <function>. Current tag is {root.tag} with {root.attrib} attributes."
        subroutine = {"tag": root.tag, "name": root.attrib["name"].lower()}
        self.summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                args = self.parseTree(node, state)
                for arg in args:
                    arg["is_arg"] = "true"
                subroutine["args"] = args
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, subState)
        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_dimension(self, root, state) -> List[Dict]:
        """ This function handles <dimension> tag. This is a tag that holds
        information about the array, such as the range and values. """

        assert (
            root.tag == "dimension"
        ), f"The root must be <dimension>. Current tag is {root.tag} with {root.attrib} attributes."
        dimension = {"tag": "dimension"}
        for node in root:
            if node.tag == "range":
                dimension["range"] = self.parseTree(node, state)
            if node.tag == "literal":
                dimension["literal"] = self.parseTree(node, state)
        return [dimension]

    def process_range(self, root, state) -> List[Dict]:
        """ This function handles <range> tag.  """

        assert (
            root.tag == "range"
        ), f"The root must be <range>. Current tag is {root.tag} with {root.attrib} attributes."
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
        ), f"The root must be <keyword-argument>. Current tag is {root.tag} with {root.attrib} attributes."
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
        return [val]

    def process_terminal(self, root, state) -> List[Dict]:
        """Handles tags that terminate the computation of a
        program unit, namely, "return", "stop", and "exit" """

        return [{"tag": root.tag}]

    def process_format(self, root, state) -> List[Dict]:
        """ This function handles <format> tag. """

        assert (
            root.tag == "format"
        ), f"The root must be <format>. Current tag is {root.tag} with {root.attrib} attributes."
        format_spec = {"tag": "format", "args": []}
        for node in root:
            if node.tag == "label":
                format_spec["label"] = node.attrib["lbl"]
            format_spec["args"] += self.parseTree(node, state)
        return [format_spec]

    def process_format_item(self, root, state) -> List[Dict]:
        """ This function handles <format-item> tag. """

        assert root.tag == "format-item", "The root must be <format-item>"
        variable_spec = {
            "tag": "literal",
            "type": "char",
            "value": root.attrib["descOrDigit"],
        }
        return [variable_spec]

    def process_use(self, root, state) -> List[Dict]:
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

    def process_private_variable(self, root, state) -> List[Dict]:
        """ This function adds the tag for private symbols. Any
        variable/function being initialized as private is added in this tag.
        """
        for node in root:
            if node.tag == "name":
                return [{"tag": "private", "name": node.attrib["id"].lower()}]

        return []

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

        Does not return anything but populates a list (self.functionList) that
        contains all the functions in the Fortran File.
        """
        for element in root.iter():
            if element.tag == "function":
                self.functionList.append(element.attrib["name"])

    def analyze(
        self, trees: List[ET.ElementTree], comments: OrderedDict
    ) -> Dict:
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
                entry["function"] = self.functionList
            if self.subroutineList:
                entry["subroutine"] = self.subroutineList

        # Load the functions list and Fortran ast to a single data structure
        # which can be pickled and hence is portable across various scripts and
        # usages.
        outputDict["ast"] = ast
        outputDict["functionList"] = self.functionList
        outputDict["comments"] = comments
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        nargs="*",
        help="Pickled version of routines for which dependency graphs should be generated",
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
    fortranFile = args.input[0]
    pickleFile = args.gen[0]

    trees = get_trees(args.files)
    comments = get_comments(fortranFile)
    translator = XMLToJSONTranslator()
    outputDict = translator.analyze(trees, comments)
    translator.print_unhandled_tags()

    with open(pickleFile, "wb") as f:
        pickle.dump(outputDict, f)
