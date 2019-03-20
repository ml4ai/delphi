"""
This script converts the XML version of AST of the Fortran
file into a JSON representation of the AST along with other
non-source code information. The output is a pickled file
which contains this information in a parsable data structure.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        python translate.py -f <ast_file> -g <pickle_file>

ast_file: The XML represenatation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

pickle_file: The file which will contain the pickled version of JSON AST and
supporting information.

"""


import sys
import argparse
import pickle
import re
import xml.etree.ElementTree as ET
from delphi.translators.for2py.get_comments import get_comments
from typing import List, Dict
from collections import OrderedDict


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
            "argument", "assignment", "call", "close", "component-decl",
            "declaration", "dimension", "dimensions",
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
            "loop", "module", "name", "open", "operation", "program",
            "range", "read", "return",
            "stop", "subroutine",
            "type",
            "use", "variable", "variables",
            "write"
        ]
        self.handled_tags += self.libRtns

        self.AST_TAG_HANDLERS = {
            "argument": self.process_argument,
            "assignment": self.process_assignment,
            "call": self.process_call,
            "close" : self.process_direct_map,
            "declaration": self.process_declaration,
            "dimension": self.process_dimension,
            "exit" : self.process_terminal,
            "format-item": self.process_format_item,
            "format": self.process_format,
            "function": self.process_function,
            "if": self.process_if,
            "index-variable": self.process_index_variable,
            "io-control-spec": self.process_io_control_spec,
            "keyword-argument": self.process_keyword_argument,
            "literal": self.process_literal,
            "loop" : self.process_loop,
            "module" : self.process_subroutine_or_program_module,
            "name": self.process_name,
            "open" : self.process_direct_map,
            "operation": self.process_operation,
            "program" : self.process_subroutine_or_program_module,
            "range": self.process_range,
            "read" : self.process_direct_map,
            "return" : self.process_terminal,
            "stop" : self.process_terminal,
            "subroutine": self.process_subroutine_or_program_module,
            "type": self.process_type,
            "use": self.process_use,
            "variable": self.process_variable,
            "write" : self.process_direct_map
        }

        self.unhandled_tags = set()  # unhandled xml tags in the current input
        self.summaries = {}
        self.asts = {}
        self.functionList = []
        self.subroutineList = []
        self.entryPoint = []
        self.deriveTypeVars = []
        # The purpose of this global dictionary is to track the declared field
        # variables and mark it's an array or not. For example, var.a.get(1)
        # will be makred as hasSubscripts = False and arrayStat = "isArray".
        # Thus, hasSubscripts represents the array existence of the very first
        # variable and arrayStat represents the array existence of the
        # following fields.
        self.deriveTypeFields = {}
        self.arrays = []

    def process_subroutine_or_program_module(self, root, state):
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
        call = {"tag": "call"}
        for node in root:
            if node.tag == "name":
                call["name"] = node.attrib["id"].lower()
                call["args"] = []
                for arg in node:
                    call["args"] += self.parseTree(arg, state)
        return [call]

    def process_argument(self, root, state) -> List[Dict]:
        return [{"tag": "arg", "name": root.attrib["name"].lower()}]

    # *** This method is in desperate need of cleanup.
    # *** TODO after Apr deadline.
    def process_declaration(self, root, state) -> List[Dict]:
        prog = []

        decVars = []
        decDims = []

        # For handling derived types
        decDevType = []
        decDevFields = []
        decDevTypeVars = []

        devTypeArrayField = {}
        decType = {}

        count = 0

        isArray = True
        isDevType = False
        isDevTypeVar = False
        devTypeHasArrayField = False

        for node in root:
            if node.tag not in self.handled_tags:
                self.unhandled_tags.add(node.tag)

            if node.tag == "format":
                prog += self.parseTree(node, state)
            if node.tag == "type":
                if "name" in node.attrib:
                    decType = {"type": node.attrib["name"]}
                    decDevType += self.parseTree(
                        node, state
                    )  # For derived types
                    if len(decDevType) > 1:
                        isDevType = True
                else:
                    # This is the case where declaring fields for the derived
                    # type
                    if node[0].tag == "derived-type-spec":
                        decType = {"type": self.parseTree(node, state)}
                        isDevTypeVar = True
            elif node.tag == "variables":
                decVars = self.parseTree(node, state)
                isArray = False
                # This is a book keeping of derived type variables
                if isDevTypeVar:
                    for i in range(0, len(decVars)):
                        devTypevarName = decVars[i]["name"]
                        self.deriveTypeVars.append(devTypevarName)
            elif node.tag == "access-spec":
                if node.attrib["keyword"] == "PRIVATE":
                    decVars = self.process_private_variable(root, state)
            elif node.tag == "dimensions":
                decDims = self.parseTree(node, state)
                count = node.attrib["count"]
            elif node.tag == "explicit-shape-spec-list__begin":
                # Check if the last derived type declaration field is an array
                # field
                devTypeHasArrayField = True
            elif node.tag == "literal" and devTypeHasArrayField:
                # If the last field is an array field, get the value from the
                # literal that is size
                devTypeArrayField["array-size"] = node.attrib["value"]
            elif node.tag == "component-decl":
                if not devTypeHasArrayField:
                    decDevType.append({"field-id": node.attrib["id"].lower()})
                    self.deriveTypeFields[
                        node.attrib["id"].lower()
                    ] = "notArray"
                else:
                    devTypeArrayField["field-id"] = node.attrib["id"].lower()
                    self.deriveTypeFields[
                        node.attrib["id"].lower()
                    ] = "isArray"
                    decDevType.append(devTypeArrayField)
                    devTypeHasArrayField = False

        for var in decVars:
            if (
                state.subroutine["name"] in self.functionList
                and var["name"] in state.args
            ):
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]
                continue
            prog.append(decType.copy())
            prog[-1].update(var)
            if var["name"] in state.args:
                state.subroutine["args"][state.args.index(var["name"])][
                    "type"
                ] = decType["type"]

        # If the statement is for declaring array
        if decDims:
            for i in range(0, len(prog)):
                if prog[i]["name"] not in self.arrays: # Checks if a name exists in the array tracker. If not, add it for later use
                    self.arrays.append(prog[i]["name"])
                counter = 0
                for dim in decDims:
                    if "literal" in dim:
                        for lit in dim["literal"]:
                            prog[i]["tag"] = "array"
                            prog[i]["count"] = count
                            prog[i]["low" + str(counter + 1)] = 1
                            prog[i]["up" + str(counter + 1)] = lit["value"]
                        counter = counter + 1
                    elif "range" in dim:
                        for ran in dim["range"]:
                            prog[i]["tag"] = "array"
                            prog[i]["count"] = count
                            if "operator" in ran["low"][0]:
                                op = ran["low"][0]["operator"]
                                value = ran["low"][0]["left"][0]["value"]
                                prog[i]["low" + str(counter + 1)] = op + value
                            else:
                                prog[i]["low" + str(counter + 1)] = ran["low"][
                                    0
                                ]["value"]
                            if "operator" in ran["high"][0]:
                                op = ran["high"][0]["operator"]
                                value = ran["high"][0]["left"][0]["value"]
                                prog[i]["up" + str(counter + 1)] = op + value
                            else:
                                prog[i]["up" + str(counter + 1)] = ran["high"][
                                    0
                                ]["value"]
                        counter = counter + 1

        # If the statement is for declaring derived type, which is a class in
        # python
        if isDevType:
            prog.append({"tag": "derived-type"})
            field_num = 0
            field_id_num = 0
            for field in decDevType:
                fieldList = []
                fields = {}
                if "derived-type" in field:
                    prog[0]["name"] = field["derived-type"]
                elif "field-type" in field:
                    fields["type"] = field["field-type"]
                    prog[0][f"field{field_id_num}"] = [fields]
                elif "field-id" in field:
                    fields["name"] = field["field-id"]
                    if "array-size" in field:
                        fields["size"] = field["array-size"]
                    if f"field{field_id_num}" in prog[0]:
                        prog[0][f"field{field_id_num}"][0].update(field)
                    else:
                        fields["type"] = prog[0][f"field{field_id_num-1}"][0][
                            "type"
                        ]
                        prog[0][f"field{field_id_num}"] = [fields]
                        prog[0][f"field{field_id_num}"][0].update(field)
                    field_id_num = field_id_num + 1
            isDevTypeVar = True

        # Adding additional attribute to distinguish derived type variables
        if len(prog) > 0:
            for pro in prog:
                pro["isDevTypeVar"] = isDevTypeVar

        # Set function (subroutine) arguments' types (variable or array)
        for var in prog:
            if "name" in var:
                if var["name"] in state.args:
                    state.subroutine["args"][state.args.index(var["name"])][
                        "arg_type"
                    ] = f"arg_{var['tag']}"
        return prog

    def process_variable(self, root, state) -> List[Dict]:
        try:
            var_name = root.attrib["name"].lower()
            for node in root:
                if node.tag == "initial-value":
                    value = self.parseTree(node, state)
                    return [
                        {"tag": "variable", "name": var_name, "value": value}
                    ]
                else:
                    return [{"tag": "variable", "name": var_name}]
        except:
            return []

    def process_loop(self, root, state) -> List[Dict]:
        if root.attrib["type"] == "do":
            do = {"tag": "do"}
            do_format = []
            for node in root:
                if node.tag == "format":
                    do_format = self.parseTree(node, state)
                elif node.tag == "header":
                    do["header"] = self.parseTree(node, state)
                    if do["header"][0]["low"][0]["tag"] == "ref":
                        lowName = do["header"][0]["low"][0]["name"]
                        if "%" in lowName:
                            curName = lowName
                            devVar = re.findall(r"\"([^\"]+)\"", curName)
                            percInd = curName.find("%")
                            fieldVar = curName[percInd + 1 : len(curName)]
                            newName = devVar[0] + "." + fieldVar
                            do["header"][0]["low"][0]["name"] = newName
                            do["header"][0]["low"][0]["isDevType"] = True
                    if do["header"][0]["high"][0]["tag"] == "ref":
                        highName = do["header"][0]["high"][0]["name"]
                        if "%" in highName:
                            curName = highName
                            devVar = re.findall(r"\"([^\"]+)\"", curName)
                            percInd = curName.find("%")
                            fieldVar = curName[percInd + 1 : len(curName)]
                            newName = devVar[0] + "." + fieldVar
                            do["header"][0]["high"][0]["name"] = newName
                            do["header"][0]["high"][0]["isDevType"] = True
                elif node.tag == "body":
                    do["body"] = self.parseTree(node, state)
            if do_format:
                return [do_format[0], do]
            else:
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

    # This function checks for an asterisk in the argument of a read/write
    # statement and stores it if found.  An asterisk in the first argument
    # specifies a input through or output to console.  An asterisk in the
    # second argument specifies a read/write without a format (implicit
    # read/writes).
    def process_io_control_spec(self, root, state) -> List[Dict]:
        x = []
        for attr in root.attrib:
            if attr == "hasAsterisk" and root.attrib[attr] == "true":
                x = [{"tag": "literal", "type": "char", "value": "*"}]
        return x

    def process_name(self, root, state) -> List[Dict]:
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
            isDevType = False
            singleReference = False
            refName = root.attrib["id"].lower()
            if "%" in refName:
                curName = refName
                devVar = re.findall(r"\"([^\"]+)\"", curName)
                percInd = curName.find("%")
                fieldVar = curName[percInd + 1 : len(curName)]
                refName = devVar[0]
                isDevType = True
                singleReference = True
            # As a solution to handle derived type variable accesses more than
            # one field variables (via % in fortran and . in python), for
            # example, var % x % y, we need to look up the variable name from
            # the variable keep and mark isDevType attribute as True, so later
            # it can be printed properly by pyTranslate.py
            if refName in self.deriveTypeVars:
                isDevType = True
            if isDevType:
                if singleReference:
                    ref = {
                        "tag": "ref",
                        "name": refName,
                        "field-name": fieldVar,
                        "isDevType": True,
                        "arrayStat": self.deriveTypeFields[fieldVar],
                    }
                else:
                    ref = {"tag": "ref", "name": refName, "isDevType": True}
            else:
                ref = {"tag": "ref", "name": refName, "isDevType": False}
            # If a very first variable (or the main variable) is an array,
            # set hasSubscripts to true else false
            if "hasSubscripts" in root.attrib:
                if root.attrib["hasSubscripts"] == "true":
                    ref["hasSubscripts"] = True
                else:
                    ref["hasSubscripts"] = False

            ref["isArray"] = False
            subscripts = []
            for node in root:
                subscripts += self.parseTree(node, state)
            if subscripts:
                ref["subscripts"] = subscripts
                if (ref["name"] in self.arrays):
                    ref["isArray"] = True

            if "arrayStat" in ref and ref["arrayStat"] == "isArray":
                ref["isArray"] = True

            return [ref]

    def process_assignment(self, root, state) -> List[Dict]:
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
            if assign["target"][0]["isDevType"]:
                devTypeAssignment = True
            assign["isDevType"] = devTypeAssignment
            return [assign]

    def process_function(self, root, state) -> List[Dict]:
        subroutine = {"tag": root.tag, "name": root.attrib["name"].lower()}
        self.summaries[root.attrib["name"]] = None
        for node in root:
            if node.tag == "header":
                subroutine["args"] = self.parseTree(node, state)
            elif node.tag == "body":
                subState = state.copy(subroutine)
                subroutine["body"] = self.parseTree(node, subState)
        self.asts[root.attrib["name"]] = [subroutine]
        return [subroutine]

    def process_dimension(self, root, state) -> List[Dict]:
        dimension = {"tag": "dimension"}
        for node in root:
            if node.tag == "range":
                dimension["range"] = self.parseTree(node, state)
            if node.tag == "literal":
                dimension["literal"] = self.parseTree(node, state)
        return [dimension]

    def process_range(self, root, state) -> List[Dict]:
        ran = {}
        for node in root:
            if node.tag == "lower-bound":
                ran["low"] = self.parseTree(node, state)
            if node.tag == "upper-bound":
                ran["high"] = self.parseTree(node, state)
        return [ran]

    def process_keyword_argument(self, root, state) -> List[Dict]:
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
        format_spec = {"tag": "format", "args": []}
        for node in root:
            if node.tag == "label":
                format_spec["label"] = node.attrib["lbl"]
            format_spec["args"] += self.parseTree(node, state)
        return [format_spec]

    def process_format_item(self, root, state) -> List[Dict]:
        variable_spec = {
            "tag": "literal",
            "type": "char",
            "value": root.attrib["descOrDigit"],
        }
        return [variable_spec]

    # This function adds the tag for use statements
    # In case of "USE .. ONLY .." statements, the symbols to be included
    # are stored in the "include" field of the "use" block
    def process_use(self, root, state) -> List[Dict]:
        tag_spec = {"tag": "use", "arg": root.attrib["name"]}
        for node in root:
            if node.tag == "only":
                tag_spec["include"] = []
                for item in node:
                    if item.tag == "name":
                        tag_spec["include"] += [item.attrib["id"]]

        return [tag_spec]

    # This function adds the tag for private symbols
    # Any variable/function being initialized as private is added in this tag
    def process_private_variable(self, root, state) -> List[Dict]:
        for node in root:
            if node.tag == "name":
                return [{"tag": "private", "name": node.attrib["id"].lower()}]

        return []

    # This function is needed for handling derived type declaration
    # It will recursively traverse down to the deepest 'type' node,
    # then returns the derived type id as well as the field types
    def process_type(self, root, state) -> List[Dict]:
        derived_types = []
        devTypeHasArrayField = False
        devTypeHasInitValue = False
        devTypeArrayField = {}
        devTypeVarField = {}

        for node in root:
            if node.tag == "type":
                derived_types = self.parseTree(node, state)
            elif node.tag == "derived-type-stmt":
                derived_types.append(
                    {"derived-type": node.attrib["id"].lower()}
                )
            elif node.tag == "intrinsic-type-spec":
                derived_types.append(
                    {"field-type": node.attrib["keyword1"].lower()}
                )
            elif node.tag == "explicit-shape-spec-list__begin":
                devTypeHasArrayField = True
            elif node.tag == "literal":
                if devTypeHasArrayField:
                    devTypeArrayField["array-size"] = node.attrib["value"]
                else:
                    devTypeVarField["value"] = node.attrib["value"]
                    devTypeHasInitValue = True
            elif node.tag == "component-decl":
                if not devTypeHasArrayField:
                    if devTypeHasInitValue:
                        devTypeVarField["field-id"] = node.attrib["id"].lower()
                        derived_types.append(devTypeVarField)
                        devTypeHasInitValue = False
                    else:
                        derived_types.append(
                            {"field-id": node.attrib["id"].lower()}
                        )
                    self.deriveTypeFields[
                        node.attrib["id"].lower()
                    ] = "notArray"
                    derived_types.append(devTypeArrayField)
                else:
                    devTypeArrayField["field-id"] = node.attrib["id"].lower()
                    self.deriveTypeFields[
                        node.attrib["id"].lower()
                    ] = "isArray"
                    derived_types.append(devTypeArrayField)
                    devTypeHasArrayField = False
            elif node.tag == "derived-type-spec":
                return node.attrib["typeName"].lower()
        return derived_types

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

        if root.tag in self.AST_TAG_HANDLERS:
            return self.AST_TAG_HANDLERS[root.tag](root, state)

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
