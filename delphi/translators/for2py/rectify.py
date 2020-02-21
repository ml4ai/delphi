"""

The purpose of this program is to do all the clean up for translate.py.
This (rectify.py) program will receive OFP generated XML file as an input.
Then, it removes any unnecessary elements and refactor randomly structured
(nested) elements into a correct structure. The output file will be
approximately 30%~40% lighter in terms of number of lines than the OFP XML.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        $python rectify.py <ast_file>

ast_file: The XML representation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

Author: Terrence J. Lim

"""

import re
import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
import copy
from delphi.translators.for2py import For2PyError, syntax, f2grfn
from os.path import isfile, join


class RectifyOFPXML:
    def __init__(self):
        # True if derived type declaration exist
        self.is_derived_type = False
        # True if derived type var. reference exist
        self.is_derived_type_ref = False
        # True if current var. is an array
        self.is_array = False
        # True if format exists in the code
        self.is_format = False
        # True if current var. is for function argument
        self.is_function_arg = False
        # True only if goto-stmt exists in the AST
        self.need_goto_elimination = False
        # True if more goto-stmt exists
        self.continue_elimination = False
        # True if operation negation is needed
        self.need_op_negation = False
        # True if any statement requires reconstruction
        self.need_reconstruct = False
        # True if each goto case is ready for reconstruction
        self.reconstruct_after_case_now = False
        self.reconstruct_before_case_now = False
        # True if each goto case encountered
        self.label_before = False
        self.label_after = False
        # True if statement nests stop statement
        self.is_stop = False
        # True if statements follow either after
        # goto or label. Both cannot be true
        # at the same time
        self.collect_stmts_after_goto = False
        self.collect_stmts_after_label = False
        # True if collecting of statement is done
        self.collecting_stmts_done = False
        # True if reconstruction (goto elimination) is done
        self.reconstruction_for_after_done = False
        self.reconstruction_for_before_done = False
        # True if one reconstructed statement needs another
        # reconstruction by nesting it under do while due to
        # label_before case exist as it's parent goto
        self.encapsulate_under_do_while = False
        # Keep a track where goto was declared
        # whether it's under program(main) or loop body
        self.goto_under_loop = False
        # Keeps track of the signed real/int literal constant inside data
        # statements
        self.is_data_stmt_constant = False
        # Keeps records of encountered <goto-stmt> lbl value
        self.goto_target_lbl_after = []
        self.goto_target_lbl_before = []
        # Keeps records of encountered <label> lbl value
        self.label_lbl_for_before = []
        self.label_lbl_for_after = []
        # A name mapper list for declared
        # 'label_flag_' variables
        self.declared_label_flags = []
        self.declared_goto_flags = []
        # A list to hold save_entity tags
        self.saved_entities = []
        # Keep a track of all encountered goto and label stmts
        self.encountered_goto_label = []
        # Keep a track of goto and label with its case
        self.goto_label_with_case = {}
        # Keeps a track of current label of goto-stmt
        self.current_label = None
        # Keep a track of operations for conditional goto
        # key will be the unique code assigned to each <goto-stmt>
        # {code:Element}
        self.conditional_op = {}
        # Counts the number of <goto-stmt> in the code
        self.goto_stmt_counter = 0
        # Keep a track of collected goto-stmts and labels
        # for goto elimination and reconstruction
        self.stmts_after_goto = {
            'goto-stmts': [],
            'labels': [],
        }
        # Dictionary to hold statements before_after case
        self.statements_to_reconstruct_before = {
            "stmts-follow-label": [],
            "count-gotos": 0,
        }
        # Dictionary to hold statements label_after case
        self.statements_to_reconstruct_after = {
            "stmts-follow-goto": [],
            "stmts-follow-label": [],
            "count-gotos": 0,
        }
        # Keeps a track of current derived type name
        self.cur_derived_type_name = None
        # Keeps a track of current scope of code
        # i.e. program, main, or function, etc.
        self.current_scope = None
        # Keep a track of both array and non-array variables
        # in the dictionary of {'name' : 'scope'}
        self.declared_non_array_vars = {}
        self.declared_array_vars = {}
        # Keep a track of declared derived type variables
        self.derived_type_var_holder_list = []
        # Holds variables extracted from derived type refenrece
        # i.e. x%y%z, then x and y and z
        self.derived_type_refs = []
        # Keeps track of subscripts of arrays
        self.subscripts_holder = []
        # Holds format XML for later reconstruction
        self.format_holder = []
        # Holds a type of parent element's type element
        self.parent_type = ET.Element('')
        # Holds XML of derived type reference for later reconstruction
        self.derived_type_ref = ET.Element('')
        # Actually holds XML of current scope
        self.current_body_scope = ET.Element('')
        # Keeps a track of parent statements to goto-stmt
        self.goto_stmt_parents = []
        # Keeps a track of parent statements to label
        self.label_parents = []
        # Flag to check if the variable is a character
        self.is_character = False
        # List to store all the character variables defined
        self.character_var_list = []
        # Keeps a track of the main body that the statement
        # is nested under, i.e. program, loop, and if, etc
        self.body_level = {
            "current": None,
            "prev": None,
            "grand-prev": None,
        }
        self.body_level_rank = {
            "program": 1,
            "loop": 2,
            "if": 2
        }
        self.body_elem_holder = {
            "program": None,
            "loop": None,
            "if": None,
        }
        # Check for the status whether current <goto-stmt> is
        # conditional or not.
        self.conditional_goto = False
        # True if goto handling needs outward or
        # inward movement.
        self.outward_move = False
        self.inward_move = False
        # True if another <if> appears before
        # goto. This <if> has nothing to do
        # conditional status of goto.
        self.if_appear_before_goto = True
        # True if goto is under <if> that is
        # a conditional goto statement.
        self.goto_under_if = False
        # If goto is conditional and under else
        # that is a case of conditional without operator
        self.goto_under_else = False
        # When handling function, collect names
        self.args_for_function = []
        # Holds arguments of subroutine or function XML object
        self.arguments_list = {}
        # Holds function argument types
        self.argument_types = {}
        # Temporarily holds the type of declaring variable type
        self.variable_type = None
        # Holds the caller arguments that are array
        self.caller_arr_arguments = {}
        # Set to true if handling <call>
        self.call_function = False
        self.original_fortran_file_abs_path = None
        self.module_log_file_path = None
        self.module_files_to_process = []
        self.modules_in_file = []
        self.derived_type_array_dimensions = {}
        self.dim = 0
        # Keeps a track of maximum number of function
        # arguments that are member of interface
        self.member_function_argument_max = 0
        # Mark whether currently handling interface
        # member functions
        self.is_interface_member = False
        # Keeps a track of interface function names
        self.interface_functions = {}
        self.interface_function_xml = {}
        # Mark wheter currently handling interface
        self.is_interface = False
        # Keep a track of currently declaring interface name
        self.cur_interface_name = None
        # Keep a track of interface XML object for later update
        self.interface_xml = {}
        self.dimensions_holder = None

    #################################################################
    #                                                               #
    #                  TAG LISTS FOR EACH HANDLER                   #
    #                                                               #
    #################################################################

    file_child_tags = [
        "program",
        "subroutine",
        "module",
        "declaration",
        "function",
        "prefix"
    ]

    program_child_tags = [
        "header",
        "body"
    ]

    statement_child_tags = [
        "assignment",
        "write",
        "format",
        "stop",
        "execution-part",
        "print",
        "open",
        "read",
        "close",
        "call",
        "statement",
        "label",
        "literal",
        "continue-stmt",
        "do-term-action-stmt",
        "return",
        "contains-stmt",
        "declaration",
        "prefix",
        "function",
        "internal-subprogram",
        "internal-subprogram-part",
        "prefix",
        "exit",
        "cycle",
    ]

    loop_child_tags = [
        "header",
        "body",
        "format"
    ]

    specification_child_tags = [
        "declaration",
        "use",
    ]

    declaration_child_tags = [
        "type",
        "dimensions",
        "variables",
        "format",
        "name",
        "type-declaration-stmt",
        "prefix-spec",
        "save-stmt",
        "saved-entity",
        "access-spec",
        "attr-spec",
        "access-stmt",
        "access-id-list",
        "constants",
        "interface",
        "subroutine",
        "intent",
        "names",
        "procedure-stmt",
        "literal",
        "values"
    ]

    value_child_tags = [
        "literal",
        "operation",
        "name",
    ]

    derived_type_child_tags = [
        "declaration-type-spec",
        "type-param-or-comp-def-stmt-list",
        "component-decl-list__begin",
        "component-initialization",
        "data-component-def-stmt",
        "component-def-stmt",
        "component-attr-spec-list",
        "component-attr-spec-list__begin",
        "explicit-shape-spec-list__begin",
        "explicit-shape-spec",
        "explicit-shape-spec-list",
        "component-attr-spec",
        "component-attr-spec-list__begin",
        "component-shape-spec-list__begin",
        "explicit-shape-spec-list__begin",
        "explicit-shape-spec",
        "component-attr-spec",
        "component-attr-spec-list",
        "end-type-stmt",
        "derived-type-def",

    ]

    header_child_tags = [
        "index-variable",
        "operation",
        "arguments",
        "names",
        "name",
        "loop-control",
        "label",
        "literal",
        "equiv-operand__equiv-op",
        "subroutine-stmt",
        "value-ranges",
    ]

    body_child_tags = [
        "specification",
        "statement",
        "loop",
        "if",
        "label",
        "stop",
        "do-term-action-stmt",
        "select",
        "case",
    ]

    operand_child_tags = [
        "name",
        "literal",
        "operation",
    ]

    subscripts_child_tags = [
        "name",
        "literal",
        "operation",
        "argument",
        "range",
    ]

    index_range_child_tags = [
        "lower-bound",
        "upper-bound",
        "step",
    ]

    bound_child_tags = [
        "literal",
        "name",
        "operation",
    ]

    module_child_tags = [
        "header",
        "body",
        "module-stmt",
        "members",
        "end-module-stmt",
        "contains-stmt",
    ]

    members_child_tags = [
        "subroutine",
        "module-subprogram",
        "module-subprogram-part",
        "declaration",
        "prefix",
        "function",
    ]

    only_child_tags = [
        "name",
        "only",
        "only-list",
    ]

    select_child_tags = [
        "header",
        "body",
    ]

    case_child_tags = [
        "header",
        "body",
    ]

    unnecessary_tags = [
        "do-variable",
        "end-program-stmt",
        "main-program",
        "char-selector",
        "declaration-type-spec",
        "type-param-or-comp-def-stmt-list",
        "component-decl-list__begin",
        "data-component-def-stmt",
        "component-def-stmt",
        "component-initialization",
        "attr-spec",
        "attr-id",
        "designator",
        "int-literal-constant",
        "char-literal-constant",
        "real-literal-constant",
        "io-control-spec",
        "array-spec-element",
        "print-stmt",
        "print-format",
        "keyword-argument",
        "end-subroutine-stmt",
        "logical-literal-constant",
        "equiv-op",
        "equiv-operand",
        "saved-entity-list__begin",
        "saved-entity-list",
        "access-id",
        "parameter-stmt",
        "type-param-value",
        "char-selector",
        "interface-block",
        "interface-stmt",
        "interface-body",
        "interface-specification",
        "end-interface-stmt",
        "select-case-stmt",
        "case-selector",
        "case-stmt",
        "end-select-stmt",
        "component-attr-spec-list__begin",
        "explicit-shape-spec-list__begin",
        "explicit-shape-spec",
        "explicit-shape-spec-list",
        "component-attr-spec",
        "component-attr-spec-list",
        "sequence-stmt",
        "private-or-sequence",
        "data-stmt-set",
        "data-stmt",
        "signed-real-literal-constant",
        "signed-int-literal-constant",
        "data-stmt-constant",
        "data-i-do-object-list__begin",
    ]

    output_child_tags = [
        "name",
        "literal",
        "operation",
    ]

    dtype_var_declaration_tags = [
        "component-decl",
        "component-decl-list",
        "component-decl-list__begin",
    ]

    variable_child_tags = [
        "initial-value",
        "length",
        "dimensions",
    ]

    #################################################################
    #                                                               #
    #                       HANDLER FUNCTIONS                       #
    #                                                               #
    #################################################################

    def handle_tag_file(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
       between the file elements.

        In order to control new sub-element creation under
        the current element, if child.tag == "__tag_name__"
        has been added. If any new tag(s) that is not being
        handled currently, appears in the future, add
        child.tag == "__tag_name__" at the end of the last
        condition. This applies all other handler functions.

        <file>
        </file>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )

            try:
                _ = self.file_child_tags.index(child.tag)
            except KeyError:
                assert (
                    False
                ), f'In handle_tag_file: "{child.tag}" not handled'

            if len(child) > 0 or child.text:
                self.parseXMLTree(child, cur_elem, current, parent, traverse)

    def handle_tag_program(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML
        elements between the program elements.

        <program>
        </program>
        """
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag in self.program_child_tags:
                if child.tag == "body":
                    self.current_body_scope = cur_elem
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_program: "{child.tag}" not handled'

    def handle_tag_header(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the header elements.

        <header>
        </header>
        """
        # This holder will be used only when refactoring of header is needed,
        # such as with odd syntax of .eqv. operator
        temp_elem_holder = []
        target_tags = [
                "name",
                "literal",
                "equiv-operand__equiv-op"
        ]
        need_refactoring = False
        for child in root:
            self.clean_attrib(child)

            if child.tag in self.header_child_tags:
                if child.tag == "subroutine-stmt":
                    current.attrib.update(child.attrib)
                else:
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )

                    # Add a new interface:[] element into the interface
                    # function tracker dictionary if current header is
                    # for declaring interface.
                    if self.is_interface and cur_elem.tag == "name":
                        self.interface_functions[cur_elem.attrib['id']] = []
                        self.interface_function_xml[cur_elem.attrib['id']] = {}
                        self.cur_interface_name = cur_elem.attrib['id']

                    if len(child) > 0 or child.text:
                        self.parseXMLTree(
                                child, cur_elem, current, parent, traverse
                        )
                        # If the current header belongs to <function>,
                        # we need to manipulate the structure of the AST
                        # to have an equivalent syntax as <subroutine>
                        if (
                                parent.tag == "function"
                                and cur_elem.tag == "names"
                        ):
                            current.remove(cur_elem)
                            count = len(self.args_for_function)
                            cur_elem = ET.SubElement(
                                                current, "arguments",
                                                {"count": str(count)})
                            for arg in self.args_for_function:
                                _ = ET.SubElement(
                                                cur_elem, "argument",
                                                {"name": arg,
                                                 "is_array": "false"}
                                )

                        # If the current header belongs to <subroutine>,
                        # add it to the arguments_list for later
                        # array status marking when a function call happens
                        if (
                            (parent.tag == "subroutine"
                             and child.tag == "arguments")
                            or parent.tag == "function"
                        ):
                            sub_name = parent.attrib["name"]
                            self.arguments_list[sub_name] = cur_elem

                    if cur_elem.tag in target_tags:
                        temp_elem_holder.append(cur_elem)
                        if cur_elem.tag == "equiv-operand__equiv-op":
                            need_refactoring = True
                    # Handler for the case where label appears under
                    # the header element. This happens when label
                    # is assigned to the if statement.
                    if (
                        traverse == 1
                        and child.tag == "label"
                    ):
                        lbl = child.attrib['lbl']
                        parent.attrib['label'] = lbl
                        self.encountered_goto_label.append(lbl)
                        # Label-before case
                        if (
                                not self.goto_target_lbl_after
                                or lbl not in self.goto_target_lbl_after
                        ):
                            self.goto_label_with_case[lbl] = "before"
                            if self.label_after:
                                self.label_before = False
                            else:
                                self.label_before = True
                            if lbl not in self.label_lbl_for_before:
                                self.label_lbl_for_before.append(lbl)
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_header: Empty elements "{child.tag}"'

        # Revert the argument list back to its empty state to accommodate for
        # new functions
        self.args_for_function = []
        # equivalent operator has a weird ast syntax,
        # so it requires refactoring.
        if need_refactoring:
            self.reconstruct_header(temp_elem_holder, current)

    def handle_tag_body(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the body elements.

        <body>
        </body>
        """
        current.attrib['parent'] = parent.tag
        self.body_elem_holder[parent.tag] = current
        # Keeping the track of the body's boundary.
        if traverse == 1:
            if self.body_level['prev'] is None:
                self.body_level['grand-prev'] = parent.tag
                self.body_level['prev'] = parent.tag
            else:
                assert (
                        self.body_level['current'] is not None
                ), "self.body_level['current'] cannot be None."

                self.body_level['grand-prev'] = self.body_level['prev']
                self.body_level['prev'] = self.body_level['current']
            self.body_level['current'] = parent.tag

        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag in self.body_child_tags:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )

                    if traverse == 1:
                        # Handling conditional <goto-stmt>
                        if (
                                parent.tag == "if"
                                and "goto-stmt" in cur_elem.attrib
                        ):
                            assert (
                                    "lbl" in cur_elem.attrib
                            ), "Label 'lbl' must be present in <if> attrib"

                            # goto-stmt counter will be used as
                            # an identifier for two statements
                            # that are nested one in another
                            unique_code = str(self.goto_stmt_counter)

                            parent.attrib['conditional-goto-stmt-lbl'] = \
                                cur_elem.attrib['lbl']
                            parent.attrib['code'] = unique_code
                            if "goto-move" in cur_elem.attrib:
                                parent.attrib['goto-move'] = "true"
                            if "goto-remove" in cur_elem.attrib:
                                parent.attrib['goto-remove'] = "true"

                            cur_elem.attrib['conditional-goto-stmt'] = "true"
                            cur_elem.attrib['code'] = unique_code
                            # If the <statement> for <goto-stmt> is nested
                            # under the conditional <if>, then the boundary of
                            # <statement> remains as the current - 1 level.
                            cur_elem.attrib['body-level'] = \
                                self.body_level['prev']

                            self.body_level['current'] = self.body_level['prev']
                        else:
                            self.body_level['grand-prev'] = \
                                self.body_level['prev']
                            self.body_level['prev'] = self.body_level['current']
                            self.body_level['current'] = parent.tag

                        # Check if conditional goto-stmt is under loop statement
                        if parent.tag == "loop":
                            if (
                                    child.tag == "if"
                                    or (
                                        child.tag == "statement"
                                        and "conditional-goto-stmt" in
                                        child.attrib
                                    )
                            ):
                                self.goto_under_loop = True

                        # Check a case where <if> under another <if>.
                        # <if>
                        #   <if>
                        if (
                            grandparent.tag == "body"
                            and "parent" in grandparent.attrib
                            and grandparent.attrib['parent'] == "if"
                        ):
                            self.goto_under_if = True

                    else:
                        # A Checker for whether current statement is
                        # nested under the loop.
                        if "body-level" in cur_elem.attrib:
                            if cur_elem.attrib['body-level'] == "loop":
                                self.goto_under_loop = True
                            else:
                                self.goto_under_loop = False

                        new_parent = current
                        # Reconstruction of statements
                        if (
                                "parent" in current.attrib
                                and (
                                (not self.goto_under_loop
                                    and not self.goto_under_if
                                    and current.attrib['parent'] == "program")
                                or (self.goto_under_if
                                    and current.attrib['parent'] == "if")
                                or (self.goto_under_loop
                                    and current.attrib['parent'] == "loop")
                                    )
                        ):
                            # Remove statements marked for removal(2nd traverse)
                            if (
                                    "goto-remove" in child.attrib
                                    or "goto-move" in child.attrib
                            ):
                                current.remove(cur_elem)
                                if (
                                        self.reconstruct_after_case_now
                                        and
                                        not self.reconstruction_for_after_done
                                ):
                                    self.reconstruct_goto_after_label(
                                        new_parent, traverse,
                                        self.statements_to_reconstruct_after
                                    )
                                    if self.label_lbl_for_before:
                                        self.continue_elimination = True
                                if (
                                        self.reconstruct_before_case_now
                                        and
                                        not self.reconstruction_for_before_done
                                ):
                                    reconstruct_target = \
                                        self.statements_to_reconstruct_before
                                    self.reconstruct_goto_before_label(
                                        new_parent, traverse, reconstruct_target
                                    )
                                    if self.label_lbl_for_after:
                                        self.continue_elimination = True
                                if (
                                        not self.label_lbl_for_before
                                        and not self.label_lbl_for_after
                                ):
                                    self.continue_elimination = False
                else:
                    assert False, f'In handle_tag_body: "{child.tag}" ' \
                                  f'not handled'
            else:
                if (
                        child.tag in self.body_child_tags
                        and child.tag != "statement"
                ):
                    _ = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                elif child.tag == "statement":
                    if len(child) > 0:
                        cur_elem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        self.parseXMLTree(
                            child, cur_elem, current, parent, traverse
                        )
                else:
                    assert (
                        False
                    ), f'In handle_tag_body: Empty elements "{child.tag}"' \
                       f' not handled'

        if self.is_format:
            self.reconstruct_format(parent, traverse)
            self.is_format = False

    def handle_tag_specification(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the specification elements.

        <specification>
        </specification>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )

                try:
                    _ = self.specification_child_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_specification: "{child.tag}" not handled'

                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                if child.tag != "declaration":
                    assert (
                        False
                    ), f'In handle_tag_specification: Empty elements ' \
                       f'"{child.tag}"'

    def handle_tag_declaration(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the declaration elements.

        <declaration>
        </declaration>
        """
        is_derived_type_dimension_setting = False
        is_end_of_one_dimension = False
        dim_number = 0
        for child in root:
            self.clean_attrib(child)
            # Temporarily hold the declaring variable's type.
            if child.tag == "type" and "name" in child.attrib:
                self.variable_type = child.attrib['name']
            # Keep a track of array in derived type dimension information.
            if child.tag == "explicit-shape-spec-list__begin":
                is_derived_type_dimension_setting = True
                self.dim += 1
                dim_number += 1
                self.derived_type_array_dimensions[self.dim] = []
            elif child.tag == "explicit-shape-spec":
                is_end_of_one_dimension = True
                dim_number += 1
            elif child.tag == "explicit-shape-spec-list":
                is_derived_type_dimension_setting = False

            if len(child) > 0 or child.text:
                if child.tag in self.declaration_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder.append(child)
                    elif (
                            child.tag == "name"
                            or child.tag == "literal"
                    ):
                        if is_derived_type_dimension_setting:
                            child.attrib["dim-number"] = str(dim_number)
                            self.derived_type_array_dimensions[self.dim].append(child)
                        else:
                            self.derived_type_var_holder_list.append(child)
                    else:
                        cur_elem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        if child.tag == "dimensions":
                            self.is_array = True
                        self.parseXMLTree(
                            child, cur_elem, current, parent, traverse
                        )
                elif (
                        child.tag == "component-array-spec"
                        or child.tag == "operation"
                ):
                    self.derived_type_var_holder_list.append(child)
                else:
                    assert (
                        False
                    ), f'In handle_tag_declaration: "{child.tag}" not handled'
            else:
                if child.tag in self.declaration_child_tags:
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    if child.tag == "saved-entity":
                        # If you find saved-entity, add the element to a list 
                        # and remove it from the XML since you want to shift 
                        # it below save-stmt
                        self.saved_entities.append(cur_elem)
                        current.remove(cur_elem)
                    elif child.tag == "save-stmt":
                        # If you find save-stmt, check if it contains 
                        # saved-entities and add it below this XML element
                        if len(self.saved_entities) > 0:
                            for item in self.saved_entities:
                                _ = ET.SubElement(cur_elem, item.tag,
                                                  item.attrib)

                            # Reinitialize this list since you'll need an
                            # empty one for the next SAVE statement
                            self.saved_entities = []
                elif (
                        child.tag == "component-decl"
                        or child.tag == "component-decl-list"
                        or child.tag == "component-decl-list__begin"
                ):
                    current.attrib['type'] = "derived-type"
                    self.derived_type_var_holder_list.append(child)
                elif child.tag == "component-array-spec":
                    self.derived_type_var_holder_list.append(child)
                else:
                    if (
                            child.tag not in self.unnecessary_tags
                            and child.tag not in self.derived_type_child_tags
                    ):
                        assert (
                            False
                        ), f'self.In handle_tag_declaration: Empty elements "' \
                            f'{child.tag}" not handled'
        if self.is_array:
            self.is_array = False

        if self.is_character:
            self.is_character = False

        # If is_derived_type is true,
        # reconstruct the derived type declaration AST structure
        if self.is_derived_type:
            if self.derived_type_var_holder_list:
                # Modify or add 'name' attribute of the MAIN (or the outer
                # most) <type> elements with the name of derived type name
                self.parent_type.set("name", self.cur_derived_type_name)
                self.reconstruct_derived_type_declaration()
            self.is_derived_type = False

        self.variable_type = None

        if self.dimensions_holder:
            self.restruct_declaration(current, parent)
            parent.remove(current)
            self.dimensions_holder = None

    def handle_tag_type(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the variables elements.

        <type>
        </type>
        """
        if current.attrib.get("name"):
            if current.attrib["name"].lower() == "character":
                self.is_character = True
                current.set("string_length", str(1))

        dim_number = 0
        is_derived_type_dimension_setting = False
        for child in root:
            self.clean_attrib(child)
            if "keyword2" in child.attrib:
                if child.attrib['keyword2'] == "":
                    current.attrib['keyword2'] = "none"
                else:
                    current.attrib['keyword2'] = child.attrib['keyword2']
            else:
                current.attrib['keyword2'] = "none"
            if child.tag == "type":
                # Having a nested "type" indicates that this is
                # a "derived type" declaration.
                #    In other word, this is a case of
                #    <type>
                #        <type>
                #            ...
                #        </type>
                #    </type>
                self.is_derived_type = True
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                self.parent_type = current
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            elif child.tag == "derived-type-stmt":
                # If child.tag is derived-type-stmt while self.is_derived_type
                # is not true, it's an indication of only a single variable
                # was declared under the derived type declaration, so the syntax
                # has no nested type case like above. Thus, in order to make
                # the syntax same, I'm adding another type and nest everything
                # under it.
                if not self.is_derived_type:
                    self.is_derived_type = True
                    type_elem = ET.SubElement(current, current.tag,
                                              current.attrib)
                    type_elem.set("is_derived_type", str(self.is_derived_type))
                    type_elem.set("name", child.attrib['id'])
                    self.parent_type = current
                # Modify or add 'name' attribute of the <type>
                # elements with the name of derived type name
                current.set("name", child.attrib['id'])
                # And, store the name of the derived type name for
                # later setting the outer most <type> elements's name attribute
                self.cur_derived_type_name = child.attrib['id']
            elif child.tag == "explicit-shape-spec-list__begin":
                is_derived_type_dimension_setting = True
                self.dim += 1
                dim_number += 1
                self.derived_type_array_dimensions[self.dim] = []
            elif child.tag == "explicit-shape-spec":
                is_end_of_one_dimension = True
                dim_number += 1
            elif child.tag == "explicit-shape-spec-list":
                is_derived_type_dimension_setting = False
            elif child.tag == "intrinsic-type-spec":
                if self.is_derived_type:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "derived-type-spec":
                if self.variable_type == None:
                    self.variable_type = child.attrib['typeName']
                if not self.is_derived_type:
                    self.is_derived_type = True
                    current.set("name", child.attrib['typeName'])
                else:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "literal":
                if self.is_character:
                    self.derived_type_var_holder_list.append(child)
                    current.set("string_length", str(child.attrib["value"]))
                elif is_derived_type_dimension_setting:
                    child.attrib["dim-number"] = str(dim_number)
                    self.derived_type_array_dimensions[self.dim].append(child)
                else:
                    self.derived_type_var_holder_list.append(child)
            elif (
                    is_derived_type_dimension_setting
                    and child.tag == "name"
            ):
                    child.attrib["dim-number"] = str(dim_number)
                    self.derived_type_array_dimensions[self.dim].append(child)
            elif child.tag == "component-array-spec":
                self.derived_type_var_holder_list.append(child)
            elif child.tag in self.dtype_var_declaration_tags:
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "length":
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, cur_elem, current, parent, traverse)
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert False, f'In handle_tag_type: "{child.tag}" not ' \
                                  f'handled'
        # This will mark whether this type declaration is for a derived type
        # declaration or not
        current.set("is_derived_type", str(self.is_derived_type))

    def handle_tag_variables(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the variables elements.

        <variables>
        </variables>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                if (
                        child.tag == "variable"
                        and self.current_scope in self.argument_types
                        and child.attrib['name'] in self.argument_types[
                        self.current_scope]
                ):
                    self.argument_types[self.current_scope][child.attrib[
                        'name']] = self.variable_type
                # Up to this point, all the child (nested or sub) elements were
                # <variable>
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                cur_elem.set("is_array", str(self.is_array).lower())
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                if (
                    child.tag == "variable"
                    and child.attrib
                ):
                    grandparent = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert (
                            child.tag == "variable"
                        ), f'In handle_tag_variables: "{child.tag}" not handled'

    def handle_tag_variable(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the variables elements.

        <variable>
        </variable>
        """
        # Store all declared variables based on their array status
        if current.attrib['is_array'] == "true":
            self.declared_array_vars.update(
                {current.attrib['name']: self.current_scope}
            )
        else:
            self.declared_non_array_vars.update(
                {current.attrib['name']: self.current_scope}
            )

        if self.is_character:
            self.character_var_list.append(current.attrib['name'])
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag in self.variable_child_tags:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                    if child.tag == "dimensions":
                        current.attrib['is_array'] = "true"
                        self.declared_array_vars.update(
                            {current.attrib['name']: self.current_scope}
                        )
                        del self.declared_non_array_vars[current.attrib['name']]
                        current.remove(self.dimensions_holder)
                else:
                    assert (
                        False
                    ), f'In handle_tag_variable: "{child.tag}" not handled'
            else:
                if child.tag == "entity-decl":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_variable: Empty elements "{child.tag}"' \
                       f' not handled'

    def handle_tag_constants(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements between the
        constants elements.

        <constants>
        </constants>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                cur_elem.set("is_array", str(self.is_array).lower())
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            elif child.tag == "parameter-stmt":
                pass
            else:
                assert (
                        child.tag == "constant"
                ), f'In handle_tag_constant: "{child.tag}" not handled'

    def handle_tag_constant(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements between the
        constants elements.

        <constant>
        </constant>
        """
        # Store all declared variables based on their array status
        if current.attrib['is_array'] == "true":
            self.declared_array_vars.update(
                {current.attrib['name']: self.current_scope}
            )
        else:
            self.declared_non_array_vars.update(
                {current.attrib['name']: self.current_scope}
            )

        for child in root:
            self.clean_attrib(child)
            if child.text or len(child) > 0:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                assert (
                    False
                ), f'In handle_tag_constant: Empty elements "{child.tag}" ' \
                   f'not handled'

    def handle_tag_statement(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the statement elements.

        <statement>
        </statement>
        """

        if traverse == 1:
            current.attrib['body-level'] = self.body_level['current']

        for child in root:
            self.clean_attrib(child)
            if child.tag in self.statement_child_tags:
                if child.tag == "stop":
                    self.is_stop = True
                    current.attrib['has-stop'] = "true"
                    current.attrib['goto-remove'] = "true"

                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )

                if child.tag == "label":
                    current.attrib['label'] = child.attrib['lbl']
                    label_presented = True
                    lbl = child.attrib['lbl']

                    self.encountered_goto_label.append(lbl)

                    if traverse == 1:
                        # Label-before case
                        if (
                                not self.goto_target_lbl_after
                                or lbl not in self.goto_target_lbl_after
                        ):
                            self.goto_label_with_case[lbl] = "before"
                            # Since we want to handle label_after case before
                            # label_before when both cases appear in the code,
                            # we ignore all label_before case until _after case
                            # get handled. Thus, mark label_before to false
                            if self.label_after:
                                self.label_before = False
                            else:
                                self.label_before = True
                            if lbl not in self.label_lbl_for_before:
                                self.label_lbl_for_before.append(lbl)
                        # Label-after case
                        else:
                            self.goto_label_with_case[lbl] = "after"
                            self.collect_stmts_after_goto = False
                            self.collect_stmts_after_label = True
                            if lbl not in self.label_lbl_for_after:
                                self.label_lbl_for_after.append(lbl)

                        if (
                                self.label_before
                                or lbl in self.label_lbl_for_before
                        ):
                            current.attrib['goto-move'] = "true"
                        else:
                            current.attrib['goto-remove'] = "true"
                        current.attrib['target-label-statement'] = "true"

                        # Since <format> is followed by <label>,
                        # check the case and undo all operations done for goto.
                        if child.tag == "format" and label_presented:
                            del self.label_lbl[-1]
                            del current.attrib['target-label-statement']
                            del current.attrib['goto-move']
                            self.label_before = False
                    else:
                        assert (
                                traverse > 1
                        ), "In handle_tag_statement. Reconstruction must be " \
                           "done in traverse > 1."
                        if self.collecting_stmts_done:
                            self.reconstruct_after_case_now = True
                            self.collecting_stmts_done = False

                if child.text or len(child) > 0:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            elif child.tag == "name":
                # If a 'name' tag is the direct sub-elements of 'statement',
                # it's an indication of this statement is handling
                # (usually assignment) derived type variables. Thus,
                # in order to make concurrent with other assignment syntax,
                # remove the outside name elements (but store it to the
                # temporary holder) and reconstruct it before the end of
                # statement
                assert is_empty(self.derived_type_var_holder_list)
                self.derived_type_var_holder_list.append(child.attrib['id'])
                self.parseXMLTree(
                    child, current, current, parent, traverse
                )
            elif child.tag == "goto-stmt":
                # self.goto_stmt_level = parent.attrib['parent']

                # <goto-stmt> met, increment the counter
                self.goto_stmt_counter += 1
                # If goto-stmt was seen, we do not construct element for it.
                # However, we collect the information (attributes) that is
                # associated to the existing OFP generated element
                self.need_goto_elimination = True
                self.if_appear_before_goto = False
                target_lbl = child.attrib['target_label']
                current.attrib['goto-stmt'] = "true"
                current.attrib['lbl'] = target_lbl
                _ = ET.SubElement(current, child.tag, child.attrib)
                # Reaching goto-stmt is a flag to stop collecting states
                if traverse == 1:
                    if (
                        "type" in parent.attrib
                        and parent.attrib['type'] == "else"
                    ):
                        self.goto_under_else = True

                    self.encountered_goto_label.append(target_lbl)
                    if self.collect_stmts_after_label:
                        current.attrib['goto-remove'] = "true"
                        current.attrib['next-goto'] = "true"
                        self.statements_to_reconstruct_after[
                            'stmts-follow-label'].append(current)
                        self.collect_stmts_after_label = False
                        self.collecting_stmts_done = True

                    # A case where label appears "before" goto
                    if target_lbl in self.label_lbl_for_before:
                        self.goto_label_with_case[target_lbl] = "before"
                        self.statements_to_reconstruct_before[
                            'count-gotos'] += 1
                        self.goto_target_lbl_before.append(target_lbl)
                    # A case where label appears "after" goto
                    else:
                        self.goto_label_with_case[target_lbl] = "after"
                        self.statements_to_reconstruct_after['count-gotos'] += 1

                        if "parent-goto" not in current.attrib:
                            current.attrib['skip-collect'] = "true"
                        self.goto_target_lbl_after.append(target_lbl)
                        self.collect_stmts_after_goto = True
                        self.label_after = True
                else:
                    if target_lbl in self.label_lbl_for_before:
                        assert (
                                traverse > 1
                        ), "Reconstruction cannot happen in the first traverse"
                        if self.label_before:
                            self.reconstruct_before_case_now = True
                        return
            else:
                assert (
                    False
                ), f'In handle_tag_statement: "{child.tag}" not handled'

        # Statement collector (1st traverse)
        if traverse == 1:
            if self.label_before and not self.label_after:
                # Since we do not want to extract the stop statement from
                # that is not a main body, check it before extraction
                if "has-stop" not in current.attrib:
                    current.attrib['goto-move'] = "true"
                    self.statements_to_reconstruct_before[
                        'stmts-follow-label'].append(current)
                else:
                    if (
                            parent.tag == "body"
                            and parent.attrib['parent'] == "program"
                    ):
                        self.statements_to_reconstruct_before[
                            'stmts-follow-label'].append(current)
            elif self.label_after:
                if self.collect_stmts_after_goto:
                    current.attrib['goto-remove'] = "true"
                    if "has-stop" not in current.attrib:
                        self.statements_to_reconstruct_after[
                            'stmts-follow-goto'].append(current)
                    else:
                        if (
                                parent.tag == "body"
                                and parent.attrib['parent'] == "program"
                        ):
                            self.statements_to_reconstruct_after[
                                'stmts-follow-goto'].append(current)

                    if "goto-stmt" in current.attrib:
                        self.stmts_after_goto['goto-stmts'].append(
                            current.attrib['lbl'])
                    elif "target-label-statement" in current.attrib:
                        self.stmts_after_goto['labels'].append(
                            current.attrib['label'])

                elif self.collect_stmts_after_label:
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after[
                        'stmts-follow-label'].append(current)

                    if (
                            (
                                parent.tag == "body"
                                and parent.attrib['parent'] == "program"
                                and "has-stop" in current.attrib
                            )
                            or self.goto_under_loop
                    ):
                        self.collect_stmts_after_label = False
                        self.reconstruct_after_case_now = True

    def handle_tag_assignment(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the assignment elements.

        <assignment>
        </assignment>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "target" or child.tag == "value":
                self.parseXMLTree(child, cur_elem, current, parent, traverse)
            else:
                assert (
                    False
                ), f'In handle_tag_assignment: "{child.tag}" not handled'

    def handle_tag_target(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the target elements.

        <target>
        </target>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag == "name":
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
            else:
                assert (
                    False
                ), f'In handle_tag_target: "{child.tag}" not handled'

    def handle_tag_names(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the names elements.

        <names>
        <names>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "name":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if "id" in child.attrib and self.is_interface:
                    self.interface_functions[self.cur_interface_name].append(
                        child.attrib['id'])
                    self.interface_function_xml[self.cur_interface_name][
                        child.attrib['id']] = cur_elem
                if grandparent.tag == "function":
                    self.args_for_function.append(cur_elem.attrib['id'])
                # If the element holds sub-elements, call the XML tree parser
                # with created new <name> element
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                # Else, update the element's attribute
                # with the default <name> element attributes
                else:
                    attributes = {
                        "hasSubscripts": "false",
                        "is_array": "false",
                        "numPartRef": "1",
                        "type": "ambiguous",
                    }
                    # Check if the variable is a function argument
                    if self.is_function_arg:
                        attributes['is_arg'] = "true"
                    else:
                        attributes['is_arg'] = "false"
                    cur_elem.attrib.update(attributes)
            else:
                assert False, f'In handle_tag_names: "{child.tag}" not handled'

    def handle_tag_name(self, root, current, parent, _, traverse):
        """This function handles cleaning up the XML elements between
        the name elements.

        <name>
        <name>

        There are three different types of names that the type attribute can
    hold:
            (1) variable  - Simple (single) variable or an array
            (2) procedure - Function (or procedure) call
            (3) ambiguous - None of the above two type
        """
        if (
            "id" in current.attrib
            and current.attrib['id'] in self.declared_array_vars
        ):
            current.attrib['is_array'] = "true"
        else:
            current.attrib['is_array'] = "false"

        # If 'id' attribute holds '%' symbol, it's an indication of derived type
        # referencing. Thus, clean up the 'id' and reconstruct the <name> AST.
        if "id" in current.attrib and "%" in current.attrib['id']:
            self.is_derived_type_ref = True
            self.clean_derived_type_ref(current)

        # Default attribute values
        current.attrib['hasSubscripts'] = "false"
        current.attrib['is_arg'] = "false"
        current.attrib['numPartRef'] = "1"
        current.attrib['type'] = "ambiguous"

        for child in root:
            self.clean_attrib(child)
            if child.text:
                if (
                        child.tag == "subscripts"
                        or child.tag == "assignment"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    if child.tag == "subscripts":
                        # If current name is for caller arguments,
                        # mark the name of the function in the subscripts
                        # as an one of its attributes
                        if parent.tag == "call":
                            cur_elem.attrib['fname'] = current.attrib['id']
                        current.attrib['hasSubscripts'] = "true"
                        # Check whether the variable is an array AND the
                        # variable is for the current scope. This is important
                        # for derived type variable referencing
                        if (
                                current.attrib['id'] in self.declared_array_vars
                                and self.declared_array_vars[
                                        current.attrib['id']
                                    ] == self.current_scope
                        ):
                            # Since the procedure "call" has a same AST syntax
                            # as an array, check its type and set the "is_array"
                            # value
                            assert (
                                    current.attrib['type'] != "procedure"
                            ), "Trying to assign a procedure call to while " \
                               "is_array true."
                            current.attrib['is_array'] = "true"
                        elif (
                                current.attrib['id']
                                in self.declared_non_array_vars
                                and self.declared_non_array_vars[
                                        current.attrib['id']
                                    ] == self.current_scope
                                and current.attrib['id'] not in
                                self.character_var_list
                        ):
                            current.attrib['hasSubscripts'] = "false"

                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                elif child.tag == "output":
                    assert (
                        is_empty(self.derived_type_var_holder_list)
                    ), "derived_type_var holder must be empty."
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.derived_type_var_holder_list.append(root.attrib['id'])
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                elif child.tag == "name":
                    self.parseXMLTree(
                        child, current, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In self.handle_tag_name: "{child.tag}" not handled'
            else:
                if child.tag == "generic_spec":
                    _ = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                elif child.tag == "data-ref":
                    current.attrib.update(child.attrib)
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert (
                            False
                        ), f'In self.handle_tag_name: Empty elements ' \
                           f'"{child.tag}" not handled'

        # If the name element is for handling
        # derived type references, reconstruct it
        if self.derived_type_refs:
            self.reconstruct_derived_type_names(current)
            self.is_derived_type_ref = False
            self.need_reconstruct = True

    def handle_tag_value(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the value elements.

        <value>
        </value>
        """
        function_call = False
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(current, child.tag, child.attrib)

            if (
                child.tag == "name"
                and child.attrib['id'] in self.arguments_list
            ):
                # if 'id' is in the arguments_list, it indicates that
                # the RHS of the assignment is a function call
                self.call_function = True
                function_call = True
                current.attrib['fname'] = child.attrib['id']

            try:
                _ = self.value_child_tags.index(child.tag)
            except ValueError:
                assert False, f'In handle_tag_value: "{child.tag}" not handled'

            self.parseXMLTree(
                child, cur_elem, current, parent, traverse
            )

            # If current assignment is done with a function call,
            # then update function definition's arguments with array status
            if function_call:
                self.update_arguments(current)

            if (
                    child.tag == "name"
                    and self.need_reconstruct
            ):
                self.reconstruct_name_element(cur_elem, current)

    def handle_tag_literal(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the literal elements.

        <literal>
        </literal>
        """
        if '"' in current.attrib['value']:
            current.attrib['value'] = self.clean_id(current.attrib['value'])

        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                if child.tag == "stop":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                elif child.tag == "literal":
                    cur_elem = ET.SubElement(
                        parent, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, parent, grandparent, traverse
                    )
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert (
                            False
                        ), f'In handle_tag_literal: "{child.tag}" not handled'
            else:
                if child.tag == "data-stmt-constant":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_literal: Empty "{child.tag}" not handled'

    def handle_tag_dimensions(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the dimensions elements.

        <dimensions>
        </dimensions>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag == "dimension":
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert (
                            False
                        ), f'In handle_tag_dimensions: "{child.tag}" not ' \
                           f'handled'
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_dimensions: Empty "{child.tag}" not ' \
                       f'handled'

        if parent.tag == "variable":
            self.dimensions_holder = current

    def handle_tag_dimension(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the dimension elements.

        <dimension>
        </dimension>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if (
                        child.tag == "literal"
                        or child.tag == "range"
                ):
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert (
                            False
                        ), f'In handle_tag_dimension: "{child.tag}" not handled'
            elif child.tag == "literal":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_dimension: Empty "{child.tag}" not ' \
                       f'handled'

    def handle_tag_loop(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the do loop elements.

        <loop>
        </loop>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text or len(child) > 0:
                if child.tag in self.loop_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder.append(child)
                    else:
                        cur_elem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        self.parseXMLTree(
                            child, cur_elem, current, parent, traverse
                        )
                else:
                    assert (
                            child.tag in self.unnecessary_tags
                    ), f'In self.handle_tag_loop: "{child.tag}" not handled'

    def handle_tag_index_variable_or_range(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the index_variable or range elements.

        <index_variable>          or        <range>
        </index_variable>                   </range>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )

                if child.tag in self.index_range_child_tags:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                            child.tag in self.unnecessary_tags
                    ), f'In handle_tag_index_variable_or_range: "{child.tag}"' \
                       f' not handled'

            else:
                if traverse > 1:
                    _ = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                else:
                    assert (
                            child.tag in self.unnecessary_tags
                    ), f'In handle_tag_index_variable_or_range: Empty ' \
                       f'"{child.tag}" not handled'

    def handle_tag_bound(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the upper_bound elements.

        <upper_bound>
        </upper_bound>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag in self.bound_child_tags:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                    if (
                            child.tag == "name"
                            and self.need_reconstruct
                    ):
                        self.reconstruct_name_element(cur_elem, current)
                else:
                    assert (
                            child.tag in self.unnecessary_tags
                    ), f'In handle_tag_upper_bound: "{child.tag}" not handled'
            else:
                if traverse > 1:
                    _ = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                else:
                    assert (
                            child.tag in self.unnecessary_tags
                    ), f'In handle_tag_upper_bound: Empty "{child.tag}" not ' \
                       f'handled'

    def handle_tag_subscripts(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the subscripts elements.

        <subscripts>
        </subscripts>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag == "subscript":
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In self.handle_tag_subscripts: "{child.tag}" not ' \
                       f'handled'

    def handle_tag_subscript(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the subscript elements.

        <subscript>
        </subscript>
        """
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                try:
                    _ = self.subscripts_child_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In self.handle_tag_subscript: "{child.tag}" not ' \
                       f'handled'

                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
                # If current subscript is for a function caller and
                # current element (argument) is an array, then store
                # it into the caller_arr_arguments map for later use
                if (
                    self.call_function
                    and (cur_elem.tag == "name"
                         and cur_elem.attrib['is_array'] == "true")
                ):
                    assert (
                        "fname" in parent.attrib
                    ), "If this subscript is for the caller argument,\
                            fname must exist in the parent"
                    fname = parent.attrib['fname']
                    arg = cur_elem.attrib['id']
                    if fname in self.caller_arr_arguments:
                        self.caller_arr_arguments[fname].append(arg)
                    else:
                        self.caller_arr_arguments[fname] = [arg]

    def handle_tag_operation(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the operation elements.

        <operation>
        </operation>
        """
        for child in root:
            self.clean_attrib(child)
            # A process of negating the operator during goto elimination
            if (
                    child.tag == "operator"
                    and self.need_op_negation
            ):
                child.attrib['operator'] = \
                    syntax.NEGATED_OP[child.attrib['operator']]
                self.need_op_negation = False
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag == "operand":
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                if child.tag != "operator":
                    assert (
                        False
                    ), f'In handle_tag_operation: "{child.tag}" not handled'

    def handle_tag_operand(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the operation elements.

        <operand>
        </operand>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )

            try:
                _ = self.operand_child_tags.index(child.tag)
            except ValueError:
                assert (
                    False
                ), f'In handle_tag_operand: "{child.tag}" not handled'

            self.parseXMLTree(
                child, cur_elem, current, parent, traverse
            )
            if (
                    child.tag == "name"
                    and self.need_reconstruct
            ):
                self.reconstruct_name_element(cur_elem, current)

    def handle_tag_write(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the operation elements.

        <operand>
        </operand>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text or len(child) > 0:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if (
                        child.tag == "io-controls"
                        or child.tag == "outputs"
                ):
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_write: "{child.tag}" not handled'

    def handle_tag_io_controls(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the io-controls elements.

        <io-controls>
        </io-controls>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.text or len(child) > 0:
                if child.tag == "io-control":
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_io_controls: "{child.tag}" not handled'

    def handle_tag_io_control(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the io-control elements.

        <io-control>
        </io-control>
        """
        for child in root:
            self.clean_attrib(child)
            # To make io-control elements simpler, the code below
            # will append io-control-spec's attributes to its
            # parent (io-control). This will eliminate at least
            # one recursion in translate.py to retrieve
            # the io-control information
            if child.tag == "io-control-spec":
                current.attrib.update(child.attrib)
            if child.text:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-control" or child.tag == "literal":
                    self.parseXMLTree(child, cur_elem, current, parent,
                                      traverse)
                else:
                    assert False, f'In handle_tag_io_control: "{child.tag}" ' \
                                  f'not handled'
            else:
                if child.tag == "literal":
                    _ = ET.SubElement(current, child.tag, child.attrib)
                else:
                    try:
                        _ = self.unnecessary_tags.index(child.tag)
                    except ValueError:
                        assert False, f'In handle_tag_io_control: Empty "' \
                                      f'{child.tag}" not handled'

    def handle_tag_outputs(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the outputs elements.

        <outputs>
        </outputs>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "output":
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, cur_elem, current, parent, traverse)
            elif child.tag == "name":
                self.parseXMLTree(child, current, current, parent, traverse)
            else:
                assert (
                    False
                ), f'In handle_tag_outputs: "{child.tag}" not handled'

    def handle_tag_output(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the output elements.

        <output>
        </output>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag in self.output_child_tags:
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
            else:
                assert (
                    False
                ), f'In handle_tag_outputs: "{child.tag}" not handled'

    def handle_tag_format(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the format elements.

        <format>
        </format>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag == "format-items":
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                if child.tag != "label":
                    assert (
                        False
                    ), f'In handle_tag_format: "{child.tag}" not handled'

    def handle_tag_format_items(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the format_items and its sub-elements

        <format_items>
        ____<format_item>
        ____</format_item>
        </format_items>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if child.tag == "format-items" or child.tag == "format-item":
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                assert (
                    False
                ), f'In handle_tag_format_items: "{child.tag}" not handled'

    def handle_tag_print(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the print tags.

        <print>
        </print>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag != "print-stmt":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
            if child.tag == "outputs":
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_print: "{child.tag}" not handled'

    def handle_tag_open(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the open elements.

        <open>
        </open>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_open: "{child.tag}" not handled'
            else:
                if child.tag == "open-stmt":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_open: Empty elements "{child.tag}" ' \
                       f'ot handled'

    def handle_tag_keyword_arguments(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements between
        the keyword-arguments and keyword-argument elements.

        <keyword-arguments>
        ____<keyword-argument>
        ____</keyword-argument>
        </keyword-arguments>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if (
                        child.tag == "keyword-argument"
                        or child.tag == "literal"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_keyword_arguments: "{child.tag}" not ' \
                       f'handled'
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_keyword_arguments: Empty elements ' \
                       f'"{child.tag}" not handled'

    def handle_tag_read(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the read elements.

        <read>
        </read>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if (
                        child.tag == "io-controls"
                        or child.tag == "inputs"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_read: "{child.tag}" not handled'
            else:
                if child.tag == "read-stmt":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_read: Empty elements "{child.tag}" ' \
                       f'not handled'

    def handle_tag_inputs(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the inputs and input elements.

        <inputs>
        ____<input>
        ____</input>
        </inputs>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if (
                        child.tag == "input"
                        or child.tag == "name"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_input - {root.tag}: "{child.tag}" not ' \
                       f'handled'
            else:
                assert (
                    False
                ), f'In handle_tag_input - {root.tag}: Empty elements ' \
                   f'"{child.tag}" not handled'

    def handle_tag_close(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the close elements.

        <close>
        </close>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_close: "{child.tag}" not handled'
            else:
                if child.tag == "close-stmt":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_close: Empty elements "{child.tag}"'

    def handle_tag_call(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the call elements.

        <call>
        </call>
        """
        self.call_function = True
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "name":
                    # fname: Function name
                    current.attrib['fname'] = child.attrib['id']
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_call: "{child.tag}" not handled'
            else:
                if child.tag == "call-stmt":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_call: Empty elements "{child.tag}"'

        # Update function definition's arguments with array status
        self.update_arguments(current)

    def handle_tag_subroutine(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the subroutine elements.

        <subroutine>
        </subroutine>
        """
        self.argument_types[root.attrib['name']] = {}
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    if child.tag == "body":
                        self.current_body_scope = cur_elem
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_subroutine: "{child.tag}" not handled'
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_subroutine: Empty elements "{child.tag}"'
        # Updating the argument attribute to hold the type.
        for arg in self.arguments_list[current.attrib['name']]:
            if arg.attrib['name'] in self.argument_types[current.attrib['name']]:
                arg.attrib['type'] = str(self.argument_types[current.attrib['name']][arg.attrib['name']])

        # Add extra XMLs under the interface function names to hold the argument types.
        for interface in self.interface_function_xml:
            if current.attrib['name'] in self.interface_function_xml[interface]:
                argument_types = ET.SubElement(
                    self.interface_function_xml[interface][current.attrib['name']],
                    "argument-types"
                )
                num_args = 0
                for arg in self.arguments_list[current.attrib['name']]:
                    num_args += 1
                    argument_type = ET.SubElement(
                        argument_types,
                        "argument-type",
                        {"type": arg.attrib['type']}
                    )
                self.interface_function_xml[interface][current.attrib['name']].attrib['num_args'] = str(num_args)

    def handle_tag_arguments(
            self, root, current, _, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the arguments.

        <arguments>
        </arguments>
        """
        num_of_args = int(root.attrib['count'])
        if self.is_interface_member:
            if grandparent.tag == "subroutine":
                for interface in self.interface_functions:
                    if (
                            grandparent.attrib['name'] in self.interface_functions[interface]
                            and interface in self.interface_xml
                    ):
                        if (
                                "max_arg" not in self.interface_xml[interface].attrib
                                or ("max_arg" in self.interface_xml[interface].attrib
                                    and int(self.interface_xml[interface].attrib['max_arg']) < num_of_args)
                        ):
                            self.interface_xml[interface].attrib['max_arg'] = str(num_of_args)
                        else:
                            pass
                    else:
                        pass
            else:
                assert False, f"Currently, {grandparent.tag} not handled for interface."
                            
        for child in root:
            self.clean_attrib(child)
            if child.tag == "argument":
                # Collect the argument names with None as a initial type.
                # Types will be updated in handle_tag_variable.
                if grandparent.tag == "subroutine":
                    self.argument_types[grandparent.attrib['name']][child.attrib['name']] = None
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                # Set a default array status to False
                cur_elem.attrib['is_array'] = "false"
            else:
                assert (
                    False
                ), f'In handle_tag_variable: "{child.tag}" not handled'

    def handle_tag_argument(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements between the
        argument.

        <argument>
        </argument>
        """
        current.attrib['is_array'] = "false"
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if len(child) > 0 or child.text:
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )

    def handle_tag_if(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the if elements.

        <if>
        </if>
        """
        if traverse == 1:
            current.attrib['if-before-goto'] = \
                str(self.if_appear_before_goto).lower()
        condition = None
        for child in root:
            self.clean_attrib(child)
            if child.text or len(child) > 0:
                if child.tag == "header" or child.tag == "body":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                    if traverse == 1:
                        # Check and hold conditional operation for <goto-stmt>
                        if child.tag == "header":
                            condition = child
                        elif child.tag == "body":
                            if "conditional-goto-stmt-lbl" in current.attrib:
                                if (
                                    "type" not in child.attrib
                                    or child.attrib['type'] != "else"
                                ):
                                    if (
                                        condition is not None
                                        and "code" in current.attrib
                                    ):
                                        unique_code = current.attrib['code']
                                        self.conditional_op[unique_code] = \
                                            condition
                else:
                    assert (
                        False
                    ), f'In handle_tag_if: "{child.tag}" not handled'
            else:
                if child.tag == "if-stmt":
                    current.attrib.update(child.attrib)
                elif child.tag == "body" and traverse > 1:
                    _ = ET.SubElement(current, child.tag, child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_if: Empty elements "{child.tag}" ' \
                       f'not handled'

        # If label appears before <goto>, mark <if>
        # with goto-move to move it later (1st traverse)
        if traverse == 1:
            # Since label_after needs to be reconstructed first,
            # we skip to collect the element if label_after is True
            # Then, once the reconstruction of label_after is done,
            # then we collect those reconstructed elements
            if self.label_before and not self.label_after:
                current.attrib['goto-move'] = "true"
                self.statements_to_reconstruct_before[
                    'stmts-follow-label'].append(current)
            if self.label_after:
                if (
                    self.collect_stmts_after_goto
                    and "conditional-goto-stmt-lbl" not in current.attrib
                    and current.attrib['if-before-goto'] == "false"
                ):
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after[
                        'stmts-follow-goto'].append(current)
                elif self.collect_stmts_after_label:
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after[
                        'stmts-follow-label'].append(current)

    def handle_tag_stop(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the stop elements

        <stop>
        </stop>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "stop-code":
                current.attrib.update(child.attrib)
            else:
                assert (
                    False
                ), f'In handle_tag_stop: "{child.tag}" not handled'

    def handle_tag_step(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the step elements.

        <step>
        </step>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if (
                        child.tag == "operation"
                        or child.tag == "literal"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_step: "{child.tag}" not handled'
            else:
                assert (
                    False
                ), f'In handle_tag_step: Empty elements "{child.tag}" ' \
                   f'not handled'

    def handle_tag_return(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the return and return-stmt elements.
        However, since 'return-stmt' is an empty elements
        with no sub-elements, the function will not keep
        the elements, but move the attribute to its parent
        elements, return.

        <return>
        </return>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "return-stmt":
                current.attrib.update(child.attrib)
            else:
                assert (
                    False
                ), f'In handle_tag_return: "{child.tag}" not handled'

    def handle_tag_function(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the function elements.

        <function>
        </function>
        """
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                    if child.tag == "header":
                        self.is_function_arg = True
                    elif child.tag == "body":
                        self.current_body_scope = cur_elem
                else:
                    assert (
                        False
                    ), f'In handle_tag_function: "{child.tag}" not handled'
            else:
                if (
                        child.tag == "function-stmt"
                        or child.tag == "end-function-stmt"
                        or child.tag == "function-subprogram"
                ):
                    cur_elem = ET.SubElement(
                        current, child.tag, child.attrib
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_function: Empty elements "{child.tag}"'

    def handle_tag_use(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the use elements.

        <use>
        </use>
        """
        assert (
            isfile(self.module_log_file_path)
        ), f"Module log file path must exist."

        with open(self.module_log_file_path) as json_f:
            module_logs = json.load(json_f)

        file_to_mod_mapper = module_logs["file_to_mod"]
        mod_to_file_mapper = module_logs["mod_to_file"]

        use_module = root.attrib['name']
        if use_module.lower() in mod_to_file_mapper:
            use_module_file_path = mod_to_file_mapper[use_module.lower()]
            if (
                    use_module_file_path[0] !=
                    self.original_fortran_file_abs_path
                    and use_module not in self.modules_in_file
            ):
                self.module_files_to_process.append(use_module_file_path[0])
            else:
                # If module resides in the same file, we don't have to do
                # anything. Handling for this case is already implemented in
                # genPGM.py
                pass
        else:
            pass

        for child in root:
            self.clean_attrib(child)
            if child.tag == "use-stmt" or child.tag == "only":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                assert (
                    False
                ), f'In handle_tag_use: "{child.tag}" not handled'

    def handle_tag_module(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the module elements.

        <module>
        </module>
        """
        for child in root:
            self.clean_attrib(child)

            try:
                _ = self.module_child_tags.index(child.tag)
            except ValueError:
                assert False, f'In handle_tag_module: "{child.tag}" not handled'

            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if len(child) > 0 or child.text:
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
        self.modules_in_file.append(root.attrib["name"])

    def handle_tag_initial_value(
            self, root, current, parent, _, traverse
    ):
        """This function handles cleaning up the XML elements
        between the initial-value elements.

        <initial-value>
        </initial-value>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    assert (
                        False
                    ), f'In handle_tag_initial_value: "{child.tag}" not handled'
            else:
                if child.tag == "initialization":
                    current.attrib.update(child.attrib)
                else:
                    assert (
                        False
                    ), f'In handle_tag_initial_value: Empty elements ' \
                       f'"{child.tag}"'

    def handle_tag_members(
            self, root, current, parent, grandparnet, traverse
    ):
        """This function handles cleaning up the XML elements
        between the members elements.

        <members>   or    <member>
        </members>        </member>
        """
        self.is_interface_member = True
        for child in root:
            self.clean_attrib(child)

            try:
                error_chk = self.members_child_tags.index(child.tag)
            except ValueError:
                assert (
                    False
                ), f'In handle_tag_members: "{child.tag}" not handled'

            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if len(child) > 0 or child.text:
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )
        # Re-initialize to false when exiting members
        self.is_interface_member = False

    def handle_tag_only(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the only elements.

        <only>
        </only>
        """
        for child in root:
            try:
                error_chk = self.only_child_tags.index(child.tag)
            except ValueError:
                assert (
                    False
                ), f'In handle_tag_only: "{child.tag}" not handled'

            cur_elem = ET.SubElement(
                current, child.tag, child.attrib
            )
            if len(child) > 0 or child.text:
                self.parseXMLTree(
                    child, cur_elem, current, parent, traverse
                )

    def handle_tag_length(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the length elements.

        <length>
        </length>
        """
        for child in root:
            if child.tag in ["literal", "char-length", "type-param-value"]:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                assert (
                    False
                ), f'In handle_tag_length: "{child.tag}" not handled'

    def handle_tag_interface(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles rectifying the elements between
        interface tag.

        <interface>
        </interface>
        """
        self.is_interface = True
        for child in root:
            if child.tag == "header" or child.tag == "body":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                assert (
                    child.tag in self.unnecessary_tags
                ), f'In handle_tag_length: "{child.tag}" not handled'

        self.interface_xml[self.cur_interface_name] = current
        # Re-initializing for next interface use
        self.cur_interface_name = None
        self.is_interface = False

    def handle_tag_select(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the select elements.

        <select>
        </select>
        """
        for child in root:
            self.clean_attrib(child)

            if child.tag in self.select_child_tags:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                assert (
                    child.tag in self.unnecessary_tags
                ), f'In handle_tag_length: "{child.tag}" not handled'

                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_select: Empty elements "{child.tag}"'

    def handle_tag_case(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the case elements.

        <case>
        </case>
        """
        for child in root:
            self.clean_attrib(child)

            if child.tag in self.case_child_tags:
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_case: Empty elements "{child.tag}"'

    def handle_tag_value_ranges(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the value-ranges elements.

        <value-ranges>
        </value-ranges>
        """
        for child in root:
            self.clean_attrib(child)

            if child.tag == "value-range":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_value_ranges: Empty eleme' \
                       f'nts "{child.tag}"'

    def handle_tag_value_range(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the value-range elements.

        <value-range>
        </value-range>
        """
        for child in root:
            self.clean_attrib(child)

            if child.tag == "value":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
                else:
                    current.remove(cur_elem)
                    if len(root) == 1:
                        parent.remove(current)
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_value_range: Empty elements ' \
                       f'"{child.tag}"'

    def handle_tag_values(
            self, root, current, parent, grandparent, traverse
    ):
        """This function handles cleaning up the XML elements
        between the values elements.

        <values>
        </values>
        """
        for child in root:
            self.clean_attrib(child)

            if child.tag == "literal":
                cur_elem = ET.SubElement(
                    current, child.tag, child.attrib
                )
                if len(child) > 0 or child.text:
                    self.parseXMLTree(
                        child, cur_elem, current, parent, traverse
                    )
            else:
                try:
                    _ = self.unnecessary_tags.index(child.tag)
                except ValueError:
                    assert (
                        False
                    ), f'In handle_tag_values: Empty elements "{child.tag}"'

        # For DATA statements, further indentation might be required
        target_child = None
        delete_child = []
        for child in current:
            if len(child) == 0:
                target_child = child
            else:
                cur_elem = ET.SubElement(
                    target_child, child.tag, child.attrib
                )
                delete_child.append(child)

        for child in delete_child:
            current.remove(child)

    #################################################################
    #                                                               #
    #                       XML TAG PARSER                          #
    #                                                               #
    #################################################################

    def parseXMLTree(
            self, root, current, parent, grandparent, traverse
    ):
        """Recursively traverse through the nested XML AST tree and
        calls appropriate tag handler, which will generate
        a cleaned version of XML tree for translate.py.
        Any new tags handlers must be added under this this function.

        parseXMLTree

        Arguments:
            root: The current root of the tree.
            current: Current element.
            parent: Parent element of the current.
            grandparent: A parent of parent statement of current.
            traverse: Keeps the track of number of traverse time.

        Returns:
            None
        """
        if root.tag == "file":
            self.handle_tag_file(root, current, parent, grandparent, traverse)
        elif root.tag == "program":
            self.handle_tag_program(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "header":
            self.handle_tag_header(root, current, parent, grandparent, traverse)
        elif root.tag == "specification":
            self.handle_tag_specification(root, current, parent, grandparent,
                                          traverse)
        elif root.tag == "body":
            self.handle_tag_body(root, current, parent, grandparent, traverse)
        elif root.tag == "declaration":
            self.handle_tag_declaration(root, current, parent, grandparent,
                                        traverse)
        elif root.tag == "type":
            self.handle_tag_type(root, current, parent, grandparent, traverse)
        elif root.tag == "variables":
            self.handle_tag_variables(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "variable":
            self.handle_tag_variable(root, current, parent, grandparent,
                                     traverse)
        elif root.tag == "statement":
            self.handle_tag_statement(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "assignment":
            self.handle_tag_assignment(root, current, parent, grandparent,
                                       traverse)
        elif root.tag == "target":
            self.handle_tag_target(root, current, parent, grandparent, traverse)
        elif root.tag == "value":
            self.handle_tag_value(root, current, parent, grandparent, traverse)
        elif root.tag == "names":
            self.handle_tag_names(root, current, parent, grandparent, traverse)
        elif root.tag == "name":
            self.handle_tag_name(root, current, parent, grandparent, traverse)
        elif root.tag == "literal":
            self.handle_tag_literal(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "dimensions":
            self.handle_tag_dimensions(root, current, parent, grandparent,
                                       traverse)
        elif root.tag == "dimension":
            self.handle_tag_dimension(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "loop":
            self.handle_tag_loop(root, current, parent, grandparent, traverse)
        elif root.tag == "index-variable" or root.tag == "range":
            self.handle_tag_index_variable_or_range(root, current, parent,
                                                    grandparent, traverse)
        elif root.tag == "lower-bound" or root.tag == "upper-bound":
            self.handle_tag_bound(root, current, parent, grandparent, traverse)
        elif root.tag == "subscripts":
            self.handle_tag_subscripts(root, current, parent, grandparent,
                                       traverse)
        elif root.tag == "subscript":
            self.handle_tag_subscript(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "operation":
            self.handle_tag_operation(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "operand":
            self.handle_tag_operand(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "write":
            self.handle_tag_write(root, current, parent, grandparent, traverse)
        elif root.tag == "io-controls":
            self.handle_tag_io_controls(root, current, parent, grandparent,
                                        traverse)
        elif root.tag == "io-control":
            self.handle_tag_io_control(root, current, parent, grandparent,
                                       traverse)
        elif root.tag == "outputs":
            self.handle_tag_outputs(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "output":
            self.handle_tag_output(root, current, parent, grandparent, traverse)
        elif root.tag == "format":
            self.handle_tag_format(root, current, parent, grandparent, traverse)
        elif root.tag == "format-items" or root.tag == "format-item":
            self.handle_tag_format_items(root, current, parent, grandparent,
                                         traverse)
        elif root.tag == "print":
            self.handle_tag_print(root, current, parent, grandparent, traverse)
        elif root.tag == "open":
            self.handle_tag_open(root, current, parent, grandparent, traverse)
        elif root.tag == "keyword-arguments" or root.tag == "keyword-argument":
            self.handle_tag_keyword_arguments(root, current, parent,
                                              grandparent, traverse)
        elif root.tag == "read":
            self.handle_tag_read(root, current, parent, grandparent, traverse)
        elif root.tag == "inputs" or root.tag == "input":
            self.handle_tag_inputs(root, current, parent, grandparent, traverse)
        elif root.tag == "close":
            self.handle_tag_close(root, current, parent, grandparent, traverse)
        elif root.tag == "call":
            self.handle_tag_call(root, current, parent, grandparent, traverse)
        elif root.tag == "subroutine":
            self.handle_tag_subroutine(root, current, parent, grandparent,
                                       traverse)
        elif root.tag == "arguments":
            self.handle_tag_arguments(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "if":
            self.handle_tag_if(root, current, parent, grandparent, traverse)
        elif root.tag == "stop":
            self.handle_tag_stop(root, current, parent, grandparent, traverse)
        elif root.tag == "step":
            self.handle_tag_step(root, current, parent, grandparent, traverse)
        elif root.tag == "return":
            self.handle_tag_return(root, current, parent, grandparent, traverse)
        elif root.tag == "function":
            self.handle_tag_function(root, current, parent, grandparent,
                                     traverse)
        elif root.tag == "use":
            self.handle_tag_use(root, current, parent, grandparent, traverse)
        elif root.tag == "module":
            self.handle_tag_module(root, current, parent, grandparent, traverse)
        elif root.tag == "initial-value":
            self.handle_tag_initial_value(root, current, parent, grandparent,
                                          traverse)
        elif root.tag == "members":
            self.handle_tag_members(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "only":
            self.handle_tag_only(root, current, parent, grandparent, traverse)
        elif root.tag == "length":
            self.handle_tag_length(root, current, parent, grandparent, traverse)
        elif root.tag == "saved-entity":
            self.handle_tag_saved_entity(root, current, parent, grandparent,
                                         traverse)
        elif root.tag == "save-stmt":
            self.handle_tag_save_statement(root, current, parent, grandparent,
                                           traverse)
        elif root.tag == "constants":
            self.handle_tag_constants(root, current, parent, grandparent,
                                      traverse)
        elif root.tag == "constant":
            self.handle_tag_constant(root, current, parent, grandparent,
                                     traverse)
        elif root.tag == "argument":
            self.handle_tag_argument(root, current, parent, grandparent,
                                     traverse)
        elif root.tag == "interface":
            self.handle_tag_interface(root, current, parent, grandparent,
                                    traverse)
        elif root.tag == "select":
            self.handle_tag_select(root, current, parent, grandparent,
                                   traverse)
        elif root.tag == "case":
            self.handle_tag_case(root, current, parent, grandparent,
                                 traverse)
        elif root.tag == "value-ranges":
            self.handle_tag_value_ranges(root, current, parent, grandparent,
                                         traverse)
        elif root.tag == "value-range":
            self.handle_tag_value_range(root, current, parent, grandparent,
                                        traverse)
        elif root.tag == "values":
            self.handle_tag_values(root, current, parent, grandparent, traverse)
        else:
            assert (
                False
            ), f"In parseXMLTree: <{root.tag}> passed from <{parent.tag}> " \
               f"not supported"

    #################################################################
    #                                                               #
    #                       RECONSTRUCTORS                          #
    #                                                               #
    #################################################################

    def reconstruct_derived_type_declaration(self):
        """This function reconstructs the derived type
        with the collected derived type declaration
        elements in the handle_tag_declaration and
        handle_tag_type.

        Args:
            None.

        Returns:
            None.
        """
        if self.derived_type_var_holder_list:
            size = ET.Element("")
            is_dimension = False

            # Since component-decl-list appears after component-decl,
            # the program needs to iterate the list once first to
            # pre-collect the variable counts.
            counts = []
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "component-decl-list":
                    counts.append(elem.attrib['count'])

            # Initialize count to 0 for <variables> count attribute.
            count = 0
            dim = 0
            # 'component-decl-list__begin' tag is an indication
            # of all the derived type member variable
            # declarations will follow.
            derived_type = ET.SubElement(self.parent_type, "derived-types")
            literal_value = None
            str_value = None
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "intrinsic-type-spec":
                    keyword2 = ""
                    if elem.attrib['keyword2'] == "":
                        keyword2 = "none"
                    else:
                        keyword2 = elem.attrib['keyword2']
                    attributes = {
                        "hasKind": "false",
                        "hasLength": "false",
                        "name": elem.attrib['keyword1'],
                        "is_derived_type": str(self.is_derived_type),
                        "keyword2": keyword2,
                    }
                    newType = ET.SubElement(derived_type, "type", attributes)
                    if newType.attrib['name'].lower() == "character":
                        assert (
                            literal_value != None
                        ), "Literal value (String length) for character cannot be None."
                        newType.set("string_length", literal_value)
                        literal_value = None  # Reset literal_value to None
                elif elem.tag == "derived-type-spec":
                    attributes = {
                        "hasKind": "false",
                        "hasLength": "false",
                        "name": elem.attrib['typeName'],
                        "is_derived_type": str(self.is_derived_type),
                        "keyword2": "none",
                    }
                    newType = ET.SubElement(derived_type, "type", attributes)
                elif (
                        elem.tag == "literal"
                        or elem.tag == "name"
                ):
                    value = elem
                    if elem.tag == "literal":
                        tag_name = "literal"
                        literal_value = elem.attrib['value']
                    else:
                        tag_name = "name"
                elif elem.tag == "component-array-spec":
                    is_dimension = True
                    dim += 1
                elif (
                    elem.tag == "component-decl-list__begin"
                    and not is_dimension

                ):
                    if len(counts) > count:
                        attr = {"count": counts[count]}
                        new_variables = ET.SubElement(
                            derived_type, "variables", attr
                        )  # <variables _attribs_>
                        count += 1
                elif elem.tag == "operation":
                    str_value = ""
                    for op in elem.iter():
                        if op.tag == "char-literal-constant":
                            str_value += op.attrib['str']
                    str_value = str_value.replace('"','')
                elif elem.tag == "component-decl":
                    if not is_dimension:
                        var_attribs = {
                            "has_initial_value": elem.attrib[
                                "hasComponentInitialization"
                            ],
                            "name": elem.attrib['id'],
                            "is_array": "false",
                        }
                        # Store variable name in the non array tracker
                        self.declared_non_array_vars.update(
                            {elem.attrib['id']: self.current_scope}
                        )
                        new_variable = ET.SubElement(
                            new_variables, "variable", var_attribs
                        )  # <variable _attribs_>
                        if elem.attrib['hasComponentInitialization'] == "true":
                            init_value_attrib = ET.SubElement(
                                new_variable, "initial-value"
                            )
                            if str_value:
                                value.attrib['value'] = str_value
                                str_value = None
                            new_size = ET.SubElement(
                                init_value_attrib, tag_name, value.attrib
                            )  # <initial-value _attribs_>
                    else:
                        total_number_of_arrays = len(self.derived_type_array_dimensions)
                        new_dimensions = ET.SubElement(
                            derived_type, "dimensions", {"count": "1"}
                        )  # <dimensions count="1">

                        if self.derived_type_array_dimensions[dim]:
                            new_dimension = ET.SubElement(
                                new_dimensions, "dimension", {"type": "simple"}
                            )  # <dimension type="simple">
                            has_lower_bound = False
                            new_range = ET.SubElement(new_dimension, "range")
                            num_of_dimensions = len(self.derived_type_array_dimensions[dim])
                            for s in range(0, num_of_dimensions):
                                value = self.derived_type_array_dimensions[dim][s]
                                if value.tag == "literal":
                                    tag_name = "literal"
                                elif value.tag == "name":
                                    tag_name = "name"
                                    value.attrib["is_derived_type_ref"] = "true"
                                else:
                                    pass

                                need_new_dimension = False 
                                need_upper_bound = False
                                if (
                                        len(self.derived_type_array_dimensions[dim]) == 1
                                        or (not has_lower_bound 
                                            and ((s+1) < len(self.derived_type_array_dimensions[dim])
                                            and value.attrib["dim-number"] != self.derived_type_array_dimensions[dim][s+1].attrib["dim-number"]))
                                ):
                                    if (
                                            (s+1) < len(self.derived_type_array_dimensions[dim])
                                            and value.attrib["dim-number"] != self.derived_type_array_dimensions[dim][s+1].attrib["dim-number"]
                                    ):
                                        need_new_dimension = True
                                    upper_bound_value = copy.copy(value)
                                    upper_bound_tag_name = tag_name
                                    tag_name = "literal"
                                    value.tag = "literal"
                                    value.attrib = {
                                            "dim-number": value.attrib["dim-number"],
                                            "type": "int",
                                            "value": "0"
                                    }
                                    need_upper_bound = True

                                if not has_lower_bound:
                                    bound = ET.SubElement(new_range, "lower-bound")
                                    has_lower_bound = True
                                else:
                                    bound = ET.SubElement(new_range, "upper-bound")
                                    has_lower_bound = False

                                new_range_value = ET.SubElement(bound, tag_name, value.attrib)

                                if need_upper_bound:
                                    bound = ET.SubElement(new_range, "upper-bound")
                                    new_range_value = ET.SubElement(bound, upper_bound_tag_name, upper_bound_value.attrib)
                                    has_lower_bound = False

                                if need_new_dimension:
                                    new_dimension = ET.SubElement(
                                        new_dimensions, "dimension", {"type": "simple"}
                                    )  # <dimension type="simple">
                                    new_range = ET.SubElement(new_dimension, "range")
                                    need_new_dimension = False

                        if len(counts) > count:
                            attr = {"count": counts[count]}
                            new_variables = ET.SubElement(
                                derived_type, "variables", attr
                            )
                            count += 1
                        var_attribs = {
                            "has_initial_value": elem.attrib[
                                "hasComponentInitialization"
                            ],
                            "name": elem.attrib['id'],
                            "is_array": "true",
                        }
                        # Store variable name in the array tracker
                        self.declared_array_vars.update(
                            {elem.attrib['id']: self.current_scope}
                        )
                        new_variable = ET.SubElement(
                            new_variables, "variable", var_attribs
                        )
                        is_dimension = False

            # Once one derived type was successfully constructed,
            # clear all the elements of a derived type list
            self.derived_type_var_holder_list.clear()

    def reconstruct_derived_type_ref(self, current):
        """This function reconstruct the id into x.y.k form
        from the messy looking id. One thing to notice is
        that this new form was generated in the python syntax,
        so it is a pre-process for translate.py and
        even pyTranslate.py that

        Args:
            current (:obj: 'ET'): Current element object.

        Returns:
            None.
        """
        assert (
                current.tag == "name"
        ), f"The tag <name> must be passed to reconstruct_derived_type_ref.\
             Currently, it's {current.tag}."
        # First the root <name> id gets the very first
        # variable reference i.e. x in x.y.k (or x%y%k in Fortran syntax)
        current.attrib['id'] = self.derived_type_var_holder_list[0]
        if (
                current.attrib['id'] in self.declared_array_vars
                and self.declared_array_vars[current.attrib['id']]
                == self.current_scope
        ):
            current.attrib['hasSubscripts'] = "true"
            current.attrib['is_array'] = "true"
        else:
            current.attrib['hasSubscripts'] = "false"
            current.attrib['is_array'] = "false"

        number_of_vars = len(self.derived_type_var_holder_list)
        attributes = {}
        parent_ref = current
        self.derived_type_refs.append(parent_ref)
        for var in range(1, number_of_vars):
            variable_name = self.derived_type_var_holder_list[var]
            attributes.update(current.attrib)
            attributes['id'] = variable_name
            if (
                    variable_name in self.declared_array_vars
                    and self.declared_array_vars[variable_name]
                    == self.current_scope
            ):
                attributes['hasSubscripts'] = "true"
                attributes['is_array'] = "true"
            else:
                attributes['is_array'] = "false"
            # Create N (number_of_vars) number of new subElement
            # under the root <name> for each referencing variable
            reference_var = ET.SubElement(
                parent_ref, "name", attributes
            )
            parent_ref = reference_var
            self.derived_type_refs.append(parent_ref)
        self.derived_type_var_holder_list.clear()  # Clean up the list for re-use

    def reconstruct_format(self, grandparent, traverse):
        """This function is for reconstructing the <format>
        under the <statement> element.

        The OFP XML nests formats under:
            (1) statement
            (2) declaration
            (3) loop

        tags, which are wrong except one that is declared
        under the statement. Therefore, those formats
        declared under (2) and (3) will be extracted
        and reconstructed to be nested under (1)
        in this function.

        Args:
            grandparent (:obj: 'ET'): Grand parent element object.
            traverse (int): Current traverse number.

        Returns:
            None.
        """
        root_scope = ET.SubElement(self.current_body_scope, "statement")
        for form in self.format_holder:
            cur_elem = ET.SubElement(root_scope, form.tag, form.attrib)
            self.parseXMLTree(
                form, cur_elem, root_scope, grandparent, traverse
            )

    def reconstruct_derived_type_names(self, current):
        """This function reconstructs derived type
        reference syntax tree. However, this functions is
        actually a preprocessor for the real final reconstruction.

        Args:
            current (:obj: 'ET'): Current element object.

        Returns:
            None.
        """
        # Update reconstruced derived type references
        assert (
                self.is_derived_type_ref == True
        ), "'self.is_derived_type_ref' must be true"
        numPartRef = int(current.attrib['numPartRef'])
        for idx in range(1, len(self.derived_type_refs)):
            self.derived_type_refs[idx].attrib.update(
                {"numPartRef": str(numPartRef)}
            )
        # Re-initialize to original values
        self.derived_type_refs.clear()

    def reconstruct_name_element(self, cur_elem, current):
        """This function performs a final reconstruction of
        derived type name element that was preprocessed by
        'reconstruct_derived_type_names' function.
        This function traverses the preprocessed name element
        (including sub-elements) and split & store <name> and
        <subscripts> into separate lists. Then, it comibines
        and reconstructs two lists appropriately.

        Args:
            cur_elem (:obj: 'ET'): Newly generated element
            for current element object.
            current (:obj: 'ET'): Current element object.

        Returns:
            None.
        """
        name_elements = [cur_elem]
        # Remove the original <name> elements.
        current.remove(cur_elem)
        # Split & Store <name> element and <subscripts>.
        subscripts_holder = []
        for child in cur_elem:
            if child.tag == "subscripts":
                subscripts_holder.append(child)
            else:
                name_elements.append(child)
                for third in child:
                    name_elements.append(third)

        # Combine & Reconstruct <name> element.
        subscript_num = 0
        cur_elem = ET.SubElement(
            current, name_elements[0].tag, name_elements[0].attrib
        )
        cur_elem.attrib['is_derived_type_ref'] = "true"
        if cur_elem.attrib['hasSubscripts'] == "true":
            cur_elem.append(subscripts_holder[subscript_num])
            subscript_num += 1

        numPartRef = int(cur_elem.attrib['numPartRef']) - 1
        name_element = ET.Element("")
        for idx in range(1, len(name_elements)):
            name_elements[idx].attrib['numPartRef'] = str(numPartRef)
            numPartRef -= 1
            name_element = ET.SubElement(
                cur_elem, name_elements[idx].tag, name_elements[idx].attrib
            )
            name_element.attrib['is_derived_type_ref'] = "true"
            # In order to handle the nested subelements of <name>,
            # update the cur_elem at each iteration.
            cur_elem = name_element
            if name_elements[idx].attrib['hasSubscripts'] == "true":
                name_element.append(subscripts_holder[subscript_num])
                subscript_num += 1

        # Clean out the lists for recyling.
        # This is not really needed as they are local lists,
        # but just in case.
        name_elements.clear()
        subscripts_holder.clear()
        self.need_reconstruct = False

    def reconstruct_goto_after_label(
            self, parent, traverse, reconstruct_target
    ):
        """This function gets called when goto appears
        after the corresponding label and all necessary
        statements are collected for the reconstruction.

        Args:
            parent (:obj: 'ET'): A parent ET object that current
            element will be nested under.
            header (list): A header tht holds conditional header.
            traverse (int): A traverse counter.
            reconstruct_target (dict): A dictionary that holds statements
            for goto and label as well as the number of goto counter.

        Return:
            None.
        """
        number_of_gotos = reconstruct_target['count-gotos']
        stmts_follow_goto = reconstruct_target['stmts-follow-goto']
        stmts_follow_label = reconstruct_target['stmts-follow-label']

        header = [None]
        self.check_conditional_goto(header, stmts_follow_goto)

        # Corrent boundaries of gotos in case of a multiple
        # nested gotos.
        self.goto_boundary_corrector(
                reconstruct_target, stmts_follow_goto,
                stmts_follow_label
        )

        # Check for the case where goto and label are
        # at different lexical levels
        self.handle_in_outward_movement(stmts_follow_goto, stmts_follow_label, parent)

        if not self.conditional_goto:
            declared_goto_flag_num = []
            self.generate_declaration_element(
                parent, "goto_flag", number_of_gotos,
                declared_goto_flag_num, traverse
            )

        # This variable is for storing goto that may appear
        # at the end of if because we want to extract one 
        # scope out and place it right after
        # the constructed if-statement
        next_goto = []
        reconstructed_goto_elem = []
        for i in range(number_of_gotos):
            # Constructor for statements if and statements nested
            self.reconstruct_stmts_follow_goto_after_case(
                    header, parent, stmts_follow_goto, next_goto,
                    traverse, reconstructed_goto_elem, i
            )
            # Constructor for statements with L_i:stmt_n
            self.reconstruct_stmts_follow_label_after_case(
                        stmts_follow_label, next_goto, reconstructed_goto_elem,
                        header, traverse, parent, i
            )
            # When unconditional goto, it generates 'goto_flag_i = False'
            # statement at the end of reconstrcted goto statement.
            # Else, nothing gets printed, but set self.conditional_goto to False
            if not self.conditional_goto:
                statement = ET.SubElement(parent, "statement")
                self.generate_assignment_element(statement, \
                        f"goto_flag_{i+1}", None, "literal", "false", traverse)
                reconstructed_goto_elem.append(statement)
                parent.remove(statement)
            else:
                self.conditional_goto = False

            if len(reconstructed_goto_elem) > 1:
                stmts_follow_label = reconstructed_goto_elem[1]

            self.encapsulate_under_do_while = False

            # next_goto holds another goto after the current label_after
            # case label, which will encapsulate reconstrcted goto element
            if next_goto:
                self.reconstruct_next_goto(next_goto, reconstructed_goto_elem, parent)

        # Set all holders and checkers (markers) to default
        self.label_after = False
        self.goto_under_if = False
        self.reconstruct_after_case_now = False
        self.reconstruction_for_after_done = True
        self.goto_target_lbl_after.clear()
        self.label_lbl_for_after.clear()
        self.statements_to_reconstruct_after.clear()

    def reconstruct_goto_before_label(
            self, parent, traverse, reconstruct_target
    ):
        """This function gets called when goto appears
        before the corresponding label and all necessary
        statements are collected for the reconstruction.

        Args:
            parent (:obj: 'ET'): A parent ET object that current
            element will be nested under.
            traverse (int): A traverse counter.
            reconstruct_target (dict): A dictionary that holds statements
            for goto and label as well as the number of goto counter.

        Return:
            None.
        """
        stmts_follow_label = reconstruct_target['stmts-follow-label']
        number_of_gotos = reconstruct_target['count-gotos']

        # This removes the statement that's a child statement of
        # if body being seprately re-added to the list.
        self.remove_dup_stmt(stmts_follow_label)

        # Declare label flag for loop condition
        declared_label_flag_num = []
        self.generate_declaration_element(
            parent, "label_flag", number_of_gotos,
            declared_label_flag_num, traverse
        )

        # Find the boundary from label to goto.
        # Remove any statements that are not within the boundary.
        goto_index_holder = []
        target_label_lbl = [None]
        statements_to_recover =\
            self.boundary_identifier_for_backward_goto(
                    stmts_follow_label, goto_index_holder,
                    number_of_gotos, target_label_lbl
            )

        # In case of multiple goto statements appears,
        # slice them into N number of list objects
        # The location of goto statement (inner to outer)
        # is represented by the increament of index
        # i.e. [0]: innermost, [N]: Outermost
        multiple_goto_stmts = []
        self.multiple_goto_identifier(
                goto_index_holder, multiple_goto_stmts,
                stmts_follow_label
        )

        # Check whether there is nested label_after
        # case goto statements. Handles one case
        # at a time.
        nested_gotos_exist =\
                self.nested_forward_goto_identifier(multiple_goto_stmts)

        # Generate loop ast
        self.construct_goto_loop(
                parent, reconstruct_target,
                nested_gotos_exist, multiple_goto_stmts,
                number_of_gotos, declared_label_flag_num,
                traverse, target_label_lbl)

        # Recover rest of the statements
        self.statement_recovery(statements_to_recover, parent, traverse)

        # Set all holders and checkers (markers) to default
        self.label_before = False
        self.reconstruct_before_case_now = False
        self.reconstruct_for_before_done = True
        self.label_lbl_for_before.clear()
        self.statements_to_reconstruct_before['stmts-follow-label'] = []
        self.statements_to_reconstruct_before['count-gotos'] = 0

    def reconstruct_header(
            self, temp_elem_holder, parent
    ):
        """This function is for reconstructing the oddly
        generated header AST to have an uniform structure
        with other multiary type operation nested headers.

        Args:
            temp_elem_holder (list): A temporary holder that
            holds statements under header for swap.
                parent (:obj: 'ET'): A parent ET object that current

        Return:
            None.
        """
        # This operation is basically for switching
        # the location of operator and 2nd operand,
        # so the output syntax can have a common structure
        # with other operation AST
        op = temp_elem_holder.pop()
        temp_elem_holder.insert(1, op)

        # First create <operation> element
        # Currently, only assume multiary reconstruction
        operation = ET.SubElement(
                parent, "operation", {"type":"multiary"}
        )
        for elem in temp_elem_holder:
            if elem.tag == "name" or elem.tag == "literal":
                operand = ET.SubElement(operation, "operand")
                value = ET.SubElement(operand, elem.tag, elem.attrib)
            else:
                assert (
                        elem.tag == "equiv-operand__equiv-op"
                ), f"Tag must be 'equiv-operand__equiv-op'. Current: {elem.tag}."
                operator = ET.SubElement(
                        operation, "operator", {"operator":elem.attrib['equivOp']}
                )
            parent.remove(elem)

    def goto_boundary_corrector(self, reconstruct_target, stmts_follow_goto, stmts_follow_label):
        """This function is for correcting the boundaries of goto
        statements in case of a multiple gotos are nested and
        crossing each other.

        Args:
            reconstruct_target (dict): A dictionary that holds statements
            for goto and label as well as the number of goto counter.
            stmts_follow_goto (list): A list that holds statements
            after goto statement.
            stmts_follow_label (list): A list that holds statements
            after label statements.

        Return:
            None.
        """
        # If [0] <goto-stmt> is an inner scope statement of the [N-1]
        # <goto-stmt>in the stmts_follow_goto, then we need to correct
        # the boundary issue by moving the [N-1] element to 
        # the end of stmts_follow_label
        last_stmt = reconstruct_target['stmts-follow-goto'][-1]
        if "goto-stmt" in last_stmt.attrib:
            first_goto = reconstruct_target['stmts-follow-goto'][0]
            if last_stmt.attrib['lbl'] == first_goto.attrib['parent-goto']:
                last_goto = reconstruct_target['stmts-follow-goto'].pop()
                last_goto.attrib['next-goto'] = "true"
                stmts_follow_label.append(last_goto)

        goto_and_label_stmts_after_goto = []
        for stmt in stmts_follow_goto:
            if "label" in stmt.attrib:
                goto_and_label_stmts_after_goto.append(stmt.attrib['label'])
            elif "goto-stmt" in stmt.attrib:
                goto_and_label_stmts_after_goto.append(stmt.attrib['lbl'])

        num_of_goto_and_label_after_label = 0
        index = 0
        for stmt in stmts_follow_label:
            if (
                    "label" in stmt.attrib
                    or "goto-stmt" in stmt.attrib
            ):
                num_of_goto_and_label_after_label += 1
                # Since the first label-statement of
                # stmts_follow_label is always a match
                # for the first goto-statement in the
                # stmt_follow_goto in the label_after case,
                # remove the goto-move (label_before) case mark
                if (
                        index == 0
                        and "goto-move" in stmt.attrib
                ):
                    del stmt.attrib['goto-move']

        # -2 disregarding the first and last statements
        num_of_goto_and_label_after_label -= 2

        for i in range(num_of_goto_and_label_after_label):
            stmt = stmts_follow_label.pop(-2)
            stmts_follow_goto.append(stmt)

    def reconstruct_stmts_follow_goto_after_case(
            self, header, parent, stmts_follow_goto,
            next_goto, traverse, reconstructed_goto_elem,
            index
    ):
        """This function generates a new if statement to
        nests statements that follow goto-stmt based on
        condition or non-condition status to eliminate
        goto.

        Args:
            next_goto (list): A list to hold next goto-stmt that may exist
            within the boundary of current goto.
            reconstructed_goto_elem (list): A list that will hold
            reconstructed if statements.
            header (list): A header tht holds conditional header.
            parent (:obj: 'ET'): A parent ET object that current
            element will be nested under.
            stmts_follow_goto (list): A list that holds statements
            within the boundary of currently handling goto.
            traverse (int): A current traverse counter.
            index (int): An index of goto.

        Return:
            None.
        """

        if self.conditional_goto:
            if not self.outward_move and not self.inward_move:
                self.need_op_negation = True
            if header[0] is not None:
                self.generate_if_element(
                        header[0], parent, stmts_follow_goto, next_goto, True, None,
                        None, None, None, traverse, reconstructed_goto_elem
                )
            elif self.outward_move:
                for stmt in stmts_follow_goto:
                    if "skip-collect" not in stmt.attrib:
                        cur_elem = ET.SubElement (parent, stmt.tag, stmt.attrib)
                        self.parseXMLTree(stmt, cur_elem, stmt, parent, traverse)
            else:
                assert (
                        False
                ), "Currently inward movement for goto is not being handled."
        else:
            self.generate_if_element(
                None, parent, stmts_follow_goto, next_goto, True, "unary",
                f"goto_flag_{index + 1}", None, ".not.", traverse,
                reconstructed_goto_elem
            )

        if reconstructed_goto_elem:
            stmts_follow_goto = reconstructed_goto_elem[0]

    def handle_in_outward_movement(self, stmts_follow_goto, stmts_follow_label, parent):
        """This function checks the lexical level of goto and label.
        Then, generate and add (remove) statements to the statement
        holders, so they can be handled appropriately.

        Args:
            stmts_follow_goto (list): It holds all the statements
            that appeared after the goto statement in the original
            code.
            stmts_follow_label (list): It holds all the statements
            that appeared after the label statement in the original
            code.
            parent (:obj: 'ET'): A parent ET object that current
            element will be nested under.

        Returns:
            None.
        """
        body_levels = {}
        for goto_stmt in stmts_follow_goto:
            # If the statements are in different level,
            # we do not want to have them in the stmts_follow_goto,
            # so check such case and remove anything follow goto-stmt.
            if (
                self.outward_move
                and ("generated-exit-stmt" not in goto_stmt.attrib
                     and "goto-stmt" not in goto_stmt.attrib)
            ):
                stmts_follow_goto.remove(goto_stmt)

            if "goto-stmt" in goto_stmt.attrib:
                lbl = goto_stmt.attrib['lbl']
                body_levels[lbl] = goto_stmt.attrib['body-level']
                for label_stmt in stmts_follow_label:
                    if 'target-label-statement' in label_stmt.attrib:
                        label = label_stmt.attrib['label']
                        label_body_level = label_stmt.attrib['body-level']
                        # A goto-forward case where goto and label are
                        # located in different levels
                        if (
                            label in body_levels
                            and body_levels[label] != label_body_level
                        ):
                            if self.body_level_rank[label_body_level]\
                                    < self.body_level_rank[body_levels[label]]:
                                self.outward_move = True
                                # Since outward movement is simply adding exit (break) to
                                # the goto-stmt place, we have to create <exit> statement,
                                # then append it to the stmts_follo_goto
                                statement = ET.SubElement(parent, "statement")
                                statement.attrib['generated-exit-stmt'] = "true"
                                exit = ET.SubElement(statement, "exit")
                                stmts_follow_goto.append(statement)
                                # We need to remove it from the parent as it was just
                                # a place holder before append to the list
                                parent.remove(statement)

                                if label_body_level != "loop":
                                    self.goto_under_loop = False
                                else:
                                    self.goto_under_loop = True

    def reconstruct_next_goto(self, next_goto, reconstructed_goto_elem, parent):
        """This function reconstruct a goto statement that appears
        after the currently handling goto case. The default case
        is that the next goto is a backward goto case, which
        requires reconstruction by reconstruct_goto_before function.
        Thus, this function prepares the ingredient for it.

        Args:
            next_goto (list): Holds statement and goto-stmt elements.
            reconstructed_goto_elem (list): Holds reconstructed if statements
            that was generated after eliminating the goto.
            header (list): A header tht holds conditional header.

        Return:
            None.
        """
        statement = ET.SubElement(
            parent, next_goto[0]['statement'].tag,
            next_goto[0]['statement'].attrib
        )
        goto_stmt = ET.SubElement(
            statement, next_goto[0]['goto-stmt'].tag,
            next_goto[0]['goto-stmt'].attrib
        )
        if (
            reconstructed_goto_elem
            and reconstructed_goto_elem[0].attrib['label']
                == goto_stmt.attrib['target_label']
        ):
            for stmt in reconstructed_goto_elem:
                self.statements_to_reconstruct_before['stmts-follow-label'].append(stmt)
            self.statements_to_reconstruct_before['stmts-follow-label'].append(statement)
            if self.statements_to_reconstruct_before['count-gotos'] < 1:
                self.statements_to_reconstruct_before['count-gotos'] = 1
            self.reconstruct_before_case_now = True
            self.reconstruction_for_before_done = False

    def check_conditional_goto(self, header, stmts_follow_goto):
        """This function checks whether the goto is conditional
        or unconditional. If it's conditional, it extracts
        conditional operation (header).

        Args:
            header (list): A header tht holds conditional header.
            stmts_follow_goto (list): It holds all the statements
            that appeared after the goto statement in the original
            code.

        Returns:
            None.
        """
        # Check for the status whether current <goto-stmt> is
        # conditional. If yes, only extract the header (condition)
        # and remove the if statement AST from the tree. 
        uniq_code = None
        for stmt in stmts_follow_goto:
            if (
                    stmt.tag == "statement"
                    and "goto-stmt" in stmt.attrib
                    and "conditional-goto-stmt" in stmt.attrib
            ):
                uniq_code = stmt.attrib['code']
                self.conditional_goto = True

                if uniq_code in self.conditional_op:
                    header[0] = self.conditional_op[uniq_code]

    def reconstruct_stmts_follow_label_after_case(
                self, stmts_follow_label, next_goto,
                reconstructed_goto_elem, header, traverse,
                parent, index
    ):
        """This function generates a new statements to
        nest statements that follow label based on
        condition or non-condition status to eliminate
        goto.

        Args:
            next_goto (list): A list to hold next goto-stmt that may exist
            within the boundary of current goto.
            reconstructed_goto_elem (list): A list that will hold
            reconstructed if statements.
            header (list): A header tht holds conditional header.
            parent (:obj: 'ET'): A parent ET object that current
            element will be nested under.
            stmts_follow_label (list): A list that holds statements
            follow label statement for currently handling goto.
            traverse (int): A current traverse counter.
            index (int): An index of goto.

        Return:
            None.
        """

        for stmt in stmts_follow_label:
            if len(stmt) > 0:
                # A case where another goto-stmt appears after the current label
                if "goto-stmt" in stmt.attrib:
                    goto_stmt = {}
                    goto_stmt['statement'] = stmt
                    for child in stmt:
                        if child.attrib['target_label'] in self.goto_target_lbl_before:
                            goto_stmt['statement'].attrib['goto-move'] = "true"
                        if (
                            child.attrib['target_label'] not in self.goto_target_lbl_after
                            and "goto-move" in goto_stmt['statement']
                        ):
                            del goto_stmt['statement'].attrib['goto-move']
                        goto_stmt['goto-stmt'] = child
                    next_goto.append(goto_stmt)
                else:
                    # A case where both goto and label are under the same level
                    if not self.outward_move and not self.inward_move:
                        reconstructed_goto_elem.append(stmt)
                        if not self.encapsulate_under_do_while:
                            statement = ET.SubElement(parent, stmt.tag, stmt.attrib)
                            for child in stmt:
                                cur_elem = ET.SubElement(statement, child.tag, child.attrib)
                                if len(child) > 0:
                                    self.parseXMLTree(child, cur_elem, statement, parent, traverse)
                    # A case where outward movement goto handling is need
                    elif self.outward_move:
                        label_body_level = self.body_elem_holder[stmt.attrib['body-level']]
                        # If goto is a conditional case, but under else then
                        # there is no header operation. Thus, we simply declare new boolean
                        # variable like a non-conditional goto, then use that variable to
                        # construct new if statement and nest statement under label.
                        if self.goto_under_else:
                            number_of_gotos = int(self.statements_to_reconstruct_after['count-gotos'])
                            declared_goto_flag_num = []
                            self.generate_declaration_element(
                                label_body_level, "goto_flag", number_of_gotos,
                                declared_goto_flag_num, traverse
                            )
                            self.generate_if_element(
                                None, label_body_level, stmts_follow_label, next_goto, False, None,
                                f"goto_flag_{index + 1}", None, None, traverse,
                                reconstructed_goto_elem
                            )
                        else:
                            self.generate_if_element(
                                    header[0], label_body_level, stmts_follow_label, next_goto, True, None,
                                    None, None, None, traverse, reconstructed_goto_elem
                            )
                    # A case where inward movement goto handling is need
                    else:
                        pass


    def restruct_declaration(self, elem_declaration, parent):
        """This function is to restructure declaration to have an uniform
        xml structure."""
        
        declaration = ET.SubElement(
            parent, elem_declaration.tag, elem_declaration.attrib
        )

        for child in elem_declaration:
            subelem = ET.SubElement(
                declaration, child.tag, child.attrib
            )
            self.generate_element(child, subelem)
            if child.tag == "type":
                dimensions = ET.SubElement(
                    declaration,
                    self.dimensions_holder.tag,
                    self.dimensions_holder.attrib
                )
                self.handle_tag_dimensions(self.dimensions_holder, dimensions, parent, parent, 1)

    def generate_element(self, current_elem, parent_elem):
        """This function is to traverse the existing xml and generate
        a new copy to the given parent element."""

        for child in current_elem:
            if len(child) > 0 or child.text:
                elem = ET.SubElement(
                    parent_elem, child.tag, child.attrib
                )
                self.generate_element(child, elem)
            else:
                subelem = ET.SubElement(
                        parent_elem, child.tag,  child.attrib
                )
                if subelem.tag == "variable":
                    subelem.attrib['is_array'] = "true"

    #################################################################
    #                                                               #
    #                       ELEMENT GENERATORS                      #
    #                                                               #
    #################################################################

    def generate_declaration_element(
            self, parent, default_name, number_of_gotos,
            declared_flag_num, traverse
    ):
        """A flag declaration and assignment xml generation.
        This will generate N number of label_flag_i or goto_i,
        where N is the number of gotos in the Fortran code
        and i is the number assigned to the flag

        Args:
            parent (:obj: 'ET'): Parent element object.
            default_name (str): A default name given for
            new variable.
            number_of_gotos (int): A number of gotos. Amount
            of variables will be generated based on this number.
            declared_flag_num (list): A list to hold the number
            of delcared varaibles (flags).
            traverse (int): A current traverse counter.

        Return:
            None.
        """

        # Declaration
        specification_attribs = {
            "declaration": "1",
            "implicit": "1",
            "imports": "0",
            "uses": "0"
        }
        specification_elem = ET.SubElement(
            parent, "specification", specification_attribs
        )
        declaration_elem = ET.SubElement(
            specification_elem, "declaration", {"type": "variable"}
        )
        type_attribs = {
            "hasKind": "false",
            "hasLength": "false",
            "is_derived_type": "False",
            "keyword2": "none",
            "name": "logical",
        }
        type_elem = ET.SubElement(
            declaration_elem, "type", type_attribs
        )
        variables_elem = ET.SubElement(
            declaration_elem,
            "variables",
            {"count": str(number_of_gotos)}
        )
        variable_attribs = {
            "hasArraySpec": "false",
            "hasCharLength": "false",
            "hasCoarraySpec": "false",
            "hasInitialValue": "false",
            "hasInitialization": "false",
            "is_array": "false",
        }
        for flag in range(number_of_gotos):
            flag_num = flag + 1
            if default_name == "label_flag":
                if flag_num in self.declared_label_flags:
                    flag_num = self.declared_label_flags[-1] + 1
            if default_name == "goto_flag":
                if flag_num in self.declared_goto_flags:
                    flag_num = self.declared_goto_flags[-1] + 1
            self.declared_label_flags.append(flag_num)
            declared_flag_num.append(flag_num)
            variable_attribs['id'] = f"{default_name}_{flag_num}"
            variable_attribs['name'] = f"{default_name}_{flag_num}"
            variable_elem = ET.SubElement(
                variables_elem, "variable", variable_attribs
            )

        # Assignment
        for flag in range(number_of_gotos):
            flag_num = declared_flag_num[flag]
            declared_flag_num.append(flag_num)
            statement_elem = ET.SubElement(parent, "statement")
            self.generate_assignment_element(
                statement_elem, f"{default_name}_{flag_num}", None, "literal",
                "true", traverse
            )

    def generate_assignment_element(
            self, parent, name_id, condition, value_type, value, traverse
    ):
        """This is a function for generating new assignment element xml
        for goto reconstruction.

        Args:
            parent (:obj: 'ET'): Parent element object.
            name_id (str): Name of a target variable.
            value_type (str): Type of value that will be assigned.
            traverse (int): A current traverse counter.

        Returns:
            None.
        """
        assignment_elem = ET.SubElement(parent, "assignment")
        target_elem = ET.SubElement(assignment_elem, "target")

        self.generate_name_element(
            target_elem, "false", name_id, "false", "1", "variable"
        )

        value_elem = ET.SubElement(assignment_elem, "value")
        # Unconditional goto has default values of literal as below
        if value_type == "literal":
            assert (
                    condition is None
            ), "Literal type assignment must not hold condition element."
            literal_elem = ET.SubElement(
                value_elem, "literal", {"type": "bool", "value": value}
            )
        # Conditional goto has dynamic values of operation
        else:
            assert (
                    condition is not None
            ), "Conditional <goto-stmt> assignment must be passed with operation."
            unique_code = parent.attrib['code']
            for stmt in condition[unique_code]:
                if stmt.tag == "operation":
                    condition_op = stmt
            operation_elem = ET.SubElement(
                value_elem, condition_op.tag, condition_op.attrib
            )
            self.parseXMLTree(
                condition_op, operation_elem,
                value_elem, assignment_elem,
                traverse
            )

    def generate_operation_element(self, parent, op_type, operator, name):
        """This is a function for generating new operation element and
        its nested subelements with the passes arguments.

        Currently, it generates only a unary operation syntax only.
        It may require update in the future.

        Args:
            parent (:obj: 'ET'): Parent element object.
            op_type (str): Operation type.
            operator (str): Operator.
            name (str): Name of a variable for new element.

        Returns:
            None.
        """
        operation_elem = ET.SubElement(parent, "operation", {"type": op_type})
        operator_elem = ET.SubElement(operation_elem, "operator",
                                      {"operator": operator})
        operand_elem = ET.SubElement(operation_elem, "operand")

        self.generate_name_element(operand_elem, "false", name, "false", "1",
                                   "ambiguous")

    def generate_name_element(self, parent, hasSubscripts, name_id, is_array,
                              numPartRef, name_type):
        """This is a function for generating new name element based on
        the provided arguments.

        Args:
            parent (:obj: 'ET'): Parent element object.
            hasSubscripts (str): "true" or "false" status in string.
            name_id (str): Name of a variable.
            numPartRef (str): Number of references.
            type (str): Type of a variable.

        Returns:
            None.
        """
        name_attribs = {
            "hasSubscripts": hasSubscripts,
            "id": name_id,
            "is_array": is_array,
            "numPartRef": numPartRef,
            "type": name_type,
        }
        name_elem = ET.SubElement(parent, "name", name_attribs)

    def generate_if_element(
            self, header, parent, stored_stmts, next_goto,
            need_operation, op_type, lhs, rhs, operator,
            traverse, reconstructed_goto_elem
    ):
        """This is a function generating new if element.
        Since header can hold unary, multiary, or name, some arguments
        may be passed with None. Check them to generate an appropriate XML.

        Args:
            header (:obj: 'ET'): Header element from if.
            parent (:obj: 'ET'): Parent element object.
            stored_stmts (list): List of statements.
            next_goto (list): Another gotos appear while
            handling current goto stmt.
            need_operation (bool): Boolean to state whether
            new if needs operation header.
            op_type (str): Operation type.
            lhs (str): Left hand side variabel name.
            rhs (str): Right hand side variabel name.
            operator (str): Operator.
            traverse (int): Current traverse counter.
            reconstructed_goto_elem (list): A list to
            hold reconstructed AST after goto elimination.

        Returns:
            None.
        """
        goto_nest_if_elem = ET.SubElement(parent, "if", {"parent":parent.attrib['parent']})

        header_elem = ET.SubElement(goto_nest_if_elem, "header")

        if need_operation:
            if header is None:
                self.generate_operation_element(header_elem, op_type, operator,
                                                lhs)
            else:
                for stmt in header:
                    operation_elem = ET.SubElement(header_elem, stmt.tag,
                                                   stmt.attrib)
                    self.parseXMLTree(stmt, operation_elem, header_elem,
                                      goto_nest_if_elem, traverse)
        else:
            self.generate_name_element(header_elem, "false", lhs, "false", "1",
                                       "variable")

        # Generate AST for statements that will be nested under if (!cond) or (cond)
        label = None
        statement_num = 0
        label_before_within_scope = False
        body_elem = ET.SubElement(goto_nest_if_elem, "body")
        for stmt in stored_stmts:
            if len(stmt) > 0:
                if "skip-collect" in stmt.attrib:
                    parent_scope = stmt.attrib['parent-goto']
                    if parent_scope in self.goto_label_with_case:
                        if self.goto_label_with_case[parent_scope] == "before":
                            goto_nest_if_elem.attrib['label'] = parent_scope
                            self.encapsulate_under_do_while = True
                else:
                    if (
                        "next-goto" not in stmt.attrib
                        or (
                            "lbl" in stmt.attrib
                            and stmt.attrib['lbl'] == self.current_label
                        )
                    ):
                        if "goto-move" in stmt.attrib and not label_before_within_scope:
                            if "target-label-statement" in stmt.attrib:
                                del stmt.attrib['goto-move']
                                del stmt.attrib['target-label-statement']
                                label_before_within_scope = True
                                # If label for label-before case is the first statement,
                                # we want to mark this entire if-statement because it
                                # represents that it needs to be encapsulated with do-while
                                if statement_num == 0:
                                    self.encapsulate_under_do_while = True
                                # Reinitialize counter to 0 to count the number of gotos
                                # only within the current scope
                                self.statements_to_reconstruct_before['count-gotos'] = 0
                                self.statements_to_reconstruct_before['stmts-follow-label'] = []
                                self.current_label = stmt.attrib['label']
                        if label_before_within_scope:
                            self.statements_to_reconstruct_before[
                                'stmts-follow-label'].append(stmt)
                            for child in stmt:
                                if child.tag == "label":
                                    label = child.attrib['lbl']
                                    if self.current_label == label:
                                        if self.encapsulate_under_do_while:
                                            goto_nest_if_elem.attrib[
                                                'label'] = label
                                if child.tag == "goto-stmt":
                                    # If current goto-stmt label is equal to the scope label,
                                    # it means that end-of-scope is met and ready to reconstruct
                                    if self.current_label == child.attrib['target_label']:
                                        self.statements_to_reconstruct_before['count-gotos'] += 1
                                        # Since we are going to handle the first label-before
                                        # case, remove the label lbl from the list
                                        del self.goto_target_lbl_before[0]
                                        label_before_within_scope = False
                                        self.current_label = None

                                        reconstruct_target = self.statements_to_reconstruct_before
                                        self.reconstruct_goto_before_label(
                                            body_elem, traverse,
                                            reconstruct_target)

                                    # Else, a new goto-stmt was found that is nested current label_before
                                    # case scope, so we need to update the parent for it
                                    else:
                                        stmt.attrib['parent-goto'] = self.current_label
                        else:
                            cur_elem = ET.SubElement(body_elem, stmt.tag,
                                                     stmt.attrib)
                            if "goto-remove" in cur_elem.attrib:
                                del cur_elem.attrib['goto-remove']
                            for child in stmt:
                                child_elem = ET.SubElement(cur_elem, child.tag,
                                                           child.attrib)
                                if len(child) > 0:
                                    self.parseXMLTree(child, child_elem,
                                                      cur_elem, parent,
                                                      traverse)
                    else:
                        if need_operation:
                            goto_stmt = {}
                            goto_stmt['statement'] = stmt
                            for child in stmt:
                                assert (
                                        child.tag == "goto-stmt"
                                ), f"Must only store <goto-stmt> in next_goto['goto-stmt']. Current: <{child.tag}>."
                                if child.attrib[
                                    'target_label'] in self.goto_target_lbl_before:
                                    goto_stmt['statement'].attrib[
                                        'goto-move'] = "true"
                                if (
                                        child.attrib[
                                            'target_label'] not in self.goto_target_lbl_after
                                        and "goto-move" in goto_stmt['statement']
                                ):
                                    del goto_stmt['statement'].attrib['goto-move']
                                goto_stmt['goto-stmt'] = child
                            next_goto.append(goto_stmt)
                    statement_num += 1

        if (
            self.encapsulate_under_do_while
            and (
                    (
                        goto_nest_if_elem.attrib['parent'] != "program"
                        and self.outward_move
                    )
                    or (
                        goto_nest_if_elem.attrib['parent'] == "program"
                        and not self.outward_move
                    )
            )
        ):
            goto_nest_if_elem.attrib['goto-move'] = "true"
            reconstructed_goto_elem.append(goto_nest_if_elem)
            parent.remove(goto_nest_if_elem)

    #################################################################
    #                                                               #
    #                       MISCELLANEOUS                           #
    #                                                               #
    #################################################################

    def clean_derived_type_ref(self, current):
        """This function will clean up the derived type referencing syntax,
        which is stored in a form of "id='x'%y" in the id attribute.
        Once the id gets cleaned, it will call the
        reconstruc_derived_type_ref function to reconstruct and replace the
        messy version of id with the cleaned version.

        Args:
            current (:obj: 'ET'): Current element object.

        Returns:
            None.
        """
        current_id = current.attrib[
            "id"
        ]  # 1. Get the original form of derived type id, which is in a form of,
        # for example, id="x"%y in the original XML.
        self.derived_type_var_holder_list.append(
            self.clean_id(current_id)
        )  # 2. Extract the first variable name, for example, x in this case.
        percent_sign = current_id.find(
            "%"
        )  # 3. Get the location of the '%' sign.
        self.derived_type_var_holder_list.append(
            current_id[percent_sign + 1: len(current_id)]
        )  # 4. Get the field variable. y in this example.
        self.reconstruct_derived_type_ref(current)

    def clean_id(self, unrefined_id):
        """This function refines id (or value) with quotation
        marks included by removing them and returns only
        the variable name. For example, from "OUTPUT"
        to OUTPUT and "x" to x.
        
        Thus, the id name will be modified as below:
            Unrefined id - id = ""OUTPUT""
            Refined id - id = "OUTPUT"

        Args:
            unrefined_id (str): Id of name element that holds
            unnecessary strings.

        Returns:
            None
        """
        return re.findall(r"\"([^\']+)\"", unrefined_id)[0]

    def clean_attrib(self, current):
        """The original XML elements holds 'eos' and
        'rule' attributes that are not necessary
        and being used. Thus, this function will 
        remove them in the rectified version of
        XML.

        Args:
            current (:obj: 'ET'): Current element object.

        Returns:
            None.
        """
        if "eos" in current.attrib:
            current.attrib.pop("eos")
        if "rule" in current.attrib:
            current.attrib.pop("rule")

    def boundary_identifier(self):
        """This function will be called to identify the boundary
        for each goto-and-label. The definition of scope here is
        that whether one goto-label is nested under another goto-label.
        
        For example:
            <label with lbl = 111>
            ____<goto-stmt with lbl = 222>
            ____<label with lbl = 222>
            <goto-stmt with lbl = 111>

        In this case, "goto-label with lbl = 222" is within
        the scope of "lbl = 111"
        Thus, the elements will be assigned with "parent-goto" attribute with 111.

        Args:
            None.

        Returns:
            None.
        """
        boundary = {}
        lbl_counter = {}
        goto_label_in_order = []
        goto_and_labels = self.encountered_goto_label
        for lbl in goto_and_labels:
            if lbl not in lbl_counter:
                lbl_counter[lbl] = 1
            else:
                lbl_counter[lbl] += 1
            # Identify each label's parent label (scope)
            if not goto_label_in_order:
                goto_label_in_order.append(lbl)
            else:
                if lbl not in goto_label_in_order:
                    parent = goto_label_in_order[-1]
                    boundary[lbl] = parent
                    goto_label_in_order.append(lbl)

        # Since the relationship betwen label:goto-stmt is 1:M,
        # find the label that has multiple goto-stmts.
        # Because that extra <goto-stmt> creates extra scope to
        # encapsulate other 'label-goto' or 'goto-label'.
        for lbl in goto_label_in_order:
            if lbl not in boundary:
                for label, counter in lbl_counter.items():
                    if counter > 1 and counter % 2 > 0:
                        boundary[lbl] = label

        # This will check for the handled goto cases.
        # If any unhandled case encountered, then it will
        # assert and give out an error. Else, return nothing
        self.case_availability (boundary)

        boundary_for_label = boundary.copy()
        self.parent_goto_assigner (
                boundary, boundary_for_label,
                self.statements_to_reconstruct_before['stmts-follow-label']
        )
        self.parent_goto_assigner (
                boundary, boundary_for_label,
                self.statements_to_reconstruct_after['stmts-follow-goto']
        )
        self.parent_goto_assigner (
                boundary, boundary_for_label,
                self.statements_to_reconstruct_after['stmts-follow-label']
        )

    def update_arguments(self, current):
        """This function handles function definition's
        arguments with array status based on the information
        that was observed during the function call

        Args:
            current (:obj: 'ET'): Current node (either call or value)

        Returns:
            None.
        """

        fname = current.attrib['fname']
        if fname in self.arguments_list:
            callee_arguments = self.arguments_list[fname]
            for arg in callee_arguments:
                # self.caller_arr_arguments holds any element
                # only when arrays are being passed to functions
                # as arguments. Thus, we first need to check if
                # callee function name exists in the list
                if (
                    fname in self.caller_arr_arguments
                    and arg.attrib['name'] in self.caller_arr_arguments[fname]
                ):
                    arg.attrib['is_array'] = "true"
                else:
                    arg.attrib['is_array'] = "false"
        # re-initialize back to initial values
        self.call_function = False

    #################################################################
    #                                                               #
    #               GOTO ELIMINATION HELPER FUNCTIONS               #
    #                                                               #
    #################################################################

    def case_availability(self, boundary):
        """This function checks for the goto cases in the code based
        on the boundary. If any unhandled case encountered, then it
        will assert and halt the program.

        Args:
            boundary (dict): A dictonary of goto label
            and boundary label.

        Returns:
            None.
        """

        # Case check for more than double nested goto case
        nested_gotos = {}
        root_boundary = None
        current_boundary = None

        for goto, boundary in boundary.items():
            if current_boundary is None:
                current_boundary = goto
                root_boundary = goto
                nested_gotos[root_boundary] = 1
            else:
                if boundary == current_boundary:
                    nested_gotos[root_boundary] += 1
                    assert (
                            nested_gotos[root_boundary] <= 2
                    ), f"Do do not handle > 2 nested goto case at this moment."
                else:
                    root_boundary = goto
                    nested_gotos[root_boundary] = 1
                current_boundary = goto

        # All cases are currently handled
        return

    def parent_goto_assigner(self, boundary, boundary_for_label,
                             statements_to_reconstruct
    ):
        """This function actually assigns boundary(s) to each goto
        and label statements.

        Args:
            boundary (list): A list of boundaries.
            boundary_for_label (dict): A dictionary of
            label as a key and its parent boundary label.
            statements_to_reconstruct (list): A list of
            statements that require reconstruction.

        Returns:
            None.
        """
        for stmt in statements_to_reconstruct:
            if "goto-stmt" in stmt.attrib:
                target_lbl = stmt.attrib['lbl']
                if target_lbl in boundary:
                    stmt.attrib['parent-goto'] = boundary[target_lbl]
                    del boundary[target_lbl]
                else:
                    stmt.attrib['parent-goto'] = "none"

            if "target-label-statement" in stmt.attrib:
                label = stmt.attrib['label']
                if label in boundary_for_label:
                    stmt.attrib['parent-goto'] = boundary_for_label[label]
                    del boundary_for_label[label]
                else:
                    stmt.attrib['parent-goto'] = "none"

    def remove_dup_stmt(self, stmts_follow_label):
        """This removes the statement that's a child statement of
        if body being seprately re-added to the list.

        Args:
           stmts_follow_label (:obj: 'ET'): A list that holds
           statements appeard under the label-statement for
           reconstruction.

        Returns:
            None.
        """
        prev_stmt = None
        for stmt in stmts_follow_label:
            if prev_stmt is not None:
                # This statement always appears right before
                # the if-statement, so check this condition
                # and remove it from the list.
                if (
                    stmt.tag == "if"
                    and (prev_stmt.tag == "statement"
                         and prev_stmt.attrib['body-level'] == "if")
                ):
                    stmts_follow_label.remove(prev_stmt)
            prev_stmt = stmt

    def boundary_identifier_for_backward_goto(
            self, stmts_follow_label, goto_index_holder,
            number_of_gotos, target_label_lbl
    ):
        """This function identifies the boundary from label to goto.
        Remove any statements that are not within the boundary.
        Then, store those removed statements seprately for later
        restoration.

        Args:
            stmts_follow_label (list): A list holding the
            statements that appear after the label-statement
            for reconstruction.
            goto_index_holder (list): A list of index of goto
            in the stmts_follow_label.
            number_of_gotos (int): Number of gotos in the
            stmts_follow_label.
            target_label_lbl (list): A list that should
            only hold one value of label-stmt's label value.

        Returns:
            (list): A list of statements that requires
            restoration after loop generation.
        """
        index = 0
        goto_counter = 0
        for stmt in stmts_follow_label:
            if (
                    index == 0
                    and "label" in stmt.attrib
            ):
                target_label_lbl[0] = stmt.attrib['label']
            for child in stmt:
                if (
                        child.tag == "goto-stmt"
                        and child.attrib['target_label'] == target_label_lbl[0]
                ):
                    goto_counter += 1
                    goto_index_holder.append(index)
            if goto_counter == number_of_gotos:
                break
            index += 1

        statements_to_recover = stmts_follow_label[index+1:len(stmts_follow_label)]
        for stmt in statements_to_recover:
            if (
                    stmt.tag == "if"
                    and "conditional-goto-stmt-lbl" in stmt.attrib
            ):
                statements_to_recover.remove(stmt)
        del stmts_follow_label[index + 1:len(stmts_follow_label)]

        return statements_to_recover

    def multiple_goto_identifier (
            self, goto_index_holder,
            multiple_goto_stmts, stmts_follow_label
    ):
        """This function identifies any additional goto
        statements may appear within the boundary of
        currently handling backward goto case.

        Args:
            stmts_follow_label (list): A list holding the
            statements that appear after the label-statement
            for reconstruction.
            goto_index_holder (list): A list of index of goto
            in the stmts_follow_label.
            multiple_goto_stmts (list): A list that will hold
            additional gotos within the boundary of current
            goto.

        Returns:
            None.
        """
        for i in range(len(goto_index_holder)):
            if i == 0:
                multiple_goto_stmts.append(
                    stmts_follow_label[0:goto_index_holder[i] + 1]
                )
            else:
                if i + 1 < len(goto_index_holder):
                    multiple_goto_stmts.append(
                        stmts_follow_label[
                        goto_index_holder[i - 1] + 1:goto_index_holder[
                                                         i + 1] + 1]
                    )
                else:
                    multiple_goto_stmts.append(
                        stmts_follow_label[
                        goto_index_holder[i - 1] + 1:goto_index_holder[-1] + 1]
                    )

    def nested_forward_goto_identifier(self, multiple_goto_stmts):
        """This function identifies any existing forward
        goto case nested under the backward goto case.

        Args:
            multiple_goto_stmts (list): A list that will hold
            additional gotos within the boundary of current
            goto.
            index_boundary (list): A list that will hold
            the indices of label of <label> and <goto-stmt>.

        Returns:
            (bool): A boolean status indicating whether the
            nested forward goto exists within the boundary.
        """
        labels = []
        index_boundary = []
        nested_gotos_exist = False
        for goto in multiple_goto_stmts:
            index = 0
            main_loop_lbl = goto[0].attrib['label']
            label_after_lbl = None
            for stmt in goto:
                if "label" in stmt.attrib:
                    labels.append(stmt.attrib["label"])
                    if stmt.attrib["label"] == label_after_lbl:
                        index_boundary.append(index)
                if "goto-stmt" in stmt.attrib:
                    if (
                            main_loop_lbl != stmt.attrib['lbl']
                            and stmt.attrib['lbl'] not in labels
                    ):
                        nested_gotos_exist = True
                        label_after_lbl = stmt.attrib['lbl']
                        index_boundary.append(index)
                index += 1

        return nested_gotos_exist

    def construct_goto_loop(
            self, parent, reconstruct_target, nested_gotos_exist,
            multiple_goto_stmts, number_of_gotos, declared_label_flag_num,
            traverse, target_label_lbl
    ):
        """This function constructs loop syntax tree for goto
        backward case.

        Args:
            parent (:obj: 'ET'): Parent element of loop.
            reconstruct_target (dict): A dictionary that
            will hold nested goto statement.
            nested_gotos_exist (bool): Boolean to indicating
            whether nested goto exists or not.
            multiple_goto_stmts (list): A list of goto and other
            statements.
            number_of_gotos (int): Number of gotos to reconstruct.
            declared_label_flag_num (list): List of flag numbers.
            traverse (int): Current traverse counter.
            target_label_lbl (list): A single value list that
            holds the label value of <label>.

        Returns:
            None.
        """
        cur_elem_parent = parent
        current_goto_num = 1
        end_of_current_goto_loop = False
        for i in range(number_of_gotos):
            loop_elem = ET.SubElement(cur_elem_parent, "loop",
                                      {"type": "do-while"})

            header_elem = ET.SubElement(loop_elem, "header")
            # The outermost flag == N and the innermost flag == 1
            flag_num = declared_label_flag_num[i]
            name = f"label_flag_{str(flag_num)}"
            name_attrib = {
                "hasSubscripts": "false",
                "id": name,
                "type": "ambiguous",
            }
            name_elem = ET.SubElement(header_elem, "name", name_attrib)
            flag_name = name
            body_elem = ET.SubElement(loop_elem, "body")
            # Keep a track of the parent and grandparent elements
            grand_parent_elem = cur_elem_parent
            cur_elem_parent = body_elem
            # Since reconstruction of multiple goto is done from outermost
            # to the inner, we are not constructing any subelements until
            # all encapsulating loops are created first
            if current_goto_num == number_of_gotos:
                for statements in multiple_goto_stmts:
                    index = 0
                    for stmt in statements:
                        if len(stmt) > 0:
                            if nested_gotos_exist:
                                self.nested_goto_handler(
                                        reconstruct_target, statements,
                                        body_elem, traverse
                                )
                                nested_gotos_exist = False
                            else:
                                elems = ET.SubElement(
                                    body_elem, stmt.tag, stmt.attrib
                                )
                                for child in stmt:
                                    if (
                                            child.tag == "goto-stmt"
                                            and target_label_lbl[0] ==
                                            child.attrib['target_label']
                                    ):
                                        # Conditional
                                        if "conditional-goto-stmt" in stmt.attrib:
                                            self.generate_assignment_element(
                                                elems, flag_name,
                                                self.conditional_op, None, None,
                                                traverse
                                            )
                                        # Unconditional
                                        else:
                                            self.generate_assignment_element(
                                                elems, flag_name, None,
                                                "literal", "true", traverse
                                            )
                                        end_of_current_goto_loop = True
                                    else:
                                        child_elem = ET.SubElement(
                                            elems, child.tag, child.attrib
                                        )
                                        if len(child) > 0:
                                            self.parseXMLTree(
                                                child, child_elem, elems,
                                                parent, traverse
                                            )
                        # If end_of_current_goto_loop is True,
                        # escape one loop out to continue
                        # construct statements
                        if end_of_current_goto_loop:
                            body_elem = grand_parent_elem
                            end_of_current_goto_loop = False
                            flag_name = f"label_flag_" \
                                f"{str(number_of_gotos + i - 1)}"
                    index += 1

            else:
                    current_goto_num += 1

    def nested_goto_handler(
                self, reconstruct_target, statements,
                body_elem, traverse
    ):
        """This function collects forward goto case
        related statements under the backward goto
        boundary. Then, it calls goto_after function
        to reconstruct goto.

        Args:
            reconstruct_target (list): A list that holds
            statements for reconstruction.
            statements (:obj: 'ET'): Statements for
            reconstructions.
            body_elem (:obj: 'ET'): Body element of
            the loop.
            traverse (int): Current traverse counter. 
        """
        reconstruct_target['stmts-follow-goto'] \
            = statements[index_scope[0]:index_scope[1]]
        reconstruct_target['stmts-follow-label'] \
            = statements[index_scope[1]]
        reconstruct_target['count-gotos'] \
            = 1

        self.reconstruct_goto_after_label(
            body_elem, traverse, reconstruct_target
        )

        self.statements_to_reconstruct_after[
            'stmts-follow-goto'] = []

    def statement_recovery (self, statements_to_recover, parent, traverse):
        """This function is for recovering any existing statements
        that follow reconstructed loop.

        Args:
            statements_to_recover (list): A list of statements.
            parent (:obj: 'ET'): A prent element.
            traverse (int): Current traverse counter.
        """
        for recover_stmt in statements_to_recover:
            statement = ET.SubElement(
                parent, recover_stmt.tag, recover_stmt.attrib
            )
            for child in recover_stmt:
                child_elem = ET.SubElement(
                    statement, child.tag, child.attrib
                )
                if len(child) > 0:
                    self.parseXMLTree(
                        child, child_elem, statement, parent, traverse
                        )


#################################################################
#                                                               #
#                     NON-CLASS FUNCTIONS                       #
#                                                               #
#################################################################

def is_empty(elem):
    """This function is just a helper function for
    check whether the passed elements (i.e. list)
    is empty or not

    Args:
        elem (:obj:): Any structured data object (i.e. list).

    Returns:
        bool: True if element is empty or false if not.
    """
    if not elem:
        return True
    else:
        return False


def indent(elem, level=0):
    """This function indents each level of XML.

    Source:
        https://stackoverflow.com/questions/3095434/inserting-newlines
        -in-xml-file-generated-via-xml-etree-elementstree-in-python

    Args:
        elem (:obj: 'ET'): An XML root.
        level (int): A root level in integer.

    Returns:
        None.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def buildNewASTfromXMLString(
        xmlString: str,
        original_fortran_file: str,
        module_log_file_path: str
) -> ET.Element:
    """This function process OFP generated XML and generates
    a rectified version by recursively calling the appropriate
    functions.

    Args:
        xmlString (str): XML in string type.

    Returns:
        ET object: A reconstructed element object.
    """
    XMLCreator = RectifyOFPXML()
    # We need the absolute path of Fortran file to lookup in the modLogFile.json
    XMLCreator.original_fortran_file_abs_path = \
        os.path.abspath(original_fortran_file)
    XMLCreator.module_log_file_path = module_log_file_path
    traverse = 1

    ofpAST = ET.XML(xmlString)
    # A root of the new AST
    newRoot = ET.Element(ofpAST.tag, ofpAST.attrib)
    # First add the root to the new AST list
    for child in ofpAST:
        # Handle only non-empty elements
        if child.text:
            cur_elem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, cur_elem, newRoot, newRoot, traverse)

    # Indent and structure the tree properly
    tree = ET.ElementTree(newRoot)
    indent(newRoot)

    # Checks if the rectified AST requires goto elimination,
    # if it does, it does a 2nd traverse to eliminate and
    # reconstruct the AST once more
    while (XMLCreator.need_goto_elimination):
        oldRoot = newRoot
        traverse += 1

        XMLCreator.boundary_identifier()

        newRoot = ET.Element(oldRoot.tag, oldRoot.attrib)
        for child in oldRoot:
            if child.text:
                cur_elem = ET.SubElement(newRoot, child.tag, child.attrib)
                XMLCreator.parseXMLTree(child, cur_elem, newRoot, newRoot,
                                        traverse)
        tree = ET.ElementTree(newRoot)
        indent(newRoot)
        if not XMLCreator.continue_elimination:
            XMLCreator.need_goto_elimination = False

    return newRoot, XMLCreator.module_files_to_process


def parse_args():
    """This function parse the arguments passed to the script.
    It returns a tuple of (input ofp xml, output xml)
    file names.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        help="OFP generated XML file needs to be passed.",
    )

    parser.add_argument(
        "-g",
        "--gen",
        nargs="+",
        help="A rectified version of XML.",
    )

    args = parser.parse_args(sys.argv[1:])

    if (
            args.file is not None
            and args.gen is not None
    ):
        ofpFile = args.file[0]
        rectifiedFile = args.gen[0]
    else:
        assert (
            False
        ), f"[[ Missing either input or output file.\
             Input: {args.file}, Output: {args.gen} ]]"

    return (ofpFile, rectifiedFile)


def fileChecker(filename, mode):
    """This function checks for the validity (file existence and
    mode). If either the file does not exist or the mode is
    not valid, throws an IO exception and terminates the program

    Args:
        filename (str): A file name that reconstructed XMl
        will be written to.
        mode (str): Open more for a file.

    Returns:
        None.
    """
    try:
        with open(filename, mode) as f:
            pass
    except IOError:
        assert (
            False
        ), f"File {filename} does not exit or invalid mode {mode}."


if __name__ == "__main__":
    (ofpFile, rectifiedFile) = parse_args()

    # Since we pass the file name to the element
    # tree parser not opening it with open function,
    # we check for the validity before the file name
    # is actually passed to the parser
    fileChecker(ofpFile, "r")
    ofpXML = ET.parse(ofpFile)
    ofpXMLRoot = ofpXML.getroot()

    # Converts the XML tree into string
    ofpXMLStr = ET.tostring(ofpXMLRoot).decode()
    # Call buildNewASTfromXMLString to rectify the XML
    rectifiedXML = buildNewASTfromXMLString(ofpXMLStr)
    rectifiedTree = ET.ElementTree(rectifiedXML)

    # The write function is used with the generated
    # XML tree object not with the file object. Thus,
    # same as the ofpFile, we do a check for the validity
    # of a file before pass to the ET tree object's write
    # function
    fileChecker(rectifiedFile, "w")
    rectifiedTree.write(rectifiedFile)
