"""

The purpose of this program is to do all the clean up for translate.py.
This (rectify.py) program will receive OFP generated XML file as an input.
Then, it removes any unnecessary elements and refactor randomly structured
(nested) elementss into a correct structure. The output file will be
approximately 30%~40% lighter in terms of number of lines than the OFP XML.

Example:
    This script is executed by the autoTranslate script as one
    of the steps in converted a Fortran source file to Python
    file. For standalone execution:::

        $python rectify.py <ast_file>

ast_file: The XML represenatation of the AST of the Fortran file. This is
produced by the OpenFortranParser.

Author: Terrence J. Lim
Last Modified: 5/23/2019

"""

import sys
import re
import argparse
import xml.etree.ElementTree as ET

NEGATED_OP = {
                ".le." : ".ge.",
                ".ge." : ".le.",
                ".lt." : ".gt.",
                ".gt." : ".lt.",
                ".eq." : ".ne.",
                ".ne." : ".eq.",
                "<=" : ">=",
                ">=" : "<="
              }

class RectifyOFPXML:
    def __init__(self):
        self.is_derived_type = False
        self.is_derived_type_ref = False
        self.is_array = False
        self.is_format = False
        self.is_function_arg = False

        # True only if <goto-stmt> exists in the AST
        self.need_goto_elimination = False
        self.need_op_negation = False
        self.need_reconstruct = False
        self.reconstruct_after_case_now = False
        self.reconstruct_before_case_now = False
        self.label_before = False
        self.label_after = False
        self.reconstructing = False
        self.is_stop = False

        self.goto_target_lbl_after = []
        self.label_lbl = []
        self.statements_to_reconstruct_before = []
        self.statements_to_reconstruct_after = []

        self.cur_derived_type_name = ""
        self.current_scope = ""

        self.goto_body = ET.Element('')

        # Keep a track of both array and non-array variables 
        # in the dictionary of {'name' : 'scope'}
        self.declared_non_array_vars = {}
        self.declared_array_vars = {}

        self.derived_type_var_holder_list = []
        self.derived_type_refs = []
        self.subscripts_holder = []

        self.format_holder = ET.Element('')
        self.parent_type = ET.Element('')
        self.derived_type_ref = ET.Element('')
        self.current_body_scope = ET.Element('')


    #################################################################
    #                                                               #
    #                  TAG LISTS FOR EACH HANDLER                   #
    #                                                               #
    #################################################################

    file_child_tags = ["program", "subroutine", "module"]

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
    ]

    loop_child_tags = ["header", "body", "format"]

    declaration_child_tags = [
        "type",
        "dimensions",
        "variables",
        "format",
        "name",
        "type-declaration-stmt",
        "prefix-spec",
        "save-stmt",
        "access-spec",
        "attr-spec",
        "access-stmt",
        "access-id-list",
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
        "loop-control",
        "label",
    ]

    body_child_tags = [
        "specification",
        "statement",
        "loop",
        "if",
        "label",
        "stop",
        "do-term-action-stmt",
    ]

    #################################################################
    #                                                               #
    #                       HANDLER FUNCTONS                        #
    #                                                               #
    #################################################################

    def handle_tag_file(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the file elementss.

            In order to control new sub-element creation under the current element,
            if child.tag == "__tag_name__" has been added. If any new tag(s) that is not being handled currently,
            appears in the future, add child.tag == "__tag_name__" at the end of the last condition.
            This applies all other handler functions.

            <file>
                ...
            </file>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag in self.file_child_tags:
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_file: "{child.tag}" not handled')

    def handle_tag_program(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the program elementss.
            <program>
                ...
            </program>
        """
        self.current_scope = root.attrib["name"]
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "header" or child.tag == "body":
                if child.tag == "body":
                    self.current_body_scope = curElem
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                if (
                    child.tag != "end-program-stmt"
                    and child.tag != "main-program"
                ):
                    print(f'In handle_tag_program: "{child.tag}" not handled')

    def handle_tag_header(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the header elementss.
            <header>
                ...
            </header>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag in self.header_child_tags:
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_header: "{child.tag}" not handled')
            else:
                if child.tag == "subroutine-stmt":
                    current.attrib.update(child.attrib)
                elif child.tag in self.header_child_tags:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    if self.reconstructing:
                        self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_header: Empty elements  "{child.tag}" not handled'
                    )


    def handle_tag_body(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the body elementss.
            <body>
                ...
            </body>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag in self.body_child_tags:
                    self.parseXMLTree(child, curElem, current, traverse)
                    # Goto elimination (1st traverse)
                    if self.reconstruct_after_case_now:
                        self.reconstruct_goto_after_label()
                    elif self.reconstruct_before_case_now:
                        self.reconstruct_goto_before_label(current, traverse)
                    # Remove "goto-target" from the <body> (2nd traverse)
                    if "goto-target" in current.attrib:
                        assert (
                            traverse == 2
                        ), "Cleaning of temporarily added attribute must be done in the 2nd traverse"
                        del current.attrib["goto-target"]
                    # Remove statements that is marked to be removed (2nd traverse)
                    if (
                        "goto-remove" in child.attrib
                        and "target-label-statement" not in child.attrib
                        or "goto-move" in child.attrib
                    ):
                        assert (
                            traverse == 2
                        ), "Removal of old statements must be done in the 2nd traverse"
                        current.remove(curElem)

                    # Add "goto-target" to the <body>, so it can be reconstructed
                    # in the second traverse (1st traverse) curElem here is 
                    # <statement>, which needs to be removed
                    if "goto-stmt" in curElem.attrib:
                        current.attrib["goto-target"] = "true"
                else:
                    print (f'In handle_tag_body: "{child.tag}" not handled')
            else:
                if child.tag in self.body_child_tags and child.tag != "statement":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                elif child.tag != "statement":
                    print(
                        f'In handle_tag_body: Empty elements  "{child.tag}" not handled'
                    )

        if self.is_format:
            self.reconstruct_format()
            self.is_format = False

    def handle_tag_specification(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the specification elementss.
            <specification>
                ...
            </specification>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "declaration" or child.tag == "use":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_specification: "{child.tag}" not handled'
                    )
            else:
                if child.tag != "declaration":
                    print(
                        f'In handle_tag_specification: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_declaration(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the declaration elementss.
            <declaration>
                ...
            </declaration>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag in self.declaration_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder = child
                    else:
                        curElem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        if child.tag == "dimensions":
                            self.is_array = True
                        self.parseXMLTree(child, curElem, current, traverse)
                elif (
                    child.tag == "component-array-spec"
                    or child.tag == "literal"
                ):
                    self.derived_type_var_holder_list.append(child)
                else:
                    print(
                        f'In handle_tag_declaration: "{child.tag}" not handled'
                    )
            else:
                if child.tag in self.declaration_child_tags:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                elif (
                    child.tag == "component-decl"
                    or child.tag == "component-decl-list"
                ):
                    current.attrib["type"] = "derived-type"
                    self.derived_type_var_holder_list.append(child)
                elif child.tag == "component-array-spec":
                    self.derived_type_var_holder_list.append(child)
                else:
                    if (
                        child.tag != "attr-spec"
                        and child.tag != "access-id"
                        and child.tag not in self.derived_type_child_tags
                    ):
                        print(
                            f'self.In handle_tag_declaration: Empty elements "{child.tag}" not handled'
                        )
        if self.is_array == True:
            self.is_array = False

        # If is_derived_type is true, reconstruct the derived type declaration AST structure
        if self.is_derived_type:
            if self.derived_type_var_holder_list:
                # Modify or add 'name' attribute of the MAIN (or the outer
                # most) <type> elements with the name of derived type name

                self.parent_type.set("name", self.cur_derived_type_name)
                self.reconstruct_derived_type_declaration()
            self.is_derived_type = False

    def handle_tag_type(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the variables elementss.
            <type>
                ...
            </type>
        """
        for child in root:
            self.clean_attrib(child)
            if "keyword2" in child.attrib:
                if child.attrib["keyword2"] == "":
                    current.attrib["keyword2"] = "none"
                else:
                    current.attrib["keyword2"] = child.attrib["keyword2"]
            else:
                current.attrib["keyword2"] = "none"
            if child.tag == "type":
                """
                    Having a nested "type" indicates that this is a "derived type" declaration.
                    In other word, this is a case of
                    <type>
                        <type>
                            ...
                        </type>
                    </type>
                """
                self.is_derived_type = True
                curElem = ET.SubElement(current, child.tag, child.attrib)
                self.parent_type = current
                self.parseXMLTree(child, curElem, current, traverse)
            elif child.tag == "intrinsic-type-spec":
                if self.is_derived_type:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "derived-type-stmt" and self.is_derived_type:
                # Modify or add 'name' attribute of the <type> elements with the name of derived type name
                current.set("name", child.attrib["id"])
                # And, store the name of the derived type name for later setting the outer most <type> elements's name attribute
                self.cur_derived_type_name = child.attrib["id"]
            elif child.tag == "derived-type-spec":
                if not self.is_derived_type:
                    self.is_derived_type = True
                    current.set("name", child.attrib["typeName"])
                else:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "literal":
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "component-array-spec":
                self.derived_type_var_holder_list.append(child)
            elif (
                child.tag == "component-decl"
                or child.tag == "component-decl-list"
            ):
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "length":
                curElem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                if (
                    child.tag not in self.derived_type_child_tags
                    and child.tag != "char-selector"
                    and child.tag != "delcaration-type-spec"
                ):
                    print(f'In handle_tag_type: "{child.tag}" not handled')
        # This will mark whether this type declaration is for a derived type
        # declaration or not
        current.set("is_derived_type", str(self.is_derived_type))

    def handle_tag_variables(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the variables elementss.
            <variables>
                ...
            </variables>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                # Up to this point, all the child (nested or sub) elements were
                # <variable>
                curElem = ET.SubElement(current, child.tag, child.attrib)
                curElem.set("is_array", str(self.is_array).lower())
                self.parseXMLTree(child, curElem, current, traverse)
            elif child.tag == "variable":
                curElem = ET.SubElement(current, child.tag, child.attrib)

    def handle_tag_variable(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the variables elementss.
            <variable>
                ...
            </variable>
        """
        # Store all declared variables based on their array status
        if current.attrib["is_array"] == "true":
            self.declared_array_vars.update(
                {current.attrib["name"]: self.current_scope}
            )
        else:
            self.declared_non_array_vars.update(
                {current.attrib["name"]: self.current_scope}
            )

        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "initial-value":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_variable: "{child.tag}" not handled')
            else:
                if child.tag == "entity-decl":
                    current.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_variable: Empty elements "{child.tag}" not handled')

    def handle_tag_statement(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the statement elementss.
            <statement>
                ...
            </statement>
        """
        assert root.tag == "statement", f"The tag <statement> must be passed to handle_tag_statement. Currently, it's {root.tag}."
        # If label appears after <goto>, mark <statement> with goto-remove to remove it later (1st traverse)
        if self.label_after:
            current.attrib['goto-remove'] = "true"

        label_presented = False
        for child in root:
            self.clean_attrib(child)
            if child.tag in self.statement_child_tags:
                if child.tag == "stop":
                    self.is_stop = True
                    current.attrib["has_stop"] = "true"
                curElem = ET.SubElement(current, child.tag, child.attrib)

                if child.tag == "label":
                    label_presented = True
                    lbl = child.attrib['lbl']
                    self.label_lbl.append(lbl)
                    # If current <statment> holds <label>,
                    # mark "self.reconstruct_after_case_now" to True, which will
                    # immediately reconstruct the goto body when returned back to
                    # <body> from current statement
                    if traverse == 2 and "goto-remove" in current.attrib:
                        self.reconstruct_after_case_now = True
                    if traverse == 1:
                        current.attrib['goto-move'] = "true"
                        current.attrib['target-label-statement'] = "true"
                        # If lbl is not in the lbl_after tracker, it can be
                        # assumed that the <label> is present before the goto statment. 
                        # If this label has nothing to do with goto, then undo at below.
                        if lbl not in self.goto_target_lbl_after:
                            self.label_before = True
                # Since <format> is followed by <label>,
                # check the case and undo all operations done for goto.
                if traverse == 1 and child.tag == "format" and label_presented:
                    del self.label_lbl[-1]
                    del self.statements_to_reconstruct_before[-1]
                    del current.attrib['target-label-statement']
                    del current.attrib['goto-move']
                    label_presented = False
                    self.label_before = False
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            elif child.tag == "name":
                """
                    If a 'name' tag is the direct sub-elements of 'statement', it's an indication of
                    this statement is handling (usually assignment) derived type variables. Thus,
                    in order to make concurrent with other assignment syntax, remove the outside
                    name elements (but store it to the temporary holder) and reconstruct it before
                    the end of statement
                """
                assert is_empty(self.derived_type_var_holder_list)
                self.derived_type_var_holder_list.append(child.attrib['id'])
                self.parseXMLTree(child, current, current, traverse)
            elif child.tag == "goto-stmt":
                """
                    If goto-stmt was seen, we do not construct element for it. However, we collect
                    the information (attributes) that is associated to the existing OFP generated element
                """
                self.need_goto_elimination = True
                self.goto_in_scope = True
                target_lbl = child.attrib['target_label']
                # A case where label appears "before" goto
                if target_lbl in self.label_lbl:
                    self.reconstruct_before_case_now = True
                    parent.remove(current)
                    return
                # A case where label appears "after" goto
                else:
                    self.goto_target_lbl_after.append(target_lbl)
                    self.label_after = True
                """
                    This is a marking for 2nd traverse to remove the <statement> element that nest 
                    <goto-stmt> element. This will add "goto-remove" attribute to <statement> tag only if 
                    "goto-stmt" exists in the child.
                """
                current.attrib["goto-stmt"] = "true"
            else:
                print (f'In self.handle_tag_statement: "{child.tag}" not handled')


        # If label appears before <goto>, mark <statement> with goto-move to move it later (1st traverse)
        if traverse == 1 and self.label_before:
            current.attrib['goto-move'] = "true"
            if "has_stop" not in current.attrib:
                self.statements_to_reconstruct_before.append(current)

    def handle_tag_assignment(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elements between the assignment elementss.
            <assignment>
                ...
            </assignment>
        """
        assert (
            root.tag == "assignment"
        ), f"The tag <assignment> must be passed to handle_tag_assignment. Currently, it's {root.tag}."

        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "target" or child.tag == "value":
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                print (f'In self.handle_tag_assignment: "{child.tag}" not handled')
     
    def handle_tag_target(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the target elementss.
            <target>
                ...
            </target>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "name":
                self.parseXMLTree(child, curElem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(curElem, current)
            else:
                print(f'In self.handle_tag_target: "{child.tag}" not handled')

    def handle_tag_names(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elements between the names elements.
            <names>
                ...
            <names>
        """
        assert (
            root.tag == "names"
        ), f"The tag <names> must be passed to handle_tag_name. Currently, it's {root.tag}."

        for child in root:
            self.clean_attrib(child)
            if child.tag == "name":
                curElem = ET.SubElement(current, child.tag, child.attrib)
                # If the element holds subelements, call the XML tree parser with created new <name> element
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
                # Else, update the element's attribute with the default <name> element attributes
                else:
                    attributes = {
                        "hasSubscripts": "false",
                        "is_array": "false",
                        "numPartRef": "1",
                        "type": "ambiguous",
                    }
                    # Check if the variable is a function argument
                    if self.is_function_arg:
                        attributes["is_arg"] = "true"
                    else:
                        attributes["is_arg"] = "false"
                    curElem.attrib.update(attributes)
            else:
                print(f'In self.handle_tag_names: "{child.tag}" not handled')

    def handle_tag_name(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the name elements.
            <name>
                ...
            <name>

            There are three different types of names that the type attribute can hold:
                (1) variable  - Simple (single) variable or an array
                (2) procedure - Function (or procedure) call
                (3) ambiguous - None of the above two type
        """
        assert (
            root.tag == "name"
        ), f"The tag <name> must be passed to handle_tag_name. Currently, it's {root.tag}."
        # All variables have default "is_array" value "false"
        current.attrib["is_array"] = "false"

        # If 'id' attribute holds '%' symbol, it's an indication of derived type referencing
        # Thus, clean up the 'id' and reconstruct the <name> AST
        if "id" in current.attrib and "%" in current.attrib["id"]:
            self.is_derived_type_ref = True
            self.clean_derived_type_ref(current)

        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "subscripts" or child.tag == "assignment":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    if child.tag == "subscripts":
                        # Default
                        current.attrib["hasSubscripts"] = "true"
                        # Check whether the variable is an array AND the
                        # variable is for the current scope. This is important
                        # for derived type variable referencing
                        if (
                            current.attrib["id"] in self.declared_array_vars
                            and self.declared_array_vars[current.attrib["id"]]
                            == self.current_scope
                        ):
                            # Since the procedure "call" has a same AST syntax as an array, check its type and set the "is_array" value
                            assert (
                                current.attrib["type"] != "procedure"
                            ), "Trying to assign a procedure call to is_array true. This is an error"
                            current.attrib["is_array"] = "true"
                        elif (
                            current.attrib["id"]
                            in self.declared_non_array_vars
                            and self.declared_non_array_vars[
                                current.attrib["id"]
                            ]
                            == self.current_scope
                        ):
                            current.attrib["hasSubscripts"] = "false"
                    self.parseXMLTree(child, curElem, current, traverse)
                elif child.tag == "output":
                    assert is_empty(self.derived_type_var_holder_list)
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.derived_type_var_holder_list.append(root.attrib["id"])
                    self.parseXMLTree(child, curElem, current, traverse)
                elif child.tag == "name":
                    self.parseXMLTree(child, current, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_name: "{child.tag}" not handled'
                    )
            else:
                if child.tag == "generic_spec":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                elif child.tag == "data-ref":
                    current.attrib.update(child.attrib)
                elif child.tag != "designator":
                    print(
                        f'In self.handle_tag_name: Empty elements "{child.tag}" not handled'
                    )

        # If the name element is for handling derived type references, reconstruct it
        if self.derived_type_refs:
            self.reconstruct_derived_type_names(current)
            self.is_derived_type_ref = False
            self.need_reconstruct = True

    def handle_tag_value(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the value elementss.
            <value>
                ...
            </value>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if (
                child.tag == "literal"
                or child.tag == "operation"
                or child.tag == "name"
            ):
                self.parseXMLTree(child, curElem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(curElem, current)
            else:
                print(f'In self.handle_tag_value: "{child.tag}" not handled')

    def handle_tag_literal(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the literal elementss.
            <literal>
                ...
            </literal>
        """
        if '"' in current.attrib["value"]:
            current.attrib["value"] = self.clean_id(current.attrib["value"])
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "stop":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_literal: "{child.tag}" not handled'
                    )

    def handle_tag_dimensions(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the dimensions elementss.
            <dimensions>
                ...
            </dimensions>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "dimension":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_dimensions: "{child.tag}" not handled'
                    )

    def handle_tag_dimension(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the dimension elementss.
            <dimension>
                ...
            </dimension>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "range":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_dimension: "{child.tag}" not handled'
                    )

    def handle_tag_loop(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the do loop elementss.
            <loop>
                ...
            </loop>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag in self.loop_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder = child
                    else:
                        curElem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_loop: "{child.tag}" not handled'
                    )

    def handle_tag_index_variable_or_range(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the index_variable or range elementss.
            <index_variable>                        <range>
                ...                 or                  ...
            </index_variable>                       </range>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "lower-bound"
                    or child.tag == "upper-bound"
                    or child.tag == "step"
                ):
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_index_variable_or_range: "{child.tag}" not handled'
                    )

    def handle_tag_lower_bound(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the lower_bound elementss.
            <lower_bound>
                ...
            </lower_bound>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_lower_bound: "{child.tag}" not handled'
                    )

    def handle_tag_upper_bound(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the upper_bound elementss.
            <upper_bound>
                ...
            </upper_bound>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "literal"
                    or child.tag == "name"
                    or child.tag == "operation"
                ):
                    self.parseXMLTree(child, curElem, current, traverse)
                    if child.tag == "name" and self.need_reconstruct:
                        self.reconstruct_name_element(curElem, current)
                else:
                    print(
                        f'In self.handle_tag_upper_bound: "{child.tag}" not handled'
                    )

    def handle_tag_subscripts(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the subscripts elementss.
            <supscripts>
                ...
            </supscripts>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "subscript":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_subscripts: "{child.tag}" not handled'
                    )

    def handle_tag_subscript(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the subscript elementss.
            <supscript>
                ...
            </supscript>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "name"
                    or child.tag == "literal"
                    or child.tag == "operation"
                ):
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_subscript: "{child.tag}" not handled'
                    )

    def handle_tag_operation(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the operation elementss.
            <operation>
                ...
            </operation>
        """
        for child in root:
            self.clean_attrib(child)
            # A process of negating the operator during goto elimination
            if child.tag == "operator" and self.need_op_negation:
                child.attrib["operator"] = NEGATED_OP[child.attrib["operator"]]
                self.need_op_negation = False
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "operand":
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                if child.tag != "operator":
                    print(
                        f'In handle_tag_operation: "{child.tag}" not handled'
                    )

    def handle_tag_operand(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the operation elementss.
            <operand>
                ...
            </operand>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if (
                child.tag == "name"
                or child.tag == "literal"
                or child.tag == "operation"
            ):
                self.parseXMLTree(child, curElem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(curElem, current)
            else:
                print(f'In handle_tag_operand: "{child.tag}" not handled')

    def handle_tag_write(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the operation elementss.
            <operand>
                ...
            </operand>
        """
        assert (
            root.tag == "write"
        ), f"The tag <write> must be passed to handle_tag_write. Currently, it's {root.tag}."
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-controls" or child.tag == "outputs":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_write: "{child.tag}" not handled')
            else:
                if self.reconstructing:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    pass # Add appropriate assert or error msg here


    def handle_tag_io_controls(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the io-controls elementss.
            <io-controls>
                ...
            </io-controls>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-control":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_io_controls: "{child.tag}" not handled'
                    )
            else:
                if self.reconstructing:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    pass # Add appropriate assert or error msg here

    def handle_tag_io_control(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the io-control elementss.
            <io-control>
                ...
            </io-control>
        """
        for child in root:
            self.clean_attrib(child)
            # To make io-control elements simpler, the code below will append io-control-spec's attributes
            # to its parent (io-control). This will eliminate at least one recursion in translate.py to
            # retrieve the io-control information
            if child.tag == "io-control-spec":
                current.attrib.update(child.attrib)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-control" or child.tag == "literal":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_io_control: "{child.tag}" not handled'
                    )
            else:
                if self.reconstructing:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    pass # Add appropriate assert or error msg here

    def handle_tag_outputs(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the outputs elementss
            <outputs>
                ...
            </outputs>
        """
        assert (
            root.tag == "outputs"
        ), f"The tag <outputs> must be passed to handle_tag_outputs. Currently, it's {root.tag}."
        for child in root:
            self.clean_attrib(child)
            if child.tag == "output":
                curElem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, curElem, current, traverse)
            elif child.tag == "name":
                # curElem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, current, current, traverse)
            else:
                print(f'In handle_tag_outputs: "{child.tag}" not handled')

    def handle_tag_output(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the output elementss.
            <output>
                ...
            </output>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "name" or child.tag == "literal":
                self.parseXMLTree(child, curElem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(curElem, current)
            else:
                print(f'In handle_tag_outputs: "{child.tag}" not handled')

    def handle_tag_format(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the format elementss.
            <format>
                ...
            </format>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "format-items":
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                if child.tag != "label":
                    print(f'In handle_tag_format: "{child.tag}" not handled')

    def handle_tag_format_items(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the format_items and its sub-elementss
            <format_items>
                <format_item>
                    ...
                </format_item>
                ...
            </format_items>
        """
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "format-items" or child.tag == "format-item":
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_format_items: "{child.tag}" not handled')

    def handle_tag_print(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the print tags.
            <print>
                ...
            </print>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag != "print-stmt":
                curElem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "outputs":
                self.parseXMLTree(child, curElem, current, traverse)
            else:
                if child.tag != "print-format" and child.tag != "print-stmt":
                    print(f'In handle_tag_print: "{child.tag}" not handled')

    def handle_tag_open(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the open elementss.
            <open>
                ...
            </open>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_open: "{child.tag}" not handled')
            else:
                if child.tag == "open-stmt":
                    current.attrib.update(child.attrib)
                else:
                    print(
                        f'In handle_tag_open: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_keyword_arguments(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the keyword-arguments and keyword-argument elementss.
            <keyword-arguments>
                <keyword-argument>
                    ...
                </keyword-argument>
                ...
            </keyword-arguments>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-argument" or child.tag == "literal":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_keyword_arguments - {root.tag}: "{child.tag}" not handled'
                    )
            else:
                if child.tag != "keyword-argument":
                    print(
                        f'In handle_tag_keyword_arguments - {root.tag}: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_read(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the read elementss. 
            <read>
                ...
            </read>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "io-controls" or child.tag == "inputs":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_read: "{child.tag}" not handled')
            else:
                if child.tag == "read-stmt":
                    current.attrib.update(child.attrib)
                else:
                    print(
                        f'In handle_tag_read: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_inputs(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the inputs and input elementss.
            <inputs>
                <input>
                    ...
                </input>
                ...
            </inputs>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "input" or child.tag == "name":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_input - {root.tag}: "{child.tag}" not handled'
                    )
            else:
                print(
                    f'In handle_tag_input - {root.tag}: Empty elements "{child.tag}" not handled'
                )

    def handle_tag_close(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the close elementss. 
            <close>
                ...
            </close>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_close: "{child.tag}" not handled')
            else:
                if child.tag == "close-stmt":
                    current.attrib.update(child.attrib)
                else:
                    print(
                        f'In handle_tag_close: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_call(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the call elements. 
            <call>
                ...
            </call>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "name":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_call: "{child.tag}" not handled')
            else:
                if child.tag == "call-stmt":
                    current.attrib.update(child.attrib)
                else:
                    print(
                        f'In handle_tag_call: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_subroutine(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the subroutine elements. 
            <subroutine>
                ...
            </subroutine>
        """
        self.current_scope = root.attrib["name"]
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    if child.tag == "body":
                        self.current_body_scope = curElem
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_subroutine: "{child.tag}" not handled'
                    )
            else:
                if child.tag != "end-subroutine-stmt":
                    print(
                        f'In handle_tag_subroutine: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_arguments(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the arguments. 
            <arguments>
                ...
            </arsuments>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "argument":
                curElem = ET.SubElement(current, child.tag, child.attrib)
            else:
                print(f'In handle_tag_variable: "{child.tag}" not handled')

    def handle_tag_if(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the if elements.
            <if>
                ...
            </if>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    if child.tag == "body" and "goto-target" in child.attrib:
                        # Keep a track of if body where label statement will be reconstrcuted under
                        self.goto_body = curElem
                    else:
                        if child.tag == "header" and self.need_goto_elimination:
                            self.need_op_negation = True
                        self.parseXMLTree(child, curElem, current, traverse)
                        if child.tag == "body" and self.is_stop:
                            current.attrib["extract-header"] = "true"
                            self.is_stop = False
                else:
                    print (f'In handle_tag_if: "{child.tag}" not handled')
            else:
                if child.tag == "if-stmt":
                    current.attrib.update(child.attrib)
                elif child.tag == "header" and self.reconstructing:
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_if: Empty elements "{child.tag}" not handled'
                    )
        # If label appears before <goto>, mark <if> with goto-move to move it later (1st traverse)
        if self.label_before:
            # REMOVE
            print ("traverse - if label_before: ", traverse)
            current.attrib['goto-move'] = "true"
            self.statements_to_reconstruct_before.append(current)

    def handle_tag_stop(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the stop elements 
            <stop>
                ...
            </stop>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "stop-code":
                current.attrib.update(child.attrib)
            else:
                print(f'In handle_tag_stop: "{child.tag}" not handled')

    def handle_tag_step(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the step elements.
            <step>
                ...
            </step>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "operation" or child.tag == "literal":
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_step: "{child.tag}" not handled')
            else:
                print(
                    f'In handle_tag_step: Empty elements "{child.tag}" not handled'
                )

    def handle_tag_return(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the return and return-stmt elementss.
            However, since 'return-stmt' is an empty elements with no sub-elementss, the function will not keep
            the elements, but move the attribute to its parent elements, return.
            <return>
                ...
            </return>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "return-stmt":
                current.attrib.update(child.attrib)
            else:
                print(f'In handle_tag_return: "{child.tag}" not handled')

    def handle_tag_function(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the function elements.
            <function>
                ...
            </function>
        """
        self.current_scope = root.attrib["name"]
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    if child.tag == "header":
                        self.is_function_arg = True
                    elif child.tag == "body":
                        self.current_body_scope = curElem
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(f'In handle_tag_function: "{child.tag}" not handled')
            else:
                if (
                    child.tag == "function-stmt"
                    or child.tag == "end-function-stmt"
                    or child.tag == "function-subprogram"
                ):
                    curElem = ET.SubElement(current, child.tag, child.attrib)
                else:
                    print(
                        f'In handle_tag_function: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_use(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the use elementss.
            <use>
                ...
            </use>
        """
        for child in root:
            self.clean_attrib(child)
            if child.tag == "use-stmt" or child.tag == "only":
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_use: "{child.tag}" not handled')

    def handle_tag_module(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the module elementss.
            <module>
                ...
            </module>
        """
        for child in root:
            self.clean_attrib(child)
            if (
                child.tag == "header"
                or child.tag == "body"
                or child.tag == "module-stmt"
                or child.tag == "members"
                or child.tag == "end-module-stmt"
                or child.tag == "contains-stmt"
            ):
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_module: "{child.tag}" not handled')

    def handle_tag_initial_value(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the initial-value elementss.
            <initial-value>
                ...
            </initial-value>
        """
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal":
                    self.parseXMLTree(child, curElem, current, traverse)
                else:
                    print(
                        f'In handle_tag_initial_value: "{child.tag}" not handled'
                    )
            else:
                if child.tag == "initialization":
                    current.attrib.update(child.attrib)
                else:
                    print(
                        f'In handle_tag_initial_value: Empty elements "{child.tag}" not handled'
                    )

    def handle_tag_members(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the members elementss.
            <members>       <member>
                ...     or      ...
            </members>      </member>
        """
        for child in root:
            self.clean_attrib(child)
            if (
                child.tag == "subroutine"
                or child.tag == "module-subprogram"
                or child.tag == "module-subprogram-part"
                or child.tag == "declaration"
                or child.tag == "prefix"
                or child.tag == "function"
            ):
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_members: "{child.tag}" not handled')

    def handle_tag_only(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the only elementss.
            <only>
                ...
            </only>
        """
        for child in root:
            if (
                child.tag == "name"
                or child.tag == "only"
                or child.tag == "only-list"
            ):
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_only: "{child.tag}" not handled')

    def handle_tag_length(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the length elementss.
            <length>
                ...
            </length>
        """
        for child in root:
            if child.tag == "literal" or child.tag == "char-length":
                curElem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem, current, traverse)
            else:
                print(f'In handle_tag_length: "{child.tag}" not handled')

    #################################################################
    #                                                               #
    #                       XML TAG PARSER                          #
    #                                                               #
    #################################################################

    def parseXMLTree(self, root, current, parent, traverse):
        """
            parseXMLTree

            Arguments:
                root: The current root of the tree.
                current: Current element.
                parent: Parent element of the current.
                traverse: Keeps the track of number of traverse time..
        
            Returns:
                None
                
            Recursively traverse through the nested XML AST tree and calls appropriate tag handler, which will generate
            a cleaned version of XML tree for translate.py. Any new tags handlers must be added under this this function.
        """
        if root.tag == "file":
            self.handle_tag_file(root, current, parent, traverse)
        elif root.tag == "program":
            self.handle_tag_program(root, current, parent, traverse)
        elif root.tag == "header":
            self.handle_tag_header(root, current, parent, traverse)
        elif root.tag == "specification":
            self.handle_tag_specification(root, current, parent, traverse)
        elif root.tag == "body":
            self.handle_tag_body(root, current, parent, traverse)
        elif root.tag == "declaration":
            self.handle_tag_declaration(root, current, parent, traverse)
        elif root.tag == "type":
            self.handle_tag_type(root, current, parent, traverse)
        elif root.tag == "variables":
            self.handle_tag_variables(root, current, parent, traverse)
        elif root.tag == "variable":
            self.handle_tag_variable(root, current, parent, traverse)
        elif root.tag == "statement":
            self.handle_tag_statement(root, current, parent, traverse)
        elif root.tag == "assignment":
            self.handle_tag_assignment(root, current, parent, traverse)
        elif root.tag == "target":
            self.handle_tag_target(root, current, parent, traverse)
        elif root.tag == "value":
            self.handle_tag_value(root, current, parent, traverse)
        elif root.tag == "names":
            self.handle_tag_names(root, current, parent, traverse)
        elif root.tag == "name":
            self.handle_tag_name(root, current, parent, traverse)
        elif root.tag == "literal":
            self.handle_tag_literal(root, current, parent, traverse)
        elif root.tag == "dimensions":
            self.handle_tag_dimensions(root, current, parent, traverse)
        elif root.tag == "dimension":
            self.handle_tag_dimension(root, current, parent, traverse)
        elif root.tag == "loop":
            self.handle_tag_loop(root, current, parent, traverse)
        elif root.tag == "index-variable" or root.tag == "range":
            self.handle_tag_index_variable_or_range(root, current, parent, traverse)
        elif root.tag == "lower-bound":
            self.handle_tag_lower_bound(root, current, parent, traverse)
        elif root.tag == "upper-bound":
            self.handle_tag_upper_bound(root, current, parent, traverse)
        elif root.tag == "subscripts":
            self.handle_tag_subscripts(root, current, parent, traverse)
        elif root.tag == "subscript":
            self.handle_tag_subscript(root, current, parent, traverse)
        elif root.tag == "operation":
            self.handle_tag_operation(root, current, parent, traverse)
        elif root.tag == "operand":
            self.handle_tag_operand(root, current, parent, traverse)
        elif root.tag == "write":
            self.handle_tag_write(root, current, parent, traverse)
        elif root.tag == "io-controls":
            self.handle_tag_io_controls(root, current, parent, traverse)
        elif root.tag == "io-control":
            self.handle_tag_io_control(root, current, parent, traverse)
        elif root.tag == "outputs":
            self.handle_tag_outputs(root, current, parent, traverse)
        elif root.tag == "output":
            self.handle_tag_output(root, current, parent, traverse)
        elif root.tag == "format":
            self.handle_tag_format(root, current, parent, traverse)
        elif root.tag == "format-items" or root.tag == "format-item":
            self.handle_tag_format_items(root, current, parent, traverse)
        elif root.tag == "print":
            self.handle_tag_print(root, current, parent, traverse)
        elif root.tag == "open":
            self.handle_tag_open(root, current, parent, traverse)
        elif root.tag == "keyword-arguments" or root.tag == "keyword-argument":
            self.handle_tag_keyword_arguments(root, current, parent, traverse)
        elif root.tag == "read":
            self.handle_tag_read(root, current, parent, traverse)
        elif root.tag == "inputs" or root.tag == "input":
            self.handle_tag_inputs(root, current, parent, traverse)
        elif root.tag == "close":
            self.handle_tag_close(root, current, parent, traverse)
        elif root.tag == "call":
            self.handle_tag_call(root, current, parent, traverse)
        elif root.tag == "subroutine":
            self.handle_tag_subroutine(root, current, parent, traverse)
        elif root.tag == "arguments":
            self.handle_tag_arguments(root, current, parent, traverse)
        elif root.tag == "if":
            self.handle_tag_if(root, current, parent, traverse)
        elif root.tag == "stop":
            self.handle_tag_stop(root, current, parent, traverse)
        elif root.tag == "step":
            self.handle_tag_step(root, current, parent, traverse)
        elif root.tag == "return":
            self.handle_tag_return(root, current, parent, traverse)
        elif root.tag == "function":
            self.handle_tag_function(root, current, parent, traverse)
        elif root.tag == "use":
            self.handle_tag_use(root, current, parent, traverse)
        elif root.tag == "module":
            self.handle_tag_module(root, current, parent, traverse)
        elif root.tag == "initial-value":
            self.handle_tag_initial_value(root, current, parent, traverse)
        elif root.tag == "members":
            self.handle_tag_members(root, current, parent, traverse)
        elif root.tag == "only":
            self.handle_tag_only(root, current, parent, traverse)
        elif root.tag == "length":
            self.handle_tag_length(root, current, parent, traverse)
        else:
            print(
                f"In the parseXMLTree and, currently, {root.tag} is not supported"
            )

    #################################################################
    #                                                               #
    #                       RECONSTRUCTORS                          #
    #                                                               #
    #################################################################


    def reconstruct_derived_type_declaration(self):
        """
            reconstruct_derived_type_declaratione reconstruct the derived type with the collected 
            derived type declaration elements in the handle_tag_declaration and handle_tag_type.
        """
        if self.derived_type_var_holder_list:
            literal = ET.Element("")
            is_dimension = False

            # Since component-decl-list appears after component-decl, the program needs to 
            # iterate the list once first to pre-collect the variable counts.
            counts = []
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "component-decl-list":
                    counts.append(elem.attrib["count"])

            # Initialize count to 0 for <variables> count attribute.
            count = 0
            # 'component-decl-list__begin' tag is an indication of all the derived type member
            # variable declarations will follow.
            derived_type = ET.SubElement(self.parent_type, "derived-types")
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "intrinsic-type-spec":
                    keyword2 = ""
                    if elem.attrib["keyword2"] == "":
                        keyword2 = "none"
                    else:
                        keyword2 = elem.attrib["keyword2"]
                    attributes = {
                        "hasKind": "false",
                        "hasLength": "false",
                        "name": elem.attrib["keyword1"],
                        "is_derived_type": str(self.is_derived_type),
                        "keyword2": keyword2,
                    }
                    newType = ET.SubElement(derived_type, "type", attributes)
                elif elem.tag == "derived-type-spec":
                    attributes = {
                        "hasKind": "false",
                        "hasLength": "false",
                        "name": elem.attrib["typeName"],
                        "is_derived_type": str(self.is_derived_type),
                        "keyword2": "none",
                    }
                    newType = ET.SubElement(derived_type, "type", attributes)
                elif elem.tag == "literal":
                    literal = elem
                elif elem.tag == "component-array-spec":
                    is_dimension = True
                elif elem.tag == "component-decl":
                    if not is_dimension:
                        if len(counts) > count:
                            attr = {"count": counts[count]}
                            new_variables = ET.SubElement(
                                derived_type, "variables", attr
                            )  # <variables _attribs_>
                            count += 1
                        var_attribs = {
                            "has_initial_value": elem.attrib[
                                "hasComponentInitialization"
                            ],
                            "name": elem.attrib["id"],
                            "is_array": "false",
                        }
                        # Store variable name in the non array tracker
                        self.declared_non_array_vars.update(
                            {elem.attrib["id"]: self.current_scope}
                        )
                        new_variable = ET.SubElement(
                            new_variables, "variable", var_attribs
                        )  # <variable _attribs_>
                        if elem.attrib["hasComponentInitialization"] == "true":
                            init_value_attrib = ET.SubElement(
                                new_variable, "initial-value"
                            )
                            new_literal = ET.SubElement(
                                init_value_attrib, "literal", literal.attrib
                            )  # <initial-value _attribs_>
                    else:
                        new_dimensions = ET.SubElement(
                            derived_type, "dimensions", {"count": "1"}
                        )  # <dimensions count="1">
                        new_dimension = ET.SubElement(
                            new_dimensions, "dimension", {"type": "simple"}
                        )  # <dimension type="simple">
                        new_literal = ET.SubElement(
                            new_dimension, "literal", literal.attrib
                        )  # <literal type="" value="">
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
                            "name": elem.attrib["id"],
                            "is_array": "true",
                        }
                        # Store variable name in the array tracker
                        self.declared_array_vars.update(
                            {elem.attrib["id"]: self.current_scope}
                        )
                        new_variable = ET.SubElement(
                            new_variables, "variable", var_attribs
                        )
                        is_dimension = False

            # Once one derived type was successfully constructed, clear all the elementss of a derived type list
            self.derived_type_var_holder_list.clear()

    def reconstruct_derived_type_ref(self, current):
        """
            This function reconstruct the id into x.y.k form from the messy looking id.
            One thing to notice is that this new form was generated in the python syntax,
            so it is a pre-process for translate.py and even pyTranslate.py that
        """
        assert (
            current.tag == "name"
        ), f"The tag <name> must be passed to reconstruct_derived_type_ref. Currently, it's {current.tag}."
        # First the root <name> id gets the very first variable reference i.e. x in x.y.k (or x%y%k in Fortran syntax)
        current.attrib["id"] = self.derived_type_var_holder_list[0]
        if (
            current.attrib["id"] in self.declared_array_vars
            and self.declared_array_vars[current.attrib["id"]]
            == self.current_scope
        ):
            current.attrib["hasSubscripts"] = "true"
            current.attrib["is_array"] = "true"
        else:
            current.attrib["hasSubscripts"] = "false"
            current.attrib["is_array"] = "false"

        number_of_vars = len(self.derived_type_var_holder_list)
        attributes = {}
        parent_ref = current
        self.derived_type_refs.append(parent_ref)
        for var in range(1, number_of_vars):
            variable_name = self.derived_type_var_holder_list[var]
            attributes.update(current.attrib)
            attributes["id"] = variable_name
            if (
                variable_name in self.declared_array_vars
                and self.declared_array_vars[variable_name]
                == self.current_scope
            ):
                attributes["hasSubscripts"] = "true"
                attributes["is_array"] = "true"
            else:
                attributes["is_array"] = "false"
            # Create N (number_of_vars) number of new subElement under the root <name> for each referencing variable
            reference_var = ET.SubElement(parent_ref, "name", attributes)
            parent_ref = reference_var
            self.derived_type_refs.append(parent_ref)
        self.derived_type_var_holder_list.clear()  # Clean up the list for re-use

    def reconstruct_format(self):
        """
            This function is for reconstructing the <format> under the <statement> element.
            The OFP XML nests formats under:
                (1) statement
                (2) declaration
                (3) loop
            tags, which are wrong except one that is declared under the statement.
            Therefore, those formats declared under (2) and (3) will be extracted and reconstructed
            to be nested under (1) in this function.
        """
        root_scope = ET.SubElement(self.current_body_scope, "statement")
        curElem = ET.SubElement(root_scope, "format")
        self.parseXMLTree(self.format_holder, curElem, traverse)

    def reconstruct_derived_type_names(self, current):
        """
            This function reconstructs derived type reference syntax tree.
            However, this functions is actually a preprocessor for the real final reconstruction.
        """
        # Update reconstruced derived type references
        assert (
            self.is_derived_type_ref == True
        ), "'self.is_derived_type_ref' must be true"
        numPartRef = int(current.attrib["numPartRef"])
        for idx in range(1, len(self.derived_type_refs)):
            self.derived_type_refs[idx].attrib.update(
                {"numPartRef": str(numPartRef)}
            )
        # Re-initialize to original values
        self.derived_type_refs.clear()

    def reconstruct_name_element(self, curElem, current):
        """
            This function performs a final reconstruction of derived type name element that was
            preprocessed by 'reconstruct_derived_type_names' function. This function traverses
            the preprocessed name element (including sub-elements) and split & store <name> and
            <subscripts> into separate lists. Then, it comibines and reconstructs two lists
            appropriately.
        """
        name_elements = [curElem]
        # Remove the original <name> elements.
        current.remove(curElem)
        # Split & Store <name> element and <subscripts>.
        subscripts_holder = []
        for child in curElem:
            if child.tag == "subscripts":
                subscripts_holder.append(child)
            else:
                name_elements.append(child)
                for third in child:
                    name_elements.append(third)

        # Combine & Reconstruct <name> element.
        subscript_num = 0
        curElem = ET.SubElement(
            current, name_elements[0].tag, name_elements[0].attrib
        )
        curElem.attrib["is_derived_type_ref"] = "true"
        if curElem.attrib["hasSubscripts"] == "true":
            curElem.append(subscripts_holder[subscript_num])
            subscript_num += 1

        numPartRef = int(curElem.attrib["numPartRef"]) - 1
        name_element = ET.Element("")
        for idx in range(1, len(name_elements)):
            name_elements[idx].attrib["numPartRef"] = str(numPartRef)
            numPartRef -= 1
            name_element = ET.SubElement(
                curElem, name_elements[idx].tag, name_elements[idx].attrib
            )
            name_element.attrib["is_derived_type_ref"] = "true"
            # In order to handle the nested subelements of <name>, update the curElem at each iteration.
            curElem = name_element
            if name_elements[idx].attrib["hasSubscripts"] == "true":
                name_element.append(subscripts_holder[subscript_num])
                subscript_num += 1

        # Clean out the lists for recyling. This is not really needed as they are local lists, but just in case.
        name_elements.clear()
        subscripts_holder.clear()
        self.need_reconstruct = False

    def reconstruct_goto_after_label(self):
        for stmt in self.statements_to_reconstruct_after:
            if "target-label-statement" not in stmt.attrib:
                # Remove "goto-target" from the <body> (2nd traverse)
                # del self.goto_body.attrib["goto-target"]
                statement_tag = ET.SubElement(self.goto_body, "statement")
                for child in stmt:
                    curElem = ET.SubElement(statement_tag, child.tag, child.attrib)
                    if child.text:
                        self.parseXMLTree(child, curElem, traverse)
        # Set all holders and checkers (markers) to default
        self.goto_body = ET.Element('')
        self.label_after = False
        self.reconstruct_after_case_now = False
        self.goto_target_lbl_after.clear()

    def reconstruct_goto_before_label(self, parent, traverse):
        self.reconstructing = True
        # REMOVE
        print ("in reconstruct_goto_before_label")
        
        loop = ET.SubElement(parent, "loop", {"type":"do-while"})
        for stmt in self.statements_to_reconstruct_before:
            # REMOVE
            print ("stmt: ", stmt.tag, stmt.attrib)
            if stmt.tag == "if" and "extract-header" in stmt.attrib:
                for child in stmt:
                    if child.tag == "header":
                        loop_header = ET.SubElement(loop, child.tag, child.attrib)
                        self.need_op_negation = True
                        self.parseXMLTree(child, loop_header, loop, traverse)

        for stmt in self.statements_to_reconstruct_before:
            if len(stmt) > 0:
                statement_tag = ET.SubElement(parent, stmt.tag, stmt.attrib)

                for child in stmt:
                    curElem = ET.SubElement(statement_tag, child.tag, child.attrib)
                    if len(child) > 0:
                        self.parseXMLTree(child, curElem, statement_tag, traverse)
                    
                if "goto-move" in statement_tag.attrib:
                    del statement_tag.attrib["goto-move"]
                if "target-label-statement" in statement_tag.attrib:
                    del statement_tag.attrib["target-label-statement"]
        # Set all holders and checkers (markers) to default
        self.label_before = False
        self.reconstruct_before_case_now = False
        self.reconstructing = False
        self.statements_to_reconstruct_before.clear()

    #################################################################
    #                                                               #
    #                       MISCELLANEOUS                           #
    #                                                               #
    #################################################################

    def clean_derived_type_ref(self, current):
        """
            This function will clean up the derived type referencing syntax, which is stored in a form of "id='x'%y" in the id attribute.
            Once the id gets cleaned, it will call the reconstruc_derived_type_ref function to reconstruct and replace the messy version
            of id with the cleaned version.
        """
        current_id = current.attrib[
            "id"
        ]  # 1. Get the original form of derived type id, which is in a form of, for example, id="x"%y in the original XML.
        self.derived_type_var_holder_list.append(
            self.clean_id(current_id)
        )  # 2. Extract the first variable name, for example, x in this case.
        percent_sign = current_id.find(
            "%"
        )  # 3. Get the location of the '%' sign.
        self.derived_type_var_holder_list.append(
            current_id[percent_sign + 1 : len(current_id)]
        )  # 4. Get the field variable. y in this example.
        self.reconstruct_derived_type_ref(current)

    def clean_id(self, unrefined_id):
        """
            This function refines id (or value) with quotation makrs included by removing them and returns only the variable name.
            For example, from "OUTPUT" to OUTPUT and "x" to x. Thus, the id name will be modified as below:
                Unrefined id: id = ""OUTPUT""
                Refined id: id = "OUTPUT"
        """
        return re.findall(r"\"([^\"]+)\"", unrefined_id)[0]

    def clean_attrib(self, elements):
        """
            The original XML elements holds 'eos' and 'rule' attributes that are not necessary and being used.
            Thus, this function will remove them in the rectified version of XML.
        """
        if "eos" in elements.attrib:
            elements.attrib.pop("eos")
        if "rule" in elements.attrib:
            elements.attrib.pop("rule")

#################################################################
#                                                               #
#                     NON-CLASS FUNCTIONS                       #
#                                                               #
#################################################################

def is_empty(elem):
    """
        This function is just a helper function for
        check whether the passed elements (i.e. list)
        is empty or not
    """
    if not elem:
        return True
    else:
        return False

def indent(elem, level=0):
    """
        This function indents each level of XML.
        Source: https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementstree-in-python
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
    return elem


def buildNewAST(filename: str):
    """
        Using the list of cleaned AST, construct a new XML AST and write to a file
    """
    # A variable for keep a track of number of traverse time
    traverse = 1

    # Read AST from the OFP generated XML file
    ast = ET.parse(filename)
    # Get a root of a tree
    root = ast.getroot()
    XMLCreator = RectifyOFPXML()
    # A root of the new AST
    newRoot = ET.Element(root.tag, root.attrib)
    # First add the root to the new AST list
    for child in root:
        # Handle only non-empty elementss
        if child.text:
            curElem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, curElem, newRoot, traverse)

    tree = ET.ElementTree(indent(newRoot))
    
    # Checks if the rectified AST requires goto elimination,
    # if it does, it does a 2nd traverse to eliminate and
    # reconstruct the AST once more
    if XMLCreator.need_goto_elimination:
        traverse += 1
        # REMOVE
        print ("2nd Traverse Begin")
        root = tree.getroot()	
        newRoot = ET.Element(root.tag, root.attrib)	
        for child in root:	
            if child.text:	
                curElem = ET.SubElement(newRoot, child.tag, child.attrib)	
                XMLCreator.parseXMLTree(child, curElem, newRoot, traverse)	
        tree = ET.ElementTree(indent(newRoot))	
        XMLCreator.need_goto_elimination = False

    rectFilename = filename.split("/")[-1]
    tree.write(f"tmp/rectified_{rectFilename}")


def buildNewASTfromXMLString(xmlString: str) -> ET.Element:
    ast = ET.XML(xmlString)
    XMLCreator = RectifyOFPXML()
    # A root of the new AST
    newRoot = ET.Element(ast.tag, ast.attrib)
    # First add the root to the new AST list
    for child in ast:
        # Handle only non-empty elementss
        if child.text:
            curElem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, curElem, newRoot, traverse)

    return newRoot


if __name__ == "__main__":
    filename = sys.argv[1]
    # Build a new cleaned AST XML
    buildNewAST(filename)
