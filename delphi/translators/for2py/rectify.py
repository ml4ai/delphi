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
                ".lt." : ".ge.",
                ".gt." : ".le.",
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
        self.continue_elimination = False
        self.need_op_negation = False
        self.need_reconstruct = False
        self.reconstruct_after_case_now = False
        self.reconstruct_before_case_now = False
        self.label_before = False
        self.label_after = False
        self.is_stop = False
        self.collect_stmts_after_goto = False
        self.collect_stmts_after_label = False
        self.reconstruction_for_after_done = False
        self.reconstruction_for_before_done = False
        self.encapsulate_under_do_while = False
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
        #
        self.reconstructing_scope = None

        # Keep a track of collected goto-stmts and labels
        # for goto elimination and reconstruction
        self.stmts_after_goto = {
                                    'goto-stmts': [],
                                    'labels': [],
        }

        # Dictionaries to hold statements
        # before and after the gotos and labels
        self.statements_to_reconstruct_before = {
                                                    "stmts-follow-label": [], 
                                                    "count-gotos" : 0,
        }
        self.statements_to_reconstruct_after = {
                                                    "stmts-follow-goto": [],
                                                    "stmts-follow-label": [],
                                                    "count-gotos" : 0,
       }
        # Keeps tracks of the goto-after scope
        self.goto_scopes = {}
        self.current_goto_scope = []

        self.cur_derived_type_name = None
        self.current_scope = None
        self.goto_eliminated_elements_for_before = None

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

    """
        Nested child tag list
    """
    file_child_tags = [
        "program", 
        "subroutine", 
        "module", 
        "declaration", 
        "function", 
        "prefix"
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
    ]

    loop_child_tags = [
        "header",
        "body",
        "format"
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
        "name",
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag in self.file_child_tags:
                self.parseXMLTree(child, cur_elem, current, traverse)
            else:
                print(f'In handle_tag_file: "{child.tag}" not handled')

    def handle_tag_program(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the program elementss.
            <program>
                ...
            </program>
        """
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "header" or child.tag == "body":
                if child.tag == "body":
                    self.current_body_scope = cur_elem
                    # Mark the body to indicate that this body is
                    # the main body of the program
                    cur_elem.attrib['type'] = "program-body"
                self.parseXMLTree(child, cur_elem, current, traverse)
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
            if child.text or len(child) > 0:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag in self.header_child_tags:
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print(f'In handle_tag_header: "{child.tag}" not handled')
            else:
                if child.tag == "subroutine-stmt":
                    current.attrib.update(child.attrib)
                elif child.tag in self.header_child_tags:
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
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
        new_parent = current
        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag in self.body_child_tags:
                    self.parseXMLTree(child, cur_elem, current, traverse)
                    # Reconstruction of statements
                    if (
                        traverse > 1
                        and "type" in current.attrib 
                        and current.attrib['type'] == "program-body"
                    ):
                        # Remove statements that is marked to be removed (2nd traverse)
                        if (
                            "goto-remove" in child.attrib 
                            or "goto-move" in child.attrib
                        ):
                            current.remove(cur_elem)

                        if self.reconstruct_after_case_now and not self.reconstruction_for_after_done:
                            self.reconstruct_goto_after_label(new_parent, traverse, self.statements_to_reconstruct_after)
                            if self.label_lbl_for_before:
                                self.continue_elimination = True
                        if self.reconstruct_before_case_now and not self.reconstruction_for_before_done:
                            self.reconstruct_goto_before_label(new_parent, traverse)
                            if self.label_lbl_for_after :
                                self.continue_elimination = True
                        if not self.label_lbl_for_before and not self.label_lbl_for_after:
                            self.continue_elimination = False
                else:
                    print (f'In handle_tag_body: "{child.tag}" not handled')
            else:
                if child.tag in self.body_child_tags and child.tag != "statement":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                elif child.tag == "statement":
                    if len(child) > 0:
                        cur_elem = ET.SubElement(current, child.tag, child.attrib)
                        self.parseXMLTree(child, cur_elem, current, traverse)
                else:
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
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "declaration" or child.tag == "use":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
            if len(child) > 0 or child.text:
                if child.tag in self.declaration_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder = child
                    else:
                        cur_elem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        if child.tag == "dimensions":
                            self.is_array = True
                        self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                elif (
                    child.tag == "component-decl"
                    or child.tag == "component-decl-list"
                ):
                    current.attrib['type'] = "derived-type"
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
                if child.attrib['keyword2'] == "":
                    current.attrib['keyword2'] = "none"
                else:
                    current.attrib['keyword2'] = child.attrib['keyword2']
            else:
                current.attrib['keyword2'] = "none"
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                self.parent_type = current
                self.parseXMLTree(child, cur_elem, current, traverse)
            elif child.tag == "intrinsic-type-spec":
                if self.is_derived_type:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "derived-type-stmt" and self.is_derived_type:
                # Modify or add 'name' attribute of the <type> elements with the name of derived type name
                current.set("name", child.attrib['id'])
                # And, store the name of the derived type name for later setting the outer most <type> elements's name attribute
                self.cur_derived_type_name = child.attrib['id']
            elif child.tag == "derived-type-spec":
                if not self.is_derived_type:
                    self.is_derived_type = True
                    current.set("name", child.attrib['typeName'])
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, cur_elem, current, traverse)
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
            if len(child) > 0 or child.text:
                # Up to this point, all the child (nested or sub) elements were
                # <variable>
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                cur_elem.set("is_array", str(self.is_array).lower())
                self.parseXMLTree(child, cur_elem, current, traverse)
            else:
                if child.tag == "variable":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)

    def handle_tag_variable(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the variables elementss.
            <variable>
                ...
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

        for child in root:
            self.clean_attrib(child)
            if len(child) > 0 or child.text:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "initial-value":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
        assert (
                root.tag == "statement"
        ), f"The tag <statement> must be passed to handle_tag_statement. Currently, it's {root.tag}."
        
        label_presented = False
        
        for child in root:
            self.clean_attrib(child)
            if child.tag in self.statement_child_tags:
                if child.tag == "stop":
                    self.is_stop = True
                    current.attrib['has-stop'] = "true"

                cur_elem = ET.SubElement(current, child.tag, child.attrib)

                if child.tag == "label":
                    current.attrib['label'] = child.attrib['lbl']
                    label_presented = True
                    lbl = child.attrib['lbl']

                    # Label-before case
                    if (
                        traverse == 1
                        and not self.goto_target_lbl_after
                        or lbl not in self.goto_target_lbl_after
                    ):
                        # Since we want to handle label_after case before
                        # label_before when both cases appear in the code,
                        # we ignore all label_bafore case until _after case
                        # get handled. Thus, mark label_before to false
                        if self.label_after:
                            self.label_before = False
                        else:
                            self.label_before = True
                        if lbl not in self.label_lbl_for_before:
                            self.label_lbl_for_before.append(lbl)
                        
                        # Keep a track of label_before case
                        # goto-statements' label scopes
                        if self.current_goto_scope:
                            self.goto_scopes[self.current_goto_scope[-1]]['label'].append(lbl)
                    # Label-after case
                    else:
                        self.collect_stmts_after_goto = False
                        self.collect_stmts_after_label = True
                        if lbl not in self.label_lbl_for_after:
                            self.label_lbl_for_after.append(lbl)

                        if self.current_goto_scope:
                            # If a goto-stmt encounters its target label,
                            # then it's the end of scope
                            self.goto_scopes[lbl]['end-of-scope'] = True
                            del self.current_goto_scope[-1]
                            for scope in self.goto_scopes:
                                if lbl in self.goto_scopes[scope]['goto']:
                                    self.goto_scopes[scope]['label'].append(lbl)

                    if traverse == 1:
                        if self.label_before or lbl in self.label_lbl_for_before:
                            current.attrib['goto-move'] = "true"
                        else:
                            current.attrib['goto-remove'] = "true"
                        current.attrib['target-label-statement'] = "true"
                    if traverse > 1 and self.collect_stmts_after_label:
                        self.reconstruct_after_case_now = True

                # Since <format> is followed by <label>,
                # check the case and undo all operations done for goto.
                if traverse == 1 and child.tag == "format" and label_presented:
                    del self.label_lbl[-1]
                    del current.attrib['target-label-statement']
                    del current.attrib['goto-move']
                    label_presented = False
                    self.label_before = False

                if child.text or len(child) > 0:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                # Reaching goto-stmt is a flag to stop collecting stmts
                if traverse == 1 and self.collect_stmts_after_label:
                    current.attrib['goto-remove'] = "true"
                    current.attrib['has-goto-stmt'] = "true"
                    self.statements_to_reconstruct_after['stmts-follow-label'].append(current)
                    self.collect_stmts_after_label = False

                self.need_goto_elimination = True
                target_lbl = child.attrib['target_label']
                current.attrib['goto-stmt'] = "true"
                current.attrib['lbl'] = target_lbl
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                # A case where label appears "before" goto
                if target_lbl in self.label_lbl_for_before:
                    if traverse == 1:
                        self.statements_to_reconstruct_before['count-gotos'] += 1
                        self.goto_target_lbl_before.append(target_lbl)
                        # self.label_before = False
                        # Keep a track of label_before case <goto-stmt> scopes
                        for scope in self.goto_scopes:
                            if target_lbl in self.goto_scopes[scope]['label']:
                                # A case where no parent scope, but it encapsulates all others
                                if target_lbl in self.goto_scopes[scope]['goto'] and not self.current_goto_scope:
                                    current.attrib['parent-scope'] = "none"
                                else:
                                    self.goto_scopes[scope]['goto'].append(target_lbl)
                                    current.attrib['parent-scope'] = scope
                    else:
                        assert traverse > 1, "Reconstruction cannot happen in the first traverse"
                        if self.label_before:
                            self.reconstruct_before_case_now = True
                        return
                # A case where label appears "after" goto
                else:
                    if traverse == 1:
                        self.statements_to_reconstruct_after['count-gotos'] += 1

                        # Keep a track of label_after case <goto-stmt> scopes
                        if not self.goto_scopes or target_lbl not in self.goto_scopes:
                            self.goto_scopes[target_lbl] = {'goto':[], 'label':[], 'end-of-scope':False}
                            if self.current_goto_scope:
                                # [-1] because the last element is the current scope in process
                                self.goto_scopes[self.current_goto_scope[-1]]['goto'].append(target_lbl)
                                current.attrib['parent-scope'] = self.current_goto_scope[-1]
                            self.current_goto_scope.append(target_lbl)

                        if "parent-scope" not in current.attrib:
                            current.attrib['skip-collect'] = "true"
                        self.goto_target_lbl_after.append(target_lbl)
                        self.collect_stmts_after_goto = True
                        self.label_after = True
            else:
                print (f'In self.handle_tag_statement: "{child.tag}" not handled')

        # Statement collector (1st traverse)
        if traverse == 1:
            if self.label_before and not self.label_after:
                # Since we do not want to extract the stop statement from
                # that is not a main body, check it before extraction
                if "has-stop" not in current.attrib:
                    current.attrib['goto-move'] = "true"
                    self.statements_to_reconstruct_before['stmts-follow-label'].append(current)
                elif (
                    "has-stop" in current.attrib 
                    and "program-body" in parent.attrib
                ):
                    self.statements_to_reconstruct_before['stmts-follow-label'].append(current)
            elif self.label_after:
                if self.collect_stmts_after_goto:
                    if "has-stop" not in current.attrib:
                        current.attrib['goto-remove'] = "true"
                        self.statements_to_reconstruct_after['stmts-follow-goto'].append(current)
                    else:
                        if "program-body" in parent.attrib:
                            self.statements_to_reconstruct_after['stmts-follow-goto'].append(current)

                    if "goto-stmt" in current.attrib:
                        self.stmts_after_goto['goto-stmts'].append(current.attrib['lbl'])
                    elif "target-label-statement" in current.attrib:
                        self.stmts_after_goto['labels'].append(current.attrib['label'])
                        
                elif self.collect_stmts_after_label:
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after['stmts-follow-label'].append(current)

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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "target" or child.tag == "value":
                self.parseXMLTree(child, cur_elem, current, traverse)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "name":
                self.parseXMLTree(child, cur_elem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                # If the element holds subelements, call the XML tree parser with created new <name> element
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                        attributes['is_arg'] = "true"
                    else:
                        attributes['is_arg'] = "false"
                    cur_elem.attrib.update(attributes)
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
        current.attrib['is_array'] = "false"

        # If 'id' attribute holds '%' symbol, it's an indication of derived type referencing
        # Thus, clean up the 'id' and reconstruct the <name> AST
        if "id" in current.attrib and "%" in current.attrib['id']:
            self.is_derived_type_ref = True
            self.clean_derived_type_ref(current)

        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "subscripts" or child.tag == "assignment":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    if child.tag == "subscripts":
                        # Default
                        current.attrib['hasSubscripts'] = "true"
                        # Check whether the variable is an array AND the
                        # variable is for the current scope. This is important
                        # for derived type variable referencing
                        if (
                            current.attrib['id'] in self.declared_array_vars
                            and self.declared_array_vars[current.attrib['id']]
                            == self.current_scope
                        ):
                            # Since the procedure "call" has a same AST syntax as an array, check its type and set the "is_array" value
                            assert (
                                current.attrib['type'] != "procedure"
                            ), "Trying to assign a procedure call to is_array true. This is an error"
                            current.attrib['is_array'] = "true"
                        elif (
                            current.attrib['id']
                            in self.declared_non_array_vars
                            and self.declared_non_array_vars[
                                current.attrib['id']
                            ]
                            == self.current_scope
                        ):
                            current.attrib['hasSubscripts'] = "false"
                    self.parseXMLTree(child, cur_elem, current, traverse)
                elif child.tag == "output":
                    assert is_empty(self.derived_type_var_holder_list)
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.derived_type_var_holder_list.append(root.attrib['id'])
                    self.parseXMLTree(child, cur_elem, current, traverse)
                elif child.tag == "name":
                    self.parseXMLTree(child, current, current, traverse)
                else:
                    print(
                        f'In self.handle_tag_name: "{child.tag}" not handled'
                    )
            else:
                if child.tag == "generic_spec":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if (
                child.tag == "literal"
                or child.tag == "operation"
                or child.tag == "name"
            ):
                self.parseXMLTree(child, cur_elem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
            else:
                print(f'In self.handle_tag_value: "{child.tag}" not handled')

    def handle_tag_literal(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the literal elementss.
            <literal>
                ...
            </literal>
        """
        if '"' in current.attrib['value']:
            current.attrib['value'] = self.clean_id(current.attrib['value'])
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "stop":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "dimension":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "range":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
            if child.text or len(child) > 0:
                if child.tag in self.loop_child_tags:
                    if child.tag == "format":
                        self.is_format = True
                        self.format_holder = child
                    else:
                        cur_elem = ET.SubElement(
                            current, child.tag, child.attrib
                        )
                        self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "lower-bound"
                    or child.tag == "upper-bound"
                    or child.tag == "step"
                ):
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "literal"
                    or child.tag == "name"
                    or child.tag == "operation"
                ):
                    self.parseXMLTree(child, cur_elem, current, traverse)
                    if child.tag == "name" and self.need_reconstruct:
                        self.reconstruct_name_element(cur_elem, current)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "subscript":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if (
                    child.tag == "name"
                    or child.tag == "literal"
                    or child.tag == "operation"
                ):
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                child.attrib['operator'] = NEGATED_OP[child.attrib['operator']]
                self.need_op_negation = False
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "operand":
                self.parseXMLTree(child, cur_elem, current, traverse)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if (
                child.tag == "name"
                or child.tag == "literal"
                or child.tag == "operation"
            ):
                self.parseXMLTree(child, cur_elem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
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
            if child.text or len(child) > 0:
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-controls" or child.tag == "outputs":
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print(f'In handle_tag_write: "{child.tag}" not handled')


    def handle_tag_io_controls(self, root, current, parent, traverse):
        """
            This function handles cleaning up the XML elementss between the io-controls elementss.
            <io-controls>
                ...
            </io-controls>
        """
        for child in root:
            self.clean_attrib(child)
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.text or len(child) > 0:
                if child.tag == "io-control":
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print(
                        f'In handle_tag_io_controls: "{child.tag}" not handled'
                    )

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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "io-control" or child.tag == "literal":
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print(
                        f'In handle_tag_io_control: "{child.tag}" not handled'
                    )
            else:
                if child.tag == "literal":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)

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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                self.parseXMLTree(child, cur_elem, current, traverse)
            elif child.tag == "name":
                # cur_elem = ET.SubElement(current, child.tag, child.attrib)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "name" or child.tag == "literal":
                self.parseXMLTree(child, cur_elem, current, traverse)
                if child.tag == "name" and self.need_reconstruct:
                    self.reconstruct_name_element(cur_elem, current)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "format-items":
                self.parseXMLTree(child, cur_elem, current, traverse)
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
            cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "format-items" or child.tag == "format-item":
                self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
            if child.tag == "outputs":
                self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    if child.tag == "body":
                        self.current_body_scope = cur_elem
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
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
            if child.text or len(child) > 0:
                if child.tag == "header" or child.tag == "body":
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print (f'In handle_tag_if: "{child.tag}" not handled')
            else:
                if child.tag == "if-stmt":
                    current.attrib.update(child.attrib)
                elif child.tag == "body" and traverse > 1:
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                else:
                    print(
                        f'In handle_tag_if: Empty elements "{child.tag}" not handled'
                    )
        # If label appears before <goto>, mark <if> with goto-move to move it later (1st traverse)
        if traverse == 1:
            # Since label_after needs to be reconstructed first,
            # we skip to collect the element if label_after is True
            # Then, once the reconstruction of label_after is done,
            # then we collect those reconstructed elements
            if self.label_before and not self.label_after:
                current.attrib['goto-move'] = "true"
                self.statements_to_reconstruct_before['stmts-follow-label'].append(current)
            if self.label_after:
                if self.collect_stmts_after_goto:
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after['stmts-follow-goto'].append(current)
                elif self.collect_stmts_after_label:
                    current.attrib['goto-remove'] = "true"
                    self.statements_to_reconstruct_after['stmts-follow-label'].append(current)

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
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
        self.current_scope = root.attrib['name']
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    if child.tag == "header":
                        self.is_function_arg = True
                    elif child.tag == "body":
                        self.current_body_scope = cur_elem
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
                    self.parseXMLTree(child, cur_elem, current, traverse)
                else:
                    print(f'In handle_tag_function: "{child.tag}" not handled')
            else:
                if (
                    child.tag == "function-stmt"
                    or child.tag == "end-function-stmt"
                    or child.tag == "function-subprogram"
                ):
                    cur_elem = ET.SubElement(current, child.tag, child.attrib)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.tag == "literal":
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                cur_elem = ET.SubElement(current, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, cur_elem, current, traverse)
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
                f"In the parseXMLTree. Currently, <{root.tag}> passed from <{parent.tag}> is not supported"
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
                    counts.append(elem.attrib['count'])

            # Initialize count to 0 for <variables> count attribute.
            count = 0
            # 'component-decl-list__begin' tag is an indication of all the derived type member
            # variable declarations will follow.
            derived_type = ET.SubElement(self.parent_type, "derived-types")
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
                elif elem.tag == "derived-type-spec":
                    attributes = {
                        "hasKind": "false",
                        "hasLength": "false",
                        "name": elem.attrib['typeName'],
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
        cur_elem = ET.SubElement(root_scope, "format")
        self.parseXMLTree(self.format_holder, cur_elem, traverse)

    def reconstruct_derived_type_names(self, current):
        """
            This function reconstructs derived type reference syntax tree.
            However, this functions is actually a preprocessor for the real final reconstruction.
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
        """
            This function performs a final reconstruction of derived type name element that was
            preprocessed by 'reconstruct_derived_type_names' function. This function traverses
            the preprocessed name element (including sub-elements) and split & store <name> and
            <subscripts> into separate lists. Then, it comibines and reconstructs two lists
            appropriately.
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
            # In order to handle the nested subelements of <name>, update the cur_elem at each iteration.
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

    def reconstruct_goto_after_label(self, parent, traverse, reconstruct_target):

        number_of_gotos = reconstruct_target['count-gotos']
        stmts_follow_goto = reconstruct_target['stmts-follow-goto']
        stmts_follow_label = reconstruct_target['stmts-follow-label']
       
        goto_and_label_stmts_after_goto = []
        for stmt in stmts_follow_goto:
            if "label" in stmt.attrib:
                goto_and_label_stmts_after_goto.append(stmt.attrib['label'])
            elif "goto-stmt" in stmt.attrib:
                goto_and_label_stmts_after_goto.append(stmt.attrib['lbl'])

        num_of_goto_and_label_after_label = 0
        for stmt in stmts_follow_label:
            if (
                "label" in stmt.attrib
                or "goto-stmt" in stmt.attrib
            ):
                num_of_goto_and_label_after_label += 1

        # -2 disregarding the first and last statements 
        num_of_goto_and_label_after_label -= 2

        for i in range(num_of_goto_and_label_after_label):
            stmt = stmts_follow_label.pop(-2)
            stmts_follow_goto.append(stmt)
 
        declared_goto_flag_num = []
        self.generate_declaration_element(parent, "goto_flag", number_of_gotos, declared_goto_flag_num)
       
        # This variable is for storing goto that may appear at the end of if
        # because we want to extract one scope out and place it right 
        # after the constructed if-statement
        next_goto = []

        reconstructed_if_elem = []
        for i in range(number_of_gotos):
            self.generate_if_element(parent, stmts_follow_goto, next_goto, True, "unary", f"goto_flag_{i+1}", None, ".not.", traverse, reconstructed_if_elem)
            if reconstructed_if_elem:
                stmts_follow_goto = reconstructed_if_elem[0]
            self.generate_if_element(parent, stmts_follow_label, next_goto, False, None, f"goto_flag_{i+1}", None, None, traverse, reconstructed_if_elem)
            if reconstructed_if_elem:
                stmts_follow_label = reconstructed_if_elem[1]
            self.encapsulate_under_do_while = False

            if next_goto:
                statement = ET.SubElement(parent, next_goto[0]['statement'].tag, next_goto[0]['statement'].attrib)
                goto_stmt = ET.SubElement(statement, next_goto[0]['goto-stmt'].tag, next_goto[0]['goto-stmt'].attrib)
                if reconstructed_if_elem and reconstructed_if_elem[0].attrib['label'] == goto_stmt.attrib['target_label']:
                    for stmt in reconstructed_if_elem:
                        self.statements_to_reconstruct_before['stmts-follow-label'].append(stmt)
                    self.statements_to_reconstruct_before['stmts-follow-label'].append(statement)
                    self.statements_to_reconstruct_before['count-gotos'] += 1
                    self.reconstruct_before_case_now = True
                    self.reconstruction_for_before_done = False
                else:
                    pass

        # Set all holders and checkers (markers) to default
        self.label_after = False
        self.reconstruct_after_case_now = False
        self.reconstruction_for_after_done = True
        self.goto_target_lbl_after.clear()
        self.label_lbl_for_after.clear()
        self.statements_to_reconstruct_after.clear()

    def reconstruct_goto_before_label(self, parent, traverse):

        stmts_follow_label = self.statements_to_reconstruct_before['stmts-follow-label']
        number_of_gotos = self.statements_to_reconstruct_before['count-gotos']
        
        declared_label_flag_num = []
        self.generate_declaration_element(parent, "label_flag", number_of_gotos, declared_label_flag_num)

        # A process for find the scope from label to goto,
        # then remove any statements that are not in the scope
        index = 0
        goto_counter = 0
        goto_index_holder = []
        target_label_lbl = None
        for stmt in stmts_follow_label:
            if (
                index == 0
                and "label" in stmt.attrib
            ):
                target_label_lbl = stmt.attrib['label']
            for child in stmt:
                if (
                    child.tag == "goto-stmt"
                    and child.attrib['target_label'] == target_label_lbl
                ):
                    goto_counter += 1
                    goto_index_holder.append(index)
            if goto_counter == number_of_gotos:
                break
            index += 1

        statements_to_recover = stmts_follow_label[index+1:len(stmts_follow_label)]
        del stmts_follow_label[index+1:len(stmts_follow_label)]

        # In case of multiple goto statements appears, slice them into N number of list objects
        # The location of goto statement (inner to outer) is represented by the increament of index
        # i.e. [0]: innermost, [N]: Outermost
        multiple_goto_stmts = []
        for i in range(len(goto_index_holder)):
            if i == 0:
                multiple_goto_stmts.append(stmts_follow_label[0:goto_index_holder[i]+1])
            else:
                if i+1 < len(goto_index_holder):
                    multiple_goto_stmts.append(stmts_follow_label[goto_index_holder[i-1]+1:goto_index_holder[i+1]+1])
                else:
                    multiple_goto_stmts.append(stmts_follow_label[goto_index_holder[i-1]+1:goto_index_holder[-1]+1])

        # Check whether there is inner label_after
        # case goto statements. Handles one case
        # at a time
        inner_gotos_exist = False
        labels = []
        index_scope = []
        for goto in multiple_goto_stmts:
            index = 0
            main_loop_lbl = goto[0].attrib['label']
            label_after_lbl = None
            for stmt in goto:
                if "label" in stmt.attrib:
                    labels.append(stmt.attrib["label"])
                    if stmt.attrib["label"] == label_after_lbl:
                        index_scope.append(index)
                if "goto-stmt" in stmt.attrib:
                    if (
                        main_loop_lbl != stmt.attrib['lbl']
                        and stmt.attrib['lbl'] not in labels
                    ):
                        inner_gotos_exist = True
                        label_after_lbl = stmt.attrib['lbl']
                        index_scope.append(index)
                index += 1

        # Generate loop ast
        cur_elem_parent = parent
        current_goto_num = 1
        end_of_current_goto_loop = False
        for i in range(number_of_gotos):
            loop_elem = ET.SubElement(cur_elem_parent, "loop", {"type":"do-while"})

            header_elem = ET.SubElement(loop_elem, "header")
            # The outermost flag == N and the innermost flag == 1
            flag_num = declared_label_flag_num[i]
            name = f"label_flag_{str(flag_num)}"
            name_attrib = {
                            "hasSubscripts":"false",
                            "id":name,
                            "type":"ambiguous",
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
                            if inner_gotos_exist:
                                reconstruct_target['stmts-follow-goto'] = statements[index_scope[0]:index_scope[1]]
                                reconstruct_target['stmts-follow-label'] = statements[index_scope[1]]
                                reconstruct_target['count-gotos'] = 1

                                self.reconstruct_goto_after_label(body_elem, traverse, reconstruct_target)

                                self.statements_to_reconstruct_after['stmts-follow-goto'] = []
                                self.statements_to_reconstruct_after['stmts-follow-label'] = stmts_follow_label_copy
                                inner_gotos_exist = False
                            else:
                                elems = ET.SubElement(body_elem, stmt.tag, stmt.attrib)
                                for child in stmt:
                                    if (
                                        child.tag == "goto-stmt"
                                        and target_label_lbl == child.attrib['target_label']
                                    ):
                                        # Unconditional
                                        self.generate_assignment_element(elems, flag_name, None, "literal", "true")
                                        end_of_current_goto_loop = True
                                    else:
                                        child_elem = ET.SubElement(elems, child.tag, child.attrib)
                                        if len(child) > 0:
                                            self.parseXMLTree(child, child_elem, elems, traverse)
                        # If end_of_current_goto_loop is True,
                        # escape one loop out to continue
                        # construct statements
                        if end_of_current_goto_loop:
                            body_elem = grand_parent_elem
                            end_of_current_goto_loop = False
                            flag_name = f"label_flag_{str(number_of_gotos+i-1)}"
                    index += 1
            else:
                current_goto_num += 1

        for recover_stmt in statements_to_recover:
            statement = ET.SubElement(parent, recover_stmt.tag, recover_stmt.attrib)
            for child in recover_stmt:
                child_elem = ET.SubElement(statement, child.tag, child.attrib)
                if len(child) > 0:
                    self.parseXMLTree(child, child_elem, statement, traverse)

        # Set all holders and checkers (markers) to default
        self.label_before = False
        self.reconstruct_before_case_now = False
        self.label_lbl_for_before.clear()
        self.statements_to_reconstruct_before['stmts-follow-label'] = []
        self.statements_to_reconstruct_before['count-gotos'] = 0

    #################################################################
    #                                                               #
    #                       ELEMENT GENERATORS                      #
    #                                                               #
    #################################################################

    def generate_declaration_element(self, parent, default_name, number_of_gotos, declared_flag_num):
        """
            A flag declaration and assginment xml generation.
            This will generate N number of label_flag_i or goto_i,
            where N is the number of gotos in the Fortran code
            and i is the number assgined to the flag
        """

        # Declaration
        specification_attribs = {"declaration":"1", "implicit":"1", "imports":"0", "uses":"0"}
        specification_elem = ET.SubElement(parent, "specification", specification_attribs)
        declaration_elem = ET.SubElement(specification_elem, "declaration", {"type":"variable"})
        type_attribs = {
                        "hasKind":"false", 
                        "hasLength":"false",
                        "is_derived_type":"False",
                        "keyword2":"none",
                        "name":"logical",
        }
        type_elem = ET.SubElement(declaration_elem, "type", type_attribs)
        variables_elem = ET.SubElement(declaration_elem, "variables", {"count":str(number_of_gotos)})
        variable_attribs = {
                            "hasArraySpec":"false",
                            "hasCharLength":"false",
                            "hasCoarraySpec":"false",
                            "hasInitialValue":"false",
                            "hasInitialization":"false",
                            "is_array":"false",
        }
        for flag in range(number_of_gotos):
            flag_num = flag+1
            if default_name == "label_flag":
                if flag_num in self.declared_label_flags:
                    flag_num = self.declared_label_flags[-1]+1
            if default_name == "goto_flag":
                if flag_num in self.declared_goto_flags:
                    flag_num = self.declared_goto_flags[-1]+1
            self.declared_label_flags.append(flag_num)
            declared_flag_num.append(flag_num)
            variable_attribs['id'] = f"{default_name}_{flag_num}"
            variable_attribs['name'] = f"{default_name}_{flag_num}"
            variable_elem = ET.SubElement(variables_elem, "variable", variable_attribs)

        # Assignment
        for flag in range(number_of_gotos):
            flag_num = declared_flag_num[flag]
            declared_flag_num.append(flag_num)
            statement_elem = ET.SubElement(parent, "statement")
            self.generate_assignment_element(statement_elem, f"{default_name}_{flag_num}", None, "literal", "true")

    def generate_assignment_element(self, parent, name_id, condition, value_type, value):
        """
            This is a function for generating new assignment element xml
            for goto reconstruction.
        """
        assignment_elem = ET.SubElement(parent, "assignment")
        target_elem = ET.SubElement(assignment_elem, "target")

        self.generate_name_element(target_elem, "false", name_id, "false", "1", "variable")

        value_elem = ET.SubElement(assignment_elem, "value")
        # Unconditional goto has default values of literal as below
        if value_type == "literal":
            literal_elem = ET.SubElement(value_elem, "literal", {"type":"bool", "value":value})
        # Conditional goto has dynamic values of operation
        else:
            pass

    def generate_operation_element(self, parent, op_type, operator, name):
        """
            This is a function for generating new operation element and
            its nested subelements with the passes arguments.

            Currently, it generates only a unary operation syntax only.
            It may require update in the future.
        """
        operation_elem = ET.SubElement(parent, "operation", {"type":op_type})
        operator_elem = ET.SubElement(operation_elem, "operator", {"operator":operator})
        operand_elem = ET.SubElement(operation_elem, "operand")

        self.generate_name_element(operand_elem, "false", name, "false", "1", "ambiguous")

    def generate_name_element(self, parent, hasSubscripts, name_id, is_array, numPartRef, name_type): 
        """
            This is a function for generating new name element based on
            the provided arguments.
        """
        name_attribs = {
                        "hasSubscripts":hasSubscripts,
                        "id":name_id,
                        "is_array":is_array,
                        "numPartRef":numPartRef,
                        "type":name_type,
        }
        name_elem = ET.SubElement(parent, "name", name_attribs)

    def generate_if_element(self, parent, stored_stmts, next_goto, need_operation, op_type, lhs, rhs, operator, traverse, reconstructed_if_elem):
        """
            This is a function generating new if element.
            Since header can hold unary, multiary, or name, some arguments
            may be passed with None. Check them to generate an appropriate XML.
        """
        goto_nest_if_elem = ET.SubElement(parent, "if")

        header_elem = ET.SubElement(goto_nest_if_elem, "header")

        if need_operation:
            self.generate_operation_element(header_elem, op_type, operator, lhs)
        else:
            self.generate_name_element(header_elem, "false", lhs, "false", "1", "variable")

        # Generate AST for statements that will be nested under if (!cond) or (cond)
        label = None
        statement_num = 0
        label_before_within_scope = False
        body_elem = ET.SubElement(goto_nest_if_elem, "body")
        for stmt in stored_stmts:
            if len(stmt) > 0 and "skip-collect" not in stmt.attrib:
                if (
                    "has-goto-stmt" not in stmt.attrib
                    or "lbl" in stmt.attrib
                    and stmt.attrib['lbl'] == self.reconstructing_scope
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
                            self.reconstructing_scope = stmt.attrib['label']
                    if label_before_within_scope:
                        # del stmt.attrib['goto-remove']
                        self.statements_to_reconstruct_before['stmts-follow-label'].append(stmt)
                        for child in stmt:
                            if child.tag == "label":
                                label = child.attrib['lbl']
                                if self.reconstructing_scope == label:
                                    if self.encapsulate_under_do_while:
                                        goto_nest_if_elem.attrib['label'] = label
                            if child.tag == "goto-stmt":
                                # If current goto-stmt label is equal to the scope label,
                                # it means that end-of-scope is met and ready to reconstruct
                                if self.reconstructing_scope == child.attrib['target_label']:
                                    self.statements_to_reconstruct_before['count-gotos'] += 1
                                    # Since we are going to handle the first label-before
                                    # case, remove the label lbl from the list
                                    del self.goto_target_lbl_before[0]
                                    label_before_within_scope = False
                                    self.reconstructing_scope = None
                                    self.reconstruct_goto_before_label(body_elem, traverse)

                                # Else, a new goto-stmt was found that is nested current label_before
                                # case scope, so we need to update the parent for it
                                else:
                                    stmt.attrib['parent-scope'] = self.reconstructing_scope
                    else:
                        cur_elem = ET.SubElement(body_elem, stmt.tag, stmt.attrib)
                        if "goto-remove" in cur_elem.attrib:
                            del cur_elem.attrib['goto-remove']
                        for child in stmt:
                            child_elem = ET.SubElement(cur_elem, child.tag, child.attrib)
                            if len(child) > 0:
                                self.parseXMLTree(child, child_elem, cur_elem, traverse)
                else:
                    goto_stmt = {}
                    goto_stmt['statement'] = stmt
                    for child in stmt:
                        assert (
                                child.tag == "goto-stmt"
                        ), f"Must only store <goto-stmt> in next_goto['goto-stmt']. Current: <{child.tag}>."
                        if child.attrib['target_label'] in self.goto_target_lbl_before:
                            goto_stmt['statement'].attrib['goto-move'] = "true"
                        if (
                            child.attrib['target_label'] not in self.goto_target_lbl_after
                            and "goto-move" in goto_stmt['statement']
                        ):
                           del goto_stmt['statement'].attrib['goto-move']
                        goto_stmt['goto-stmt'] = child
                    next_goto.append(goto_stmt)
                statement_num += 1

        if self.encapsulate_under_do_while:
            goto_nest_if_elem.attrib['goto-move'] = "true"
            reconstructed_if_elem.append(goto_nest_if_elem)
            parent.remove(goto_nest_if_elem)

        # Unconditional goto sets goto_flag always to false when it enters 2nd if-statement
        if not need_operation:
            statement = ET.SubElement(body_elem, "statement")
            self.generate_assignment_element(statement, lhs, None, "literal", "false")

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
        return re.findall(r"\"([^\']+)\"", unrefined_id)[0]

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
            cur_elem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, cur_elem, newRoot, traverse)

    tree = ET.ElementTree(newRoot)
    indent(newRoot)
    
    # Checks if the rectified AST requires goto elimination,
    # if it does, it does a 2nd traverse to eliminate and
    # reconstruct the AST once more
    while (XMLCreator.need_goto_elimination):
        traverse += 1
        # REMOVE
        print (f"{traverse} Traverse Begin")
        root = tree.getroot()	
        newRoot = ET.Element(root.tag, root.attrib)
        for child in root:	
            if child.text:	
                cur_elem = ET.SubElement(newRoot, child.tag, child.attrib)	
                XMLCreator.parseXMLTree(child, cur_elem, newRoot, traverse)	
        tree = ET.ElementTree(newRoot)	
        indent(newRoot)
        if not XMLCreator.continue_elimination:
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
            cur_elem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, cur_elem, newRoot, traverse)

    return newRoot


if __name__ == "__main__":
    filename = sys.argv[1]
    # Build a new cleaned AST XML
    buildNewAST(filename)
