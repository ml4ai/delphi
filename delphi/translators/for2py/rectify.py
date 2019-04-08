"""
    The purpose of this program is to do all the clean up for transplate.py.
    This (rectify.py) program will receive OFP generated XML file as an input.
    Then, it removes any unnecessary elementss and refactor randomly structured
    (nested) elementss into a correct structure. The output file will be apprx.
    30%~40% lighter in terms of number of lines than the OFP XML.
  
    Example:
        This script is executed by the autoTranslate script as one
        of the steps in converted a Fortran source file to Python
        file. For standalone execution:::

            $python rectify.py <ast_file>

    ast_file: The XML represenatation of the AST of the Fortran file. This is
    produced by the OpenFortranParser.
"""

import sys
import re
import xml.etree.ElementTree as ET

class RectifyOFPXML:
    def __init__(self):
        self.is_derived_type = False
        self.is_array = False

        self.cur_derived_type_name = ""
        self.derived_type_var_holder_list = []
        self.parent_type = ET.Element('')

    """
        Since there are many children tags under 'statement',
        I'm creating a separate tag list to check in the handle_tag_statement function.
    """
    STATEMENT_CHILD_TAGS = {
        "assignment", "write", "format", "stop",
        "execution-part", "print", "open", "read",
        "close", "call", "statement", "label",
        "literal", "continue-stmt", "do-term-action-stmt",
        "return", "contains-stmt", "declaration", "prefix",
        "function", "internal-subprogram", "internal-subprogram-part",
        "prefix",
    }

    DERIVED_TYPE_TAGS = {
        "declaration-type-spec", "type-param-or-comp-def-stmt-list",
        "component-decl-list__begin", "component-initialization",
        "data-component-def-stmt", "component-def-stmt", "component-attr-spec-list",
        "component-attr-spec-list__begin", "explicit-shape-spec-list__begin",
        "explicit-shape-spec", "explicit-shape-spec-list", "component-attr-spec",
        "component-attr-spec-list__begin", "component-shape-spec-list__begin",
        "explicit-shape-spec-list__begin", "explicit-shape-spec", "component-attr-spec",
        "component-attr-spec-list",
    }

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
    def handle_tag_file(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "program" or child.tag == "subroutine" or child.tag == "module":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_file: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the program elementss.
        <program>
            ...
        </program>
    """
    def handle_tag_program(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "header" or child.tag == "body":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "end-program-stmt" and child.tag != "main-program":
                    print (f'In handle_tag_program: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the header elementss.
        <header>
            ...
        </header>
    """
    def handle_tag_header(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "index-variable" or child.tag == "operation" or child.tag == "arguments" or child.tag == "names":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_header: "{child.tag}" not handled')
            else:
                if child.tag == "subroutine-stmt":
                    parElem.attrib.update(child.attrib)
                elif child.tag == "loop-control" or child.tag == "label":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                else:
                    print (f'In handle_tag_header: Empty elements  "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the body elementss.
        <body>
            ...
        </body>
    """
    def handle_tag_body(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "specification" or child.tag == "statement" or child.tag == "loop" or child.tag == "if":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_body: "{child.tag}" not handled')
            else:
                if child.tag == "label" or child.tag == "do-term-action-stmt":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                elif child.tag != "statement":
                    print (f'In handle_tag_body: Empty elements  "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the specification elementss.
        <specification>
            ...
        </specification>
    """
    def handle_tag_specification(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "declaration" or child.tag == "use":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_specification: "{child.tag}" not handled')
            else:
                if child.tag != "declaration":
                    print (f'In handle_tag_specification: Empty elements "{child.tag}" not handled')
                

    """
        This function handles cleaning up the XML elementss between the declaration elementss.
        <declaration>
            ...
        </declaration>
    """
    def handle_tag_declaration(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "type" or child.tag == "dimensions" or child.tag == "variables" or child.tag == "format" or child.tag == "name":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    if child.tag == "dimensions":
                        self.is_array = True
                    self.parseXMLTree(child, curElem)
                elif child.tag == "component-array-spec" or child.tag == "literal":
                    self.derived_type_var_holder_list.append(child)
                else:
                    print (f'In handle_tag_declaration: "{child.tag}" not handled')
            else:
                if child.tag == "type-declaration-stmt" or child.tag == "prefix-spec" or child.tag == "save-stmt" or child.tag == "access-spec"\
                   or child.tag == "attr-spec" or child.tag == "access-stmt" or child.tag == "access-id-list":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                elif child.tag == "component-decl" or child.tag == "component-decl-list":
                    self.derived_type_var_holder_list.append(child)
                elif child.tag == "component-array-spec":
                    self.derived_type_var_holder_list.append(child)
                else:
                    if child.tag != "attr-spec" and child.tag != "access-id":
                        print (f'In handle_tag_declaration: Empty elements "{child.tag}" not handled')

        if self.is_array == True:
            self.is_array = False

        # If is_derived_type is true, reconstruct the derived type declaration AST structure
        if self.is_derived_type:
            # Modify or add 'name' attribute of the MAIN (or the outer most) <type> elements with the name of derived type name
            self.parent_type.set('name', self.cur_derived_type_name)
            self.reconstruct_derived_type_declaration()

    """
        This function handles cleaning up the XML elementss between the variables elementss.
        <type>
            ...
        </type>
    """
    def handle_tag_type(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            """
                Having a nested "type" indicates that this is a "derived type" declaration.
                In other word, this is a case of
                <type>
                    <type>
                        ...
                    </type>
                </type>
            """
            if child.tag == "type":
                self.is_derived_type = True
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                self.parent_type = parElem
                self.parseXMLTree(child, curElem)
            elif child.tag == "intrinsic-type-spec":
                if self.is_derived_type:
                    self.derived_type_var_holder_list.append(child)
                parElem.attrib['keyword2'] = child.attrib['keyword2']
            elif child.tag == "derived-type-stmt" and self.is_derived_type:
                # Modify or add 'name' attribute of the <type> elements with the name of derived type name
                parElem.set('name', child.attrib['id'])
                # And, store the name of the derived type name for later setting the outer most <type> elements's name attribute
                self.cur_derived_type_name = child.attrib['id'];
                # curElem = ET.SubElement(parElem, child.tag, child.attrib)
            elif child.tag == "derived-type-spec":
                if not self.is_derived_type:
                    parElem.set('name', child.attrib['typeName'])
                else:
                    self.derived_type_var_holder_list.append(child)
            elif child.tag == "literal":
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "component-array-spec":
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "component-decl" or child.tag == "component-decl-list":
                self.derived_type_var_holder_list.append(child)
            elif child.tag == "length":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                self.parseXMLTree(child, curElem)
            else:
                if child.tag not in self.DERIVED_TYPE_TAGS and child.tag != "char-selector" and child.tag != "delcaration-type-spec":
                    print (f'In handle_tag_type: "{child.tag}" not handled')
        # This will mark whether this type declaration is for a derived type declaration or not
        parElem.set('is_derived_type', str(self.is_derived_type))

    """
        This function handles cleaning up the XML elementss between the variables elementss.
        <variables>
            ...
        </variables>
    """
    def handle_tag_variables(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                # Up to this point, all the child (nested or sub) elementss were <variable>
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                curElem.set('is_array', str(self.is_array))
                self.parseXMLTree(child, curElem)

    """
        This function handles cleaning up the XML elementss between the variables elementss.
        <variable>
            ...
        </variable>
    """
    def handle_tag_variable(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "initial-value":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_variable: "{child.tag}" not handled')
            else:
                if child.tag == "entity-decl":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_variable: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the statement elementss.
        <statement>
            ...
        </statement>
    """
    def handle_tag_statement(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag in self.STATEMENT_CHILD_TAGS:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem) 
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
                self.parseXMLTree(child, parElem)
            else:
                print (f'In self.handle_tag_statement: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the assignment elementss.
        <assignment>
            ...
        </assignment>
    """
    def handle_tag_assignment(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "target" or child.tag == "value":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_assignment: "{child.tag}" not handled')
     
    """
        This function handles cleaning up the XML elementss between the target elementss.
        <target>
            ...
        </target>
    """
    def handle_tag_target(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_target: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the names and/or name elementss.
        <names>         <name>
            ...     or     ...
        </names>        <name>
    """
    def handle_tag_name(self, root, parElem):
        if 'id' in parElem.attrib and '%' in parElem.attrib['id']:
            self.clean_derived_type_ref(parElem)

        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "subscripts" or child.tag == "assignment":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                elif child.tag == "output":
                    assert is_empty(self.derived_type_var_holder_list)
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.derived_type_var_holder_list.append(root.attrib['id'])
                    self.parseXMLTree(child, curElem)
                elif child.tag == "name":
                    self.parseXMLTree(child, parElem)
                else:
                    print (f'In self.handle_tag_name: "{child.tag}" not handled')
            else:
                if child.tag == "name" or child.tag == "generic_spec":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                elif child.tag == "data-ref":
                    parElem.attrib.update(child.attrib)
                elif child.tag != "designator":
                    print (f'In self.handle_tag_name: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the value elementss.
        <value>
            ...
        </value>
    """
    def handle_tag_value(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "literal" or child.tag == "operation" or child.tag == "name":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_value: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the literal elementss.
        <literal>
            ...
        </literal>
    """
    def handle_tag_literal(self, root, parElem):
        if '"' in parElem.attrib['value']:
            parElem.attrib['value'] = self.clean_id(parElem.attrib['value'])
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "stop":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_literal: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the dimensions elementss.
        <dimensions>
            ...
        </dimensions>
    """
    def handle_tag_dimensions(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "dimension":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_dimensions: "{child.tag}" not handled')
                
    """
        This function handles cleaning up the XML elementss between the dimension elementss.
        <dimension>
            ...
        </dimension>
    """
    def handle_tag_dimension(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "range":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_dimension: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the do loop elementss.
        <loop>
            ...
        </loop>
    """
    def handle_tag_loop(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "header" or child.tag == "body" or child.tag == "format":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_loop: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the index_variable or range elementss.
        <index_variable>                        <range>
            ...                 or                  ...
        </index_variable>                       </range>
    """      
    def handle_tag_index_variable_or_range(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "lower-bound" or child.tag == "upper-bound" or child.tag == "step":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_index_variable_or_range: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the lower_bound elementss.
        <lower_bound>
            ...
        </lower_bound>
    """
    def handle_tag_lower_bound(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_lower_bound: "{child.tag}" not handled')
                
    """
        This function handles cleaning up the XML elementss between the upper_bound elementss.
        <upper_bound>
            ...
        </upper_bound>
    """
    def handle_tag_upper_bound(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "name" or child.tag == "operation":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_upper_bound: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the subscripts elementss.
        <supscripts>
            ...
        </supscripts>
    """
    def handle_tag_subscripts(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "subscript":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_subscripts: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the subscript elementss.
        <supscript>
            ...
        </supscript>
    """
    def handle_tag_subscript(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "name" or child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_subscript: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the operation elementss.
        <operation>
            ...
        </operation>
    """
    def handle_tag_operation(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "operand":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "operator":
                    print (f'In handle_tag_operation: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the operation elementss.
        <operand>
            ...
        </operand>
    """
    def handle_tag_operand(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name" or child.tag == "literal" or child.tag == "operation":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_operand: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the operation elementss.
        <operand>
            ...
        </operand>
    """
    def handle_tag_write(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "io-controls" or child.tag == "outputs":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_write: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the io-controls elementss.
        <io-controls>
            ...
        </io-controls>
    """
    def handle_tag_io_controls(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "io-control":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_io_controls: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the io-control elementss.
        <io-control>
            ...
        </io-control>
    """
    def handle_tag_io_control(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            # To make io-control elements simpler, the code below will append io-control-spec's attributes
            # to its parent (io-control). This will eliminate at least one recursion in translate.py to
            # retrieve the io-control information
            if child.tag == "io-control-spec":
                parElem.attrib.update(child.attrib)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "io-control" or child.tag == "literal":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_io_control: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the outputs elementss
        <outputs>
            ...
        </outputs>
    """
    def handle_tag_outputs(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "output":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                self.parseXMLTree(child, curElem)
            elif child.tag == "name":
                self.parseXMLTree(child, root)
            else:
                print (f'In handle_tag_outputs: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the output elementss.
        <output>
            ...
        </output>
    """
    def handle_tag_output(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name" or child.tag == "literal":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_outputs: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the format elementss.
        <format>
            ...
        </format>
    """
    def handle_tag_format(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "format-items":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "label":
                    print (f'In handle_tag_format: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the format_items and its sub-elementss
        <format_items>
            <format_item>
                ...
            </format_item>
            ...
        </format_items>
    """
    def handle_tag_format_items(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "format-items" or child.tag == "format-item":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_format_items: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the print tags.
        <print>
            ...
        </print>
    """
    def handle_tag_print(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag != "print-stmt":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "outputs":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "print-format" and child.tag != "print-stmt":
                    print (f'In handle_tag_print: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the open elementss.
        <open>
            ...
        </open>
    """
    def handle_tag_open(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments": 
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_open: "{child.tag}" not handled')
            else:
                if child.tag == "open-stmt":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_open: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the keyword-arguments and keyword-argument elementss.
        <keyword-arguments>
            <keyword-argument>
                ...
            </keyword-argument>
            ...
        </keyword-arguments>
    """
    def handle_tag_keyword_arguments(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-argument" or child.tag == "literal":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_keyword_arguments - {root.tag}: "{child.tag}" not handled')
            else:
                if child.tag != "keyword-argument":
                    print (f'In handle_tag_keyword_arguments - {root.tag}: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the read elementss. 
        <read>
            ...
        </read>
    """
    def handle_tag_read(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "io-controls" or child.tag == "inputs":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_read: "{child.tag}" not handled')
            else:
                if child.tag == "read-stmt":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_read: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the inputs and input elementss.
        <inputs>
            <input>
                ...
            </input>
            ...
        </inputs>
    """
    def handle_tag_inputs(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "input" or child.tag == "name":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_input - {root.tag}: "{child.tag}" not handled')
            else:
                print (f'In handle_tag_input - {root.tag}: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the close elementss. 
        <close>
            ...
        </close>
    """
    def handle_tag_close(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "keyword-arguments":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_close: "{child.tag}" not handled')
            else:
                if child.tag == "close-stmt":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_close: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the call elements. 
        <call>
            ...
        </call>
    """
    def handle_tag_call(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "name":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_call: "{child.tag}" not handled')
            else:
                if child.tag == "call-stmt":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_call: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the subroutine elements. 
        <subroutine>
            ...
        </subroutine>
    """
    def handle_tag_subroutine(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_subroutine: "{child.tag}" not handled')
            else:
                if child.tag != "end-subroutine-stmt":
                    print (f'In handle_tag_subroutine: Empty elements "{child.tag}" not handled')


    """
        This function handles cleaning up the XML elementss between the arguments. 
        <arguments>
            ...
        </arsuments>
    """
    def handle_tag_arguments(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "argument":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
            else:
                print (f'In handle_tag_variable: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the if elements.
        <if>
            ...
        </if>
    """
    def handle_tag_if(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_if: "{child.tag}" not handled')
            else:
                if child.tag == "if-stmt":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_if: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the stop elements 
        <stop>
            ...
        </stop>
    """
    def handle_tag_stop(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "stop-code":
                parElem.attrib.update(child.attrib)
            else:
                print (f'In handle_tag_stop: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the step elements.
        <step>
            ...
        </step>
    """
    def handle_tag_step(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "operation" or child.tag == "literal":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_step: "{child.tag}" not handled')
            else:
                print (f'In handle_tag_step: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the return and return-stmt elementss.
        However, since 'return-stmt' is an empty elements with no sub-elementss, the function will not keep
        the elements, but move the attribute to its parent elements, return.
        <return>
            ...
        </return>
    """
    def handle_tag_return(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "return-stmt":
                parElem.attrib.update(child.attrib)
            else:
                print (f'In handle_tag_return: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the function elements.
        <function>
            ...
        </function>
    """
    def handle_tag_function(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                if child.tag == "header" or child.tag == "body":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_function: "{child.tag}" not handled')
            else:
                if child.tag == "function-stmt" or child.tag == "end-function-stmt" or child.tag == "function-subprogram":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                else:
                    print (f'In handle_tag_function: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the use elementss.
        <use>
            ...
        </use>
    """
    def handle_tag_use(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "use-stmt" or child.tag == "only":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_use: "{child.tag}" not handled')


    """
        This function handles cleaning up the XML elementss between the module elementss.
        <module>
            ...
        </module>
    """
    def handle_tag_module(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "header" or child.tag == "body" or child.tag == "module-stmt" or child.tag == "members"\
               or child.tag == "end-module-stmt" or child.tag == "contains-stmt":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_module: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the initial-value elementss.
        <initial-value>
            ...
        </initial-value>
    """
    def handle_tag_initial_value(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_initial_value: "{child.tag}" not handled')
            else:
                if child.tag == "initialization":
                    parElem.attrib.update(child.attrib)
                else:
                    print (f'In handle_tag_initial_value: Empty elements "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the members elementss.
        <members>       <member>
            ...     or      ...
        </members>      </member>
    """
    def handle_tag_members(self, root, parElem):
        for child in root:
            self.clean_attrib(child)
            if child.tag == "subroutine" or child.tag == "module-subprogram" or child.tag == "module-subprogram-part"\
               or child.tag == "declaration" or child.tag == "prefix" or child.tag == "function":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_members: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the only elementss.
        <only>
            ...
        </only>
    """
    def handle_tag_only(self, root, parElem):
        for child in root:
            if child.tag == "name" or child.tag == "only" or child.tag == "only-list":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_only: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elementss between the length elementss.
        <length>
            ...
        </length>
    """
    def handle_tag_length(self, root, parElem):
        for child in root:
            if child.tag == "literal" or child.tag == "char-length":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_length: "{child.tag}" not handled')

    """
        parseXMLTree

        Arguments:
            root: The current root of the tree.
            parElem: The parent elements.

        Returns:
            None
            
        Recursively traverse through the nested XML AST tree and calls appropriate tag handler, which will generate
        a cleaned version of XML tree for translate.py. Any new tags handlers must be added under this this function.
    """
    def parseXMLTree(self, root, parElem):
        if root.tag == "file":
            self.handle_tag_file(root, parElem)
        elif root.tag == "program":
            self.handle_tag_program(root, parElem)
        elif root.tag == "header":
            self.handle_tag_header(root, parElem)
        elif root.tag == "specification":
            self.handle_tag_specification(root, parElem)
        elif root.tag == "body":
            self.handle_tag_body(root, parElem)
        elif root.tag == "declaration":
            self.handle_tag_declaration(root, parElem)
        elif root.tag == "type":
            self.handle_tag_type(root, parElem)
        elif root.tag == "variables":
            self.handle_tag_variables(root, parElem)
        elif root.tag == "variable":
            self.handle_tag_variable(root, parElem)
        elif root.tag == "statement":
            self.handle_tag_statement(root, parElem)
        elif root.tag == "assignment":
            self.handle_tag_assignment(root, parElem)
        elif root.tag == "target":
            self.handle_tag_target(root, parElem)
        elif root.tag == "value":
            self.handle_tag_value(root, parElem)
        elif root.tag == "names" or root.tag == "name":
            self.handle_tag_name(root, parElem)
        elif root.tag == "literal":
            self.handle_tag_literal(root, parElem)
        elif root.tag == "dimensions":
            self.handle_tag_dimensions(root, parElem)
        elif root.tag == "dimension":
            self.handle_tag_dimension(root, parElem)
        elif root.tag == "loop":
            self.handle_tag_loop(root, parElem)
        elif root.tag == "index-variable" or root.tag == "range":
            self.handle_tag_index_variable_or_range(root, parElem)
        elif root.tag == "lower-bound":
            self.handle_tag_lower_bound(root, parElem)
        elif root.tag == "upper-bound":
            self.handle_tag_upper_bound(root, parElem)
        elif root.tag == "subscripts":
            self.handle_tag_subscripts(root, parElem)
        elif root.tag == "subscript":
            self.handle_tag_subscript(root, parElem)
        elif root.tag == "operation":
            self.handle_tag_operation(root, parElem)
        elif root.tag == "operand":
            self.handle_tag_operand(root, parElem)
        elif root.tag == "write":
            self.handle_tag_write(root, parElem)
        elif root.tag == "io-controls":
            self.handle_tag_io_controls(root, parElem)
        elif root.tag == "io-control":
            self.handle_tag_io_control(root, parElem)
        elif root.tag == "outputs":
            self.handle_tag_outputs(root, parElem)
        elif root.tag == "output":
            self.handle_tag_output(root, parElem)
        elif root.tag == "format":
            self.handle_tag_format(root, parElem)
        elif root.tag == "format-items" or root.tag == "format-item":
            self.handle_tag_format_items(root, parElem)
        elif root.tag == "print":
            self.handle_tag_print(root, parElem)
        elif root.tag == "open":
            self.handle_tag_open(root, parElem)
        elif root.tag == "keyword-arguments" or root.tag == "keyword-argument":
            self.handle_tag_keyword_arguments(root, parElem)
        elif root.tag == "read":
            self.handle_tag_read(root, parElem)
        elif root.tag == "inputs" or root.tag == "input":
            self.handle_tag_inputs(root, parElem)
        elif root.tag == "close":
            self.handle_tag_close(root, parElem)
        elif root.tag == "call":
            self.handle_tag_call(root, parElem)
        elif root.tag == "subroutine":
            self.handle_tag_subroutine(root, parElem)
        elif root.tag == "arguments":
            self.handle_tag_arguments(root, parElem)
        elif root.tag == "if":
            self.handle_tag_if(root, parElem)
        elif root.tag == "stop":
            self.handle_tag_stop(root, parElem)
        elif root.tag == "step":
            self.handle_tag_step(root, parElem)
        elif root.tag == "return":
            self.handle_tag_return(root, parElem)
        elif root.tag == "function":
            self.handle_tag_function(root, parElem)
        elif root.tag == "use":
            self.handle_tag_use(root, parElem)
        elif root.tag == "module":
            self.handle_tag_module(root, parElem)
        elif root.tag == "initial-value":
            self.handle_tag_initial_value(root, parElem)
        elif root.tag == "members":
            self.handle_tag_members(root, parElem)
        elif root.tag == "only":
            self.handle_tag_only(root, parElem)
        elif root.tag == "length":
            self.handle_tag_length(root, parElem)
        else:
            print (f"In the parseXMLTree and, currently, {root.tag} is not supported")

    """
        reconstruct_derived_type_declaratione reconstruct the derived type with the collected derived type declaration elementss
        in the handle_tag_declaration and handle_tag_type.
    """
    def reconstruct_derived_type_declaration(self):
        if self.derived_type_var_holder_list:
            literal = ET.Element('')
            is_dimension = False 
            
            # Since component-decl-list appears after component-decl, the program needs to iterate the list
            # once first to pre-collect the variable counts
            counts = []
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "component-decl-list":
                    counts.append(elem.attrib['count'])

            # Initialize count to 0 for <variables> count attribute
            count = 0
            # 'component-decl-list__begin' tag is an indication of all the derived type member variable declarations will follow
            derived_type = ET.SubElement(self.parent_type, 'component-decl-list__begin')
            for elem in self.derived_type_var_holder_list:
                if elem.tag == "intrinsic-type-spec":
                    attributes = {'hasKind': 'false', 'hasLength': 'false', 'name': elem.attrib['keyword1'], 'is_derived_type': str(self.is_derived_type)}
                    newType = ET.SubElement(derived_type, 'type', attributes)
                elif elem.tag == "literal":
                    literal = elem;
                elif elem.tag == "component-array-spec":
                    is_dimension = True
                elif elem.tag == "component-decl":
                    if not is_dimension:
                        if len(counts) > count:
                            attr = {'count': counts[count]}
                            new_variables = ET.SubElement(derived_type, 'variables', attr) # <variables _attribs_>
                            count += 1
                        var_attribs = {'has_initial_value': elem.attrib['hasComponentInitialization'], 'name': elem.attrib['id']}
                        new_variable = ET.SubElement(new_variables, 'variable', var_attribs) # <variable _attribs_>
                        if elem.attrib['hasComponentInitialization'] == "true":
                            init_value_attrib = ET.SubElement(new_variable, 'initial-value')
                            new_literal = ET.SubElement(init_value_attrib, 'literal', literal.attrib) # <initial-value _attribs_>
                    else:
                        new_dimensions = ET.SubElement(derived_type, 'dimensions', {'count': '1'}) # <dimensions count="1">
                        new_dimension = ET.SubElement(new_dimensions, 'dimension', {'type': 'simple'}) # <dimension type="simple">
                        new_literal = ET.SubElement(new_dimension, 'literal', literal.attrib) # <literal type="" value="">
                        if len(counts) > count:
                            attr = {'count': counts[count]}
                            new_variables = ET.SubElement(derived_type, 'variables', attr)
                            count += 1
                        var_attribs = {'has_initial_value': elem.attrib['hasComponentInitialization'], 'name': elem.attrib['id']}
                        new_variable = ET.SubElement(new_variables, 'variable', var_attribs)
                        is_dimension = False

            # Once one derived type was successfully constructed, clear all the elementss of a derived type list
            self.derived_type_var_holder_list.clear()
            self.is_derived_type = False

    """
        This function will clean up the derived type referencing syntax, which is stored in a form of "id='x'%y" in the id attribute.
        Once the id gets cleaned, it will call the reconstruc_derived_type_ref function to reconstruct and replace the messy version
        of id with the cleaned version.
    """
    def clean_derived_type_ref(self, parElem):
        current_id = parElem.attrib['id'] # 1. Get the original form of derived type id, which is in a form of, for example, id="x"%y in the original XML.
        self.derived_type_var_holder_list.append(self.clean_id(current_id)) # 2. Extract the first variable name, for example, x in this case.
        percent_sign = current_id.find("%") # 3. Get the location of the '%' sign
        self.derived_type_var_holder_list.append(current_id[percent_sign + 1 : len(current_id)]) # 4. Get the field variable. y in this example.
        self.reconstruct_derived_type_ref(parElem)

    """
        This function reconstruct the id into x.y.k form from the messy looking id.
        One thing to notice is that this new form was generated in the python syntax, so it is a pre-process for translate.py and even pyTranslate.py that
    """
    def reconstruct_derived_type_ref(self, parElem):
        num_of_vars = len(self.derived_type_var_holder_list)
        cleaned_id = ""
        for var in self.derived_type_var_holder_list:
            cleaned_id += var
            if num_of_vars > 1:
                cleaned_id += '.'
                num_of_vars -= 1
        parElem.attrib['id'] = cleaned_id
        self.derived_type_var_holder_list.clear() # Clean up the list for re-use

    """
        This function refines id (or value) with quotation makrs included by removing them and returns only the variable name.
        For example, from "OUTPUT" to OUTPUT and "x" to x. Thus, the id name will be modified as below:
            Unrefined id: id = ""OUTPUT""
            Refined id: id = "OUTPUT"
    """
    def clean_id(self, unrefined_id):
        return re.findall(r"\"([^\"]+)\"", unrefined_id)[0]

    """
        The original XML elements holds 'eos' and 'rule' attributes that are not necessary and being used.
        Thus, this function will remove them in the rectified version of XML.
    """
    def clean_attrib(self, elements):
        if "eos" in elements.attrib: 
            elements.attrib.pop("eos")
        if "rule" in elements.attrib:
            elements.attrib.pop("rule")

# ================================================================================================================================================================================
"""
    This function is just a helper function for check whether the passed elements (i.e. list) is empty or not
"""
def is_empty(elem):
    if not elem:
        return True
    else:
        return False

"""
    This function indents each level of XML.
    Source: https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementstree-in-python
"""
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    return elem

"""
    Using the list of cleaned AST, construct a new XML AST and write to a file
"""
def buildNewAST(root, filename):
    tree = ET.ElementTree(indent(root))
    rectFilename = filename.split('/')[-1]
    tree.write(f"rectified_{rectFilename}")

def main():
    filename = sys.argv[1]
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
            XMLCreator.parseXMLTree(child, curElem)
    # Build a new cleaned AST XML
    buildNewAST(newRoot, filename);

main()
# ================================================================================================================================================================================
