import sys
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom

# Read AST from the OFP generated XML file

class OFPXMLToCleanedXML:
    def __init__(self):
        self.is_derived_type = False
        self.hasInitialValue = False
        self.is_array = False

        self.curDerivedTypeName = ""
        self.derivedTypeList = []
        self.parentType = ET.Element('')

    """
        This function handles cleaning up the XML elements between the file elements.
        <file>
            ...
        </file>
    """
    def handle_tag_file(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "program":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_file: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the program elements.
        <program>
            ...
        </program>
    """
    def handle_tag_program(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "header" or child.tag == "body":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "end-program-stmt" and child.tag != "main-program":
                    print (f'In handle_tag_program: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the header elements.
        <header>
            ...
        </header>
    """
    def handle_tag_header(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "index-variable":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_header: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the body elements.
        <body>
            ...
        </body>
    """
    def handle_tag_body(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "specification" or child.tag == "statement" or child.tag == "loop":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_body: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the specification elements.
        <specification>
            ...
        </specification>
    """
    def handle_tag_specification(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "declaration":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_specification: "{child.tag}" not handled')
                

    """
        This function handles cleaning up the XML elements between the declaration elements.
        <declaration>
            ...
        </declaration>
    """
    def handle_tag_declaration(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "type" or child.tag == "dimensions" or child.tag == "variables":
                    if child.tag == "dimensions":
                        self.is_array = True
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_declaration: "{child.tag}" not handled')
            else:
                if child.tag == "component-decl" or child.tag == "component-decl-list":
                    self.derivedTypeList.append(child)

        if self.is_array == True:
            self.is_array = False

        if self.is_derived_type:
            # Modify or add 'name' attribute of the MAIN (or the outer most) <type> element with the name of derived type name
            self.parentType.set('name', self.curDerivedTypeName)
            self.reconDerivedType()

    """
        This function handles cleaning up the XML elements between the variables elements.
        <type>
            ...
        </type>
    """
    def handle_tag_type(self, root, parElem):
        for child in root:
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
                self.parentType = parElem
                self.parseXMLTree(child, curElem)
            elif child.tag == "intrinsic-type-spec":
                if self.is_derived_type:
                    self.derivedTypeList.append(child)
            elif child.tag == "derived-type-stmt" and self.is_derived_type:
                # Modify or add 'name' attribute of the <type> element with the name of derived type name
                parElem.set('name', child.attrib['id'])
                # And, store the name of the derived type name for later setting the outer most <type> element's name attribute
                self.curDerivedTypeName = child.attrib['id'];
                # curElem = ET.SubElement(parElem, child.tag, child.attrib)
            elif child.tag == "derived-type-spec":
                if not self.is_derived_type:
                    parElem.set('name', child.attrib['typeName'])
                else:
                    self.derivedTypeList.append(child)
            elif child.tag == "literal":
                self.hasInitialValue = True
                self.derivedTypeList.append(child)
            elif child.tag == "component-decl" or child.tag == "component-decl-list":
                self.derivedTypeList.append(child)
            else:
                if child.tag != "declaration-type-spec" and child.tag != "type-param-or-comp-def-stmt-list" and\
                   child.tag != "component-decl-list__begin" and child.tag != "component-initialization" and\
                   child.tag != "data-component-def-stmt" and child.tag != "component-def-stmt":
                    print (f'In handle_tag_type: "{child.tag}" not handled')
        # This will mark whether this type declaration is for a derived type declaration or not
        parElem.set('is_derived_type', str(self.is_derived_type))

    """
        This function handles cleaning up the XML elements between the variables elements.
        <variables>
            ...
        </variables>
    """
    def handle_tag_variables(self, root, parElem):
        for child in root:
            if child.text:
                # Up to this point, all the child (nested or sub) elements were <variable>
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                curElem.set('is_array', str(self.is_array))
                self.parseXMLTree(child, curElem)

    """
        This function handles cleaning up the XML elements between the variables elements.
        <variable>
            ...
        </variable>
    """
    def handle_tag_variable(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "initial-value":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_variable: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the statement elements.
        <statement>
            ...
        </statement>
    """
    def handle_tag_statement(self, root, parElem):
        for child in root:
            if child.tag == "assignment" or child.tag == "write" or child.tag == "format" or child.tag == "stop" or child.tag == "execution-part":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.text:
                    self.parseXMLTree(child, curElem) 
            elif child.tag == "name":
                """
                    If a 'name' tag is the direct sub-element of 'statement', it's an indication of
                    this statement is handling (usually assignment) derived type variables. Thus,
                    in order to make concurrent with other assignment syntax, remove the outside
                    name element (but store it to the temporary holder) and reconstruct it before
                    the end of statement
                """
                assert is_empty(self.derivedTypeList)
                self.derivedTypeList.append(child.attrib['id'])
                self.parseXMLTree(child, parElem)
            else:
                print (f'In self.handle_tag_statement: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the assignment elements.
        <assignment>
            ...
        </assignment>
    """
    def handle_tag_assignment(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "target" or child.tag == "value":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_assignment: "{child.tag}" not handled')
     
    """
        This function handles cleaning up the XML elements between the target elements.
        <target>
            ...
        </target>
    """
    def handle_tag_target(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_target: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the name elements.
        <name>
            ...
        </name>
    """
    def handle_tag_name(self, root, parElem):
        if 'id' in parElem.attrib and '%' in parElem.attrib['id']:
            self.clean_derived_type_ref(parElem)

        for child in root:
            if child.text:
                if child.tag == "subscripts" or child.tag == "assignment":
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.parseXMLTree(child, curElem)
                elif child.tag == "output":
                    assert is_empty(self.derivedTypeList)
                    curElem = ET.SubElement(parElem, child.tag, child.attrib)
                    self.derivedTypeList.append(root.attrib['id'])
                    self.parseXMLTree(child, curElem)
                elif child.tag == "name":
                    self.parseXMLTree(child, parElem)
                else:
                    print (f'In self.handle_tag_name: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the value elements.
        <value>
            ...
        </value>
    """
    def handle_tag_value(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "literal" or child.tag == "operation":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In self.handle_tag_value: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the literal elements.
        <literal>
            ...
        </literal>
    """
    def handle_tag_literal(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)

    """
        This function handles cleaning up the XML elements between the dimensions elements.
        <dimensions>
            ...
        </dimensions>
    """
    def handle_tag_dimensions(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "dimension":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_dimensions: "{child.tag}" not handled')
                
    """
        This function handles cleaning up the XML elements between the dimension elements.
        <dimension>
            ...
        </dimension>
    """
    def handle_tag_dimension(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "range":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_dimension: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the do loop elements.
        <loop>
            ...
        </loop>
    """
    def handle_tag_loop(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "header" or child.tag == "body":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_loop_do: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the index_variable or range elements.
        <index_variable>                        <range>
            ...                 or                  ...
        </index_variable>                       </range>
    """      
    def handle_tag_index_variable_or_range(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "lower-bound" or child.tag == "upper-bound":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_index_variable: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the lower_bound elements.
        <lower_bound>
            ...
        </lower_bound>
    """
    def handle_tag_lower_bound(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_lower_bound: "{child.tag}" not handled')
                
    """
        This function handles cleaning up the XML elements between the upper_bound elements.
        <upper_bound>
            ...
        </upper_bound>
    """
    def handle_tag_upper_bound(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "literal":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_upper_bound: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the subscripts elements.
        <supscripts>
            ...
        </supscripts>
    """
    def handle_tag_subscripts(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "subscript":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_subscripts: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the subscript elements.
        <supscript>
            ...
        </supscript>
    """
    def handle_tag_subscript(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "name" or child.tag == "literal" or child.tag == "operation":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In self.handle_tag_subscript: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the operation elements.
        <operation>
            ...
        </operation>
    """
    def handle_tag_operation(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "operand":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "operator":
                    print (f'In handle_tag_operation: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the operation elements.
        <operand>
            ...
        </operand>
    """
    def handle_tag_operand(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name" or child.tag == "literal" or child.tag == "operation":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_operand: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the operation elements.
        <operand>
            ...
        </operand>
    """
    def handle_tag_write(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "io-controls" or child.tag == "outputs":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_write: "{child.tag}" not handled')

    def handle_tag_io_controls(self, root, parElem):
        for child in root:
            if child.text:
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                if child.tag == "io-control":
                    self.parseXMLTree(child, curElem)
                else:
                    print (f'In handle_tag_io_controls: "{child.tag}" not handled')


    def handle_tag_io_control(self, root, parElem):
        for child in root:
            # To make io-control element simpler, the code below will append io-control-spec's attributes
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
        This function handles cleaning up the XML elements between the outputs.
        <outputs>
            ...
        </outputs>
    """
    def handle_tag_outputs(self, root, parElem):
        for child in root:
            if child.tag == "output":
                curElem = ET.SubElement(parElem, child.tag, child.attrib)
                self.parseXMLTree(child, curElem)
            elif child.tag == "name":
                self.parseXMLTree(child, root)
            else:
                print (f'In handle_tag_outputs: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the output.
        <output>
            ...
        </output>
    """
    def handle_tag_output(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "name":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_outputs: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the format.
        <format>
            ...
        </format>
    """
    def handle_tag_format(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "format-items":
                self.parseXMLTree(child, curElem)
            else:
                if child.tag != "label":
                    print (f'In handle_tag_format: "{child.tag}" not handled')

    """
        This function handles cleaning up the XML elements between the format_items and its sub-element output.
        <format_items>
            <format_item>
                ...
            </format_item>
            ...
        </format_items>
    """
    def handle_tag_format_items(self, root, parElem):
        for child in root:
            curElem = ET.SubElement(parElem, child.tag, child.attrib)
            if child.tag == "format-items" or child.tag == "format-item":
                self.parseXMLTree(child, curElem)
            else:
                print (f'In handle_tag_format-items: "{child.tag}" not handled')


    """
        parseXMLTree

        Arguments:
            root: The current root of the tree.
            parElem: The parent element.

        Returns:
            None
            
        Recursively traverse through the nested XML AST tree
        and calls the appropriate tag handler, which will generate
        a cleaned version of XML tree for translate.py 
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
        elif root.tag == "name":
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
        else:
            print (f"Currently, {root.tag} not supported")

    """
        reconDerivedType reconstruct the derived type with the collected derived type declaration elements
        in the handle_tag_declaration and handle_tag_type.
    """
    def reconDerivedType(self):
        if self.derivedTypeList:
            if self.hasInitialValue:
                literal = ET.Element('')
            
            # Since component-decl-list appears after component-decl, the program needs to iterate the list
            # once first to pre-collect the variable counts
            counts = []
            for elem in self.derivedTypeList:
                if elem.tag == "component-decl-list":
                    counts.append(elem.attrib['count'])

            # Initialize count to 0 for <variables> count attribute
            count = 0
            # 'component-decl-list__begin' tag is an indication of all the derived type member variable declarations will follow
            derivedType = ET.SubElement(self.parentType, 'component-decl-list__begin')
            for elem in self.derivedTypeList:
                if elem.tag == "intrinsic-type-spec":
                    attributes = {'hasKind': 'false', 'hasLength': 'false', 'name': elem.attrib['keyword1'], 'is_derived_type': str(self.is_derived_type)}
                    newType = ET.SubElement(derivedType, 'type', attributes)
                    # intrinsic = ET.SubElement(newType, elem.tag, elem.attrib) # I'm commenting out this line currently, but may be required later
                elif elem.tag == "derived-type-spec":
                    attributes = {'hasKind': 'false', 'hasLength': 'false', 'name': elem.attrib['typeName'], 'is_derived_type': str(self.is_derived_type)}
                    newType = ET.SubElement(derivedType, 'type', attributes)
                    derived = ET.SubElement(newType, elem.tag, elem.attrib)
                elif elem.tag == "literal":
                    literal = elem;
                elif elem.tag == "component-decl":
                    if len(counts) > count:
                        attr = {'count': counts[count]}
                        newVariables = ET.SubElement(derivedType, 'variables', attr)
                        count += 1
                    var_attribs = {'hasInitialValue': elem.attrib['hasComponentInitialization'], 'name': elem.attrib['id']}
                    newVariable = ET.SubElement(newVariables, 'variable', var_attribs)
                    entity_attribs = {'hasInitialValue': elem.attrib['hasComponentInitialization'], 'id': elem.attrib['id']}
                    if self.hasInitialValue:
                        initValue = ET.SubElement(newVariable, 'initial-value')
                        litValue = ET.SubElement(initValue, 'literal', literal.attrib)
                        self.hasInitialValue = False
                    # newEntity_decl = ET.SubElement(newVariable, 'entity-decl', entity_attribs) # I'm commenting out this line currently, but may be required later

            # Once one derived type was successfully constructed, clear all the elements of a derived type list
            self.derivedTypeList.clear()
            self.is_derived_type = False

    """
        This function will clean up the derived type referencing syntax, which is stored in a form of "id='x'%y" in the id attribute.
        Once the id gets cleaned, it will call the reconstruc_derived_type_ref function to reconstruct and replace the messy version
        of id with the cleaned version.
    """
    def clean_derived_type_ref(self, parElem):
        current_id = parElem.attrib['id'] # 1. Get the original form of derived type id, which is in a form of, for example, id="x"%y in the original XML.
        self.derivedTypeList.append(re.findall(r"\"([^\"]+)\"", current_id)[0]) # 2. Extract the first variable name, for example, x in this case.
        percent_sign = current_id.find("%") # 3. Get the location of the '%' sign
        self.derivedTypeList.append(current_id[percent_sign + 1 : len(current_id)]) # 4. Get the field variable. y in this example.
        self.reconstruct_derived_type_ref(parElem)

    """
        This function reconstruct the id into x.y.k form from the messy looking id.
        One thing to notice is that this new form was generated in the python syntax, so it is a pre-process for translate.py and even pyTranslate.py that
    """
    def reconstruct_derived_type_ref(self, parElem):
        num_of_vars = len(self.derivedTypeList)
        cleaned_id = ""
        for var in self.derivedTypeList:
            cleaned_id += var
            if num_of_vars > 1:
                cleaned_id += '.'
                num_of_vars -= 1
        parElem.attrib['id'] = cleaned_id
        self.derivedTypeList.clear() # Clean up the list for re-use

# ==========================================================================================================================================================================================
"""
    This function is just a helper function for check whether the passed element (i.e. list) is empty or not
"""
def is_empty(elem):
    if not elem:
        return True
    else:
        return False

"""
    This function indents each level of XML.
    Source: https://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
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

    XMLCreator = OFPXMLToCleanedXML()
    # A root of the new AST 
    newRoot = ET.Element(root.tag, root.attrib)

    # First add the root to the new AST list
    for child in root:
        # Handle only non-empty elements
        if child.text:
            curElem = ET.SubElement(newRoot, child.tag, child.attrib)
            XMLCreator.parseXMLTree(child, curElem)

    # Build a new cleaned AST XML
    buildNewAST(newRoot, filename);

main()
# ==========================================================================================================================================================================================
