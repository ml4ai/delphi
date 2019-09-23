# Copied from boolangery's fork of sphinx-jsondomain
# https://github.com/boolangery/sphinx-jsondomain/blob/master/sphinxjsondomain.py

import json
import re

from docutils import nodes
from docutils.parsers.rst import directives as rst_directives
from sphinx import addnodes, directives, domains, roles
from sphinx.util import docfields
from sphinx.util import nodes as node_utils
import faker

try:
    import yaml
except ImportError:
    yaml = None


def is_json_array(typename):
    """ Check if a property type is an array.
        For example: [int], [uuid4] are array
    """
    return typename and typename.startswith('[') and typename.endswith(']')


def strip_json_array(typename):
    """ Strip json array enclosing brackets from a typename.
        [int] -> int
    """
    if typename and typename.startswith('[') and typename.endswith(']'):
        return typename[1:-1]
    return typename


class JsonField(docfields.TypedField):
    def __init__(self, name, names=(), typenames=[], label=None, examplenames=(),
                 rolename=None, typerolename=None, can_collapse=False):
        docfields.TypedField.__init__(self, name, names, typenames, label, rolename, typerolename, can_collapse)
        self.examplenames = examplenames


class JsonFieldTransformer(docfields.DocFieldTransformer):
    def __init__(self, directive):
        docfields.DocFieldTransformer.__init__(self, directive)

        self.examplemap = {}

        for fieldtype in self.directive.__class__.doc_field_types:
            for name in fieldtype.examplenames:
                self.examplemap[name] = True

    def transform(self, node):
        # type: (nodes.Node) -> None
        """Transform a single field list *node*."""
        typemap = self.typemap

        entries = []
        groupindices = {}  # type: Dict[unicode, int]
        types = {}  # type: Dict[unicode, Dict]

        # step 1: traverse all fields and collect field types and content
        for field in node:
            fieldname, fieldbody = field
            try:
                # split into field type and argument
                fieldtype, fieldarg = fieldname.astext().split(None, 1)
            except ValueError:
                # maybe an argument-less field type?
                fieldtype, fieldarg = fieldname.astext(), ''
            typedesc, is_typefield = typemap.get(fieldtype, (None, None))

            # collect the content, trying not to keep unnecessary paragraphs
            if docfields._is_single_paragraph(fieldbody):
                content = fieldbody.children[0].children
            else:
                content = fieldbody.children

            # sort out unknown fields
            if typedesc is None or typedesc.has_arg != bool(fieldarg):
                continue

            typename = typedesc.name

            # if the field specifies a type, put it in the types collection
            if is_typefield:
                # filter out only inline nodes; others will result in invalid
                # markup being written out
                content = [n for n in content if isinstance(n, nodes.Inline) or
                           isinstance(n, nodes.Text)]
                if content:
                    types.setdefault(typename, {})[fieldarg] = content
                continue

            # also support syntax like ``:param type name:``
            if typedesc.is_typed:
                try:
                    argtype, argname = fieldarg.split(None, 1)
                except ValueError:
                    pass
                else:
                    xrefs = typedesc.make_xrefs(
                        nodes.Text(argtype),
                        self.directive.domain,
                        nodes.Text(argtype),
                    )
                    xrefs[0].attributes['json:name'] = strip_json_array(nodes.Text(argtype))
                    types.setdefault(typename, {})[argname] = xrefs
                    fieldarg = argname

            translatable_content = nodes.inline(fieldbody.rawsource,
                                                translatable=True)
            translatable_content.document = fieldbody.parent.document
            translatable_content.source = fieldbody.parent.source
            translatable_content.line = fieldbody.parent.line
            translatable_content += content

            # grouped entries need to be collected in one entry, while others
            # get one entry per field
            if typedesc.is_grouped:
                if typename in groupindices:
                    group = entries[groupindices[typename]]
                else:
                    groupindices[typename] = len(entries)
                    group = [typedesc, []]
                    entries.append(group)
                entry = typedesc.make_entry(fieldarg, [translatable_content])
                group[1].append(entry)
            else:
                entry = typedesc.make_entry(fieldarg, [translatable_content])
                entries.append([typedesc, entry])

        # step 2: all entries are collected, construct the new field list
        new_list = nodes.field_list()
        for entry in entries:
            if isinstance(entry, nodes.field):
                # pass-through old field
                new_list += entry
            else:
                fieldtype, content = entry
                fieldtypes = types.get(fieldtype.name, {})
                env = self.directive.state.document.settings.env
                new_list += fieldtype.make_field(fieldtypes, self.directive.domain,
                                                 content, env=env)

        node.replace_self(new_list)


class JSONObject(directives.ObjectDescription):
    """
    Implementation of ``json:object``.

    The :meth:`run` method is called to process each ``.. json:object::``
    directive.  It builds the docutil nodes that the builder uses to
    generate the output.  We don't do anything that is too complex
    with formatting.  Cross-references and examples are a different
    story.  The :class:`JSONDomain` instance is responsible for
    creating the examples since we have to wait until all of the JSON
    objects have been parsed to do that.

    Cross-references are generated using the :meth:`JSONDomain.add_object`
    method of the Domain instance.  This stores the JSON object definition
    so that we can access it via the ``:json:object:`` role as well.
    Take a look at :class:`JSONXRef` and :meth:`JSONDomain.resolve_xref`
    for the implementation of the lookup.

    """

    doc_field_types = [JsonField('property',
                                            label='Object Properties',
                                            names=('property', 'property-opt', 'member'),
                                            rolename='prop',
                                            typerolename='jsonprop',
                                            typenames=('proptype', 'type'),
                                            examplenames=['propexample'])]
    """A list of fields that are implemented."""

    option_spec = {
        'noindex': rst_directives.flag,
        'showexample': rst_directives.unchanged,
    }
    """Mapping from supported option to an option processor."""

    has_content = True
    """JSONObject directives accept content."""

    # Inherited Attribute Notes
    # --------- --------- -----
    # self.name - directive name ... probably `object'
    # self.arguments - pos args as list of strings
    # self.options - mapping of name -> value after passing thru option_spec
    # self.content - content as a list of lines (strings)
    # self.lineno - input line number
    # self.src - input file/path
    # self.srcline - input line number in source file
    # self.content_offset - line offset of the first line
    # self.block_text - raw text
    # self.state - `state' which called the directive (?)
    # self.state_machine - `state machine' that controls the state (?)
    #
    # NB self.domain is required to be the name of the sphinx domain
    # inside of DocFieldTransformer

    def run(self):
        """
        Process a ``json:object`` directive.

        This method parses the property definitions and generates a
        single :class:`sphinx.addnodes.desc` element to hold the
        generated docutils tree.  The structure is::

            <sphinx.addnodes.desc>
                <sphinx.addnodes.desc_signature>name
                <sphinx.addnodes.desc_name>name
                <sphinx.addnodes.desc_content>
                    # sub-tree generated from parsing content
                <sphinx.addnodes.compact_paragraph>
                    <docutils.nodes.strong>"JSON Example"
                    <docutils.nodes.literal_block>{...}

        The example block is generated when :meth:`JSONDomain.process_doc`
        is called.  We create the ``compact_paragraph`` node, add it to the
        ``desc`` node and tell the Domain instance to populate the
        ``compact_paragraph`` later on.

        """
        self.domain, sep, objtype = self.name.partition(':')
        if not objtype:  # swap domain and objtype
            objtype = self.domain
            self.domain = ''

        env = self.state.document.settings.env
        self.names = []
        self.indexnode = addnodes.index(entries=[])
        self.domain_obj = env.domains['json']

        node = addnodes.desc()
        node.document = self.state.document
        node['domain'] = self.domain
        node['objtype'] = node['desctype'] = objtype
        noindex = node['noindex'] = 'noindex' in self.options

        name = self.arguments[0]
        signode = addnodes.desc_signature(name)
        node.append(signode)
        signode += addnodes.desc_name(text=name)

        contentnode = addnodes.desc_content()
        node.append(contentnode)
        self.before_content()
        self.state.nested_parse(self.content, self.content_offset, contentnode)

        props = self.domain_obj.get_object(name)
        if props:
            env.warn(env.docname,
                     'JSON type {} already documented in {}'.format(
                         name,
                         env.doc2path(props.docname)),
                     self.lineno)
        else:
            self.domain_obj.add_object(name, env, contentnode)
            if not noindex:
                self.add_target_and_index(name, node, signode)

        JsonFieldTransformer(self).transform_all(contentnode)

        if 'showexample' in self.options:
            paragraph = addnodes.compact_paragraph()
            contentnode.append(paragraph)
            self.domain_obj.data['examples'].append(
                (name, self.options['showexample'] or 'json', paragraph))

        self.after_content()

        return [self.indexnode, node]

    def add_target_and_index(self, name, sig, signode):
        """
        Add an entry for ``name`` to the general index.

        :param str name: name to insert into the index.
        :param docutils.nodes.Element sig: unused
        :param docutils.nodes.Element signode: node to index

        The entry is stored as a child of the ``JSON Objects`` index
        entry.

        """
        key = normalize_object_name(name)
        if key in self.state.document.ids:
            return

        signode['names'].append(name)
        signode['ids'].append(key)
        signode['first'] = not self.names
        self.indexnode['entries'].append(
            ('single', 'JSON Objects; {}'.format(name), key, '', None))


class JSONXRef(roles.XRefRole):
    """Implementation of the ``:json:object:`` cross-reference."""

    def process_link(self, env, refnode, has_explicit_title, title, target):
        """
        Process a single link.

        :param refnode: reference node
        :param title: used as-is
        :param str target: target value
        :return: :class:`tuple` containing the `title` and the target
            after normalizing it as a object name

        The ``json:name`` attribute of `refnode` is set to the normalized
        target name.  This is used in :meth:`JSONDomain.resolve_xref` to
        recognize explicit cross-references and link to the correct
        information.

        """
        refnode['json:name'] = normalize_object_name(target)
        return title, normalize_object_name(target)


class JSONDomain(domains.Domain):
    """
    Implementation of the JSON domain.

    The majority of the work is done by :class:`JSONObject` and the
    existing Sphinx/docutils classes.  The domain instance is responsible
    for tying everything together and providing somewhere to store data
    as the documents are processed.  The ``data`` dictionary contains
    three entries:

    :objects: mapping from normalized object name to
        :class:`PropertyDefinition` instance.  This mapping is cleared for
        each document in :meth:`.clear_doc`.

    :examples: list of "object key", "format", "content node" tuples
        that is processed in :meth:`.process_doc` to generate example
        nodes in the document.  See the :meth:`.generate_examples` method
        for details.

    :all_objects: mapping from normalized object name to
        :class:`PropertyDefinition` instance.  This mapping persists
        for as long as the Domain object.  It is used to keep object
        descriptions around for processing examples.

    """

    name = 'json'
    label = 'JSON'
    data_version = 1
    object_types = {
        'object': domains.ObjType('object', 'object', 'obj'),
    }
    directives = {
        'object': JSONObject,
    }
    roles = {
        'object': JSONXRef(),
    }
    initial_data = {
        'objects': {},  # name -> PropertyDefinition
        'all_objects': {},  # name -> PropertyDefinition
        'examples': [],  # tuple(obj-key, format, content-parent)
    }
    indicies = []

    REF_TYPES = {  # type-name -> (URL, tool tip)
        'uri': ('https://tools.ietf.org/html/rfc3986',
                'URI as described in RFC3986'),
        'boolean': (
            'https://docs.python.org/library/stdtypes.html#boolean-values',
            'Python Boolean'),
        'string': ('https://docs.python.org/library/stdtypes.html#str',
                   'Python String'),
        'integer': ('https://docs.python.org/library/stdtypes.html#int',
                    'Python Integer'),
        'float': ('https://docs.python.org/library/stdtypes.html#float',
                  'Python Float'),
        'null': ('https://docs.python.org/library/constants.html#None',
                 'Python None'),
        'email': ('https://tools.ietf.org/html/rfc2822#section-3.4.1',
                  'Email Address'),
        'iso8601': ('https://tools.ietf.org/html/rfc3339#section-5.6',
                    'ISO8601 Date/Time'),
        'uuid4': ('https://tools.ietf.org/html/rfc4122#section-4.4',
                  'UUIDv4 in canonical syntax'),
        'md5': ('https://tools.ietf.org/html/rfc1321', 'MD5 checksum'),
        'sha1': ('https://tools.ietf.org/html/rfc3174', 'SHA1 checksum'),
        'sha256': ('https://tools.ietf.org/html/rfc6234', 'SHA256 checksum'),
        'sentence': ('https://faker.readthedocs.io/en/master/locales/en_US.html#faker-providers-lorem', 'A sentence'),
    }
    for alias, target in [('url', 'uri'), ('int', 'integer'),
                          ('str', 'string'), ('user_name', 'string'),
                          ('number', 'float'), ('bool', 'boolean')]:
        REF_TYPES[alias] = REF_TYPES[target]

    # create aliases for array
    for target in REF_TYPES.copy().keys():
        REF_TYPES['[%s]' % target] = REF_TYPES[target]

    def clear_doc(self, docname):
        names = [k for k, v in self.data['objects'].items()
                 if v.docname == docname]
        for name in names:
            del self.data['objects'][name]
        del self.data['examples'][:]

    def process_doc(self, env, docname, document):
        super(JSONDomain, self).process_doc(env, docname, document)
        self.generate_examples(docname)

    def get_objects(self):
        for objdef in self.data['objects'].values():
            yield (objdef.name, objdef.name, 'object', objdef.docname,
                   objdef.key, 1)

    def resolve_xref(self, env, fromdocname, builder, typ, target,
                     node, contnode):
        """
        Generate a cross-reference node.

        :param str fromdocname: document name that contains the reference
        :param sphinx.builders.Builder builder: active builder
        :param str typ: identifies the type of cross reference
        :param str target: cross-reference target.  In the case of a
            property, this will be the property type.
        :param docutils.nodes.Element node: unresolved cross-reference
            node.  If this has a ``json:name`` attribute then it was
            processed by :class:`JSONXRef`
        :param docutils.nodes.Element contnode: content node.  This is
            what the new reference node should wrap
        :return: a cross-reference node or :data:`None`
        :rtype: docutils.nodes.reference|NoneType

        This method is also used to make property types ``clickable``.
        In this case the `typ` will be ``jsonprop`` and `target` will
        be the property type.  :data:`.REF_TYPES` is consulted to
        generate an appropriate link and tool tip -- for example, if
        the type is ``uri``, then a link to the RFC is generated.

        """
        if node.get('json:name'):
            objdef = self.get_object(node['json:name'])
            if objdef:
                return node_utils.make_refnode(builder, fromdocname,
                                               objdef.docname, objdef.key,
                                               contnode)

        if typ in self.REF_TYPES:
            try:
                ref = nodes.reference(internal=False)
                ref['refuri'], ref['reftitle'] = self.REF_TYPES[target]
                ref.append(contnode)
                return ref
            except KeyError:
                pass

    def get_object(self, name):
        """
        Retrieve an existing object.

        :param str name: object name

        :return: object defintion or :data:`None`
        :rtype: PropertyDefinition|NoneType

        """
        try:
            return self.data['objects'][normalize_object_name(name)]
        except KeyError:
            return None

    def add_object(self, name, env, contentnode):
        """
        Create a new object.

        :param str name:
        :param env:
        :param docutils.nodes.Element contentnode:

        :return: a new :class:`PropertyDefinition` instance
        :rtype: PropertyDefinition

        """
        props = PropertyDefinition(name, env.docname)
        props.gather(contentnode)
        self.data['objects'][props.key] = props
        self.data['all_objects'][props.key] = props
        return props

    def generate_examples(self, docname):
        """
        Generate example snippets after the document has been processed.

        :param str docname: documentat that is being processed.  This is
            used to report warnings.

        This is called from within :meth:`.process_doc` after our superclass
        has done all of the work.  When an object definition that wants an
        example generated is encountered by :meth:`JSONObject.run`, a new
        empty content node is inserted into the tree and a entry is added
        to our list of examples.

        Once we have processed *all* of the object descriptions, we can
        safely populate examples for objects that refer to each other.
        That is what this method is doing.

        """
        fake_factory = faker.Factory.create()
        for name, language, parent in self.data['examples']:
            props = self.get_object(name)
            sample_data = props.generate_sample_data(self.data['all_objects'],
                                                     fake_factory)
            if language == 'yaml' and yaml is not None:
                title = 'YAML Example'
                code_text = yaml.safe_dump(sample_data, indent=4,
                                           default_flow_style=False,
                                           explicit_start=True,
                                           version=(1, 2))
            else:
                if language == 'yaml':
                    self.env.warn(docname,
                                  'YAML support is disabled, pip install yaml '
                                  'to enable.')
                title = 'JSON Example'
                language = 'json'
                code_text = json.dumps(sample_data, indent=4,
                                       ensure_ascii=False)

            example = nodes.literal_block(code_text, code_text)
            example['language'] = language
            parent.append(nodes.strong(title, title))
            parent.append(example)


class PropertyQualifier:
    def __init__(self):
        self.is_array = False
        self.example_data = None


class PropertyDefinition(object):
    """
    Information about a specific JSON Object definition.

    :param str name: display name (title) of the JSON object
    :param str docname: document that the object is described in
    :param bool should_index: should this be placed in the index?

    """

    def __init__(self, name, docname, should_index=True):
        self.name = name
        self.key = normalize_object_name(name)
        self.docname = docname
        self.should_index = should_index
        self.property_types = {}
        self.property_options = {}
        self.property_qualifiers = {}  # a map of property name --> PropertyQualifier

    def gather(self, contentnode):
        """
        :param docutils.nodes.Element contentnode:
        """
        field_nodes = {}
        optional_props = {}
        for node in contentnode:
            if isinstance(node, nodes.field_list):
                children = list(node)
                for field in node:
                    description, content = field
                    tokens = description.astext().split()
                    if tokens[0] == 'property' or tokens[0] == 'property-opt':
                        if len(tokens) == 3:
                            typ = tokens[1]
                            name = tokens[2]
                        else:
                            typ = None
                            name = tokens[1]

                        # check if name is enclosed by []
                        if is_json_array(typ):
                            typ = strip_json_array(typ)

                            self.property_qualifiers[name] = self.property_qualifiers.get(name, PropertyQualifier())
                            self.property_qualifiers[name].is_array = True

                        self.set_property_type(name, typ)
                        field_nodes[name] = content

                        if tokens[0] == 'property-opt':
                            optional_props[name] = True

                    elif tokens[0] == 'proptype':
                        name = tokens[1]
                        try:
                            maybe_xref = content[0][0]
                            typ = maybe_xref['json:name']
                        except (IndexError, KeyError, TypeError):
                            typ = content.astext()

                        self.set_property_type(name, typ)

                    elif tokens[0] == 'propexample':
                        name = tokens[1]

                        self.property_qualifiers[name] = self.property_qualifiers.get(name, PropertyQualifier())
                        self.property_qualifiers[name].example_data = content[0][0].lstrip()

                    elif tokens[0] == 'options':
                        name = tokens[1]
                        self.property_options[name] = \
                            content.astext().replace(' ', '').split(',')
                        children.remove(field)

                node.children = children

        for opt_name in optional_props.keys():
            try:
                n = field_nodes[opt_name][0]
                n += nodes.inline(' (', ' (')
                n += nodes.strong('optional', 'optional')
                n += nodes.inline(')', ')')
            except KeyError:
                pass

        for name, options in self.property_options.items():
            if not options:
                continue

            try:
                n = field_nodes[name][0]
                n += nodes.inline(' (', ' (')
                n += nodes.emphasis(options[0], options[0])
                for opt in options[1:]:
                    n += nodes.inline(', ', ', ')
                    n += nodes.emphasis(opt, opt)
                n += nodes.inline(')', ')')
            except KeyError:
                pass

    def set_property_type(self, name, typ):
        if name in self.property_types and not typ:
            return

        self.property_types[name] = typ

    def generate_sample_data(self, all_objects, fake_factory):
        sample_data = {}
        for name, typ in self.property_types.items():
            # the data generator
            def gen_data():
                if name in self.property_qualifiers and self.property_qualifiers[name].example_data:

                    # try to cast data to user type
                    data = self.property_qualifiers[name].example_data
                    try:
                        data = __builtins__[typ](data)
                    except:
                        pass

                    return data
                else:
                    return self.generate_sample_data_for_type(typ, all_objects, fake_factory)

            if name in self.property_qualifiers:
                if self.property_qualifiers[name].is_array:
                    sample_data[name] = [
                        gen_data(),
                        gen_data()]
                else:
                    sample_data[name] = gen_data()

            else:
                sample_data[name] = gen_data()

        return sample_data

    @staticmethod
    def generate_sample_data_for_type(typ, all_objects, fake_factory):
        if typ:
            # if name in self.property_qualifiers:
            #    if self.property_qualifiers[name].is_array:

            try:
                other = all_objects[typ]
                value = other.generate_sample_data(all_objects,
                                                   fake_factory)
            except KeyError:
                value = None

            if value is None:
                if hasattr(fake_factory, typ):
                    value = getattr(fake_factory, typ)()
                elif typ in ('integer', 'int'):
                    value = fake_factory.pyint()
                elif typ in ('boolean', 'bool'):
                    value = fake_factory.pybool()
                elif typ in ('float'):
                    value = fake_factory.pyfloat()
                elif typ == 'null':
                    value = None

            if value is None and typ != 'null':
                value = '{%s object}' % typ

        else:
            value = '\uFFFD (Unspecified)'

        return value


def normalize_object_name(obj_name):
    return re.sub(r'\s+', '-', obj_name).strip('-').lower()


def setup(app):
    app.add_domain(JSONDomain)
