import ast
import sys


class PrintState:
    def __init__(self, sep=None, add=None):
        self.sep = sep if sep != None else "\n"
        self.add = add if add != None else "    "

    def copy(self, sep=None, add=None):
        return PrintState(
            self.sep if sep == None else sep, self.add if add == None else add
        )


def genCode(node, state):
    codeStr = ""

    if isinstance(node, list):
        for cur in node:
            codeStr += "{0}{1}".format(genCode(cur, state), state.sep)

    # Function: name, args, body, decorator_list, returns
    elif isinstance(node, ast.FunctionDef):
        codeStr = "def {0}({1}):{2}{3}".format(
            node.name,
            genCode(node.args, state),
            state.sep + state.add,
            genCode(node.body, state.copy(state.sep + state.add)),
        )

    # arguments: ('args', 'vararg', 'kwonlyargs', 'kw_defaults', 'kwarg', 'defaults')
    elif isinstance(node, ast.arguments):
        codeStr = "{0}".format(
            ", ".join([genCode(arg, state) for arg in node.args])
        )

    # arg: ('arg', 'annotation')
    elif isinstance(node, ast.arg):
        codeStr = node.arg

    # Load: ()
    elif isinstance(node, ast.Load):
        sys.stderr.write("genCode found ast.Load, is there a bug?\n")

    # Store: ()
    elif isinstance(node, ast.Store):
        sys.stderr.write("genCode found ast.Store, is there a bug?\n")

    # Index: ('value',)
    elif isinstance(node, ast.Index):
        codeStr = "[{0}]".format(genCode(node.value, state))

    # Num: ('n',)
    elif isinstance(node, ast.Num):
        codeStr = "{0}".format(node.n)

    # List: ('elts', 'ctx')
    elif isinstance(node, ast.List):
        elements = [genCode(elmt, state) for elmt in node.elts]
        codeStr = (
            "{0}".format(elements[0])
            if len(elements) == 1
            else "[{0}]".format(", ".join(elements))
        )

    # Str: ('s',)
    elif isinstance(node, ast.Str):
        codeStr = '"{0}"'.format(node.s)

    # For: ('target', 'iter', 'body', 'orelse')
    elif isinstance(node, ast.For):
        codeStr = "for {0} in {1}:{2}{3}".format(
            genCode(node.target, state),
            genCode(node.iter, state),
            state.sep + state.add,
            genCode(node.body, state.copy(state.sep + state.add)),
        )

    # If: ('test', 'body', 'orelse')
    elif isinstance(node, ast.If):
        codeStr = "if ({0}):{1}{2}{3}else:{4}{5}".format(
            genCode(node.test, state),
            state.sep + state.add,
            genCode(node.body, state.copy(state.sep + state.add)),
            state.sep,
            state.sep + state.add,
            genCode(node.orelse, state.copy(state.sep + state.add)),
        )

    # UnaryOp: ('op', 'operand')
    elif isinstance(node, ast.UnaryOp):
        codeStr = "{0}({1})".format(
            genCode(node.op, state), genCode(node.operand, state)
        )

    # BinOp: ('left', 'op', 'right')
    elif isinstance(node, ast.BinOp):
        codeStr = "({0}{1}{2})".format(
            genCode(node.left, state),
            genCode(node.op, state),
            genCode(node.right, state),
        )

    # Add: ()
    elif isinstance(node, ast.Add):
        codeStr = "+"

    elif isinstance(node, ast.Sub):
        codeStr = "-"

    elif isinstance(node, ast.Mult):
        codeStr = "*"

    # Pow: ()
    elif isinstance(node, ast.Pow):
        codeStr = "**"

    # Div: ()
    elif isinstance(node, ast.Div):
        codeStr = "/"

    # USub: ()
    elif isinstance(node, ast.USub):
        codeStr = "-"

    # Eq: ()
    elif isinstance(node, ast.Eq):
        codeStr = "="

    # LtE: ()
    elif isinstance(node, ast.LtE):
        codeStr = "<="

    # Lt: ()
    elif isinstance(node, ast.Lt):
        codeStr = "<"

    # Gt: ()
    elif isinstance(node, ast.Gt):
        codeStr = ">"

    # Expr: ('value',)
    elif isinstance(node, ast.Expr):
        codeStr = genCode(node.value, state)

    # Compare: ('left', 'ops', 'comparators')
    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1:
            sys.stderr.write(
                "Fix Compare in genCode! Don't have an example of what this will look like\n"
            )
        else:
            codeStr = "({0}{1}{2})".format(
                genCode(node.left, state),
                genCode(node.ops[0], state),
                genCode(node.comparators[0], state),
            )

    # Subscript: ('value', 'slice', 'ctx')
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.slice.value, ast.Num):
            sys.stderr.write("can't handle arrays in genCode right now\n")
            sys.exit(1)
        # typical:
        # codeStr = '{0}{1}'.format(genCode(node.value, state), genCode(node.slice, state))
        codeStr = "{0}".format(genCode(node.value, state))

    # Name: ('id', 'ctx')
    elif isinstance(node, ast.Name):
        codeStr = node.id

    # AnnAssign: ('target', 'annotation', 'value', 'simple')
    elif isinstance(node, ast.AnnAssign):
        codeStr = "{0} = {1}".format(
            genCode(node.target, state), genCode(node.value, state)
        )

    # Assign: ('targets', 'value')
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            codeStr = "{0} = ".format(genCode(target, state))

        codeStr += "{0}".format(genCode(node.value, state))

    # Call: ('func', 'args', 'keywords')
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            fnNode = node.func
            module = fnNode.value.id
            fnName = fnNode.attr
            fnName = module + '.' + fnName
        else:
            fnName = node.func.id
        codeStr = "{0}(".format(fnName)

        if len(node.args) > 0:
            codeStr += ", ".join([genCode(arg, state) for arg in node.args])

        codeStr += ")"

    elif isinstance(node, ast.Import):
        codeStr = "import {0}{1}".format(
            ", ".join([genCode(name, state) for name in node.names]), state.sep
        )

    elif isinstance(node, ast.alias):
        if node.asname == None:
            codeStr = node.name
        else:
            codeStr = "{0} as {1}".format(node.name, node.asname)

    # Module: body
    elif isinstance(node, ast.Module):
        codeStr = genCode(node.body, state)

    elif isinstance(node, ast.AST):
        sys.stderr.write(
            "No handler for AST.{0} in genCode, fields: {1}\n".format(
                node.__class__.__name__, node._fields
            )
        )

    else:
        sys.stderr.write(
            "No handler for {0} in genCode, value: {1}\n".format(
                node.__class__.__name__, str(node)
            )
        )

    return codeStr
