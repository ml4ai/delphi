from ast import AST, iter_fields
import ast
import sys
import tokenize
from typing import Union, List


def dump(
    node: AST,
    annotate_fields: bool = True,
    include_attributes: bool = False,
    indent="  ",
) -> str:
    """ This is mainly useful for debugging purposes. The returned string will show
    the names and the values for fields.  This makes the code impossible to
    evaluate, so if evaluation is wanted, *annotate_fields* must be set to False.
    Attributes such as line numbers and column offsets are not dumped by
    default. If this is wanted, *include_attributes* can be set to True.

    Args:
        annotate_fields: flag for whether to include fields (when true,
                         cannot execute AST)
        include_attributes: flag for whether to include attributes
        indent: tree indentation string
    Returns:
        A string representation of the AST node.
    """
    if not isinstance(node, AST):
        raise TypeError("expected AST, got %r" % node.__class__.__name__)

    def _format(node: Union[AST, List], level=0) -> str:
        if isinstance(node, AST):
            fields = [(a, _format(b, level)) for a, b in iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend(
                    [
                        (a, _format(getattr(node, a), level))
                        for a in node._attributes
                    ]
                )
            return "".join(
                [
                    node.__class__.__name__,
                    "(",
                    ", ".join(
                        ("%s=%s" % field for field in fields)
                        if annotate_fields
                        else (b for a, b in fields)
                    ),
                    ")",
                ]
            )
        elif isinstance(node, list):
            lines = ["["]
            lines.extend(
                (
                    indent * (level + 2) + _format(x, level + 2) + ","
                    for x in node
                )
            )
            if len(lines) > 1:
                lines.append(indent * (level + 1) + "]")
            else:
                lines[-1] += "]"
            return "\n".join(lines)
        return repr(node)

    return _format(node)


def importAst(filename):
    return ast.parse(tokenize.open(filename).read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: printAst: <files>")
        sys.exit(1)

    asts = list()
    asts.append(importAst(sys.argv[1]))

    for tree in asts:
        print(dump(tree, annotate_fields=True, include_attributes=True))
