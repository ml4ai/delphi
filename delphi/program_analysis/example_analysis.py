from .printAst import importAst, dump
import os
from pathlib import Path


def get_AST_str_rep(filepath: str) -> str:
    """ Returns a string representation of the AST of a given Python source
    file."""

    try:
        f = Path(filepath).resolve()
    except FileNotFoundError:
        pass

    return dump(importAst(f), annotate_fields=True, include_attributes=True)


if __name__ == '__main__':
    module_name = 'crop_yield.py'
    filename = (Path(__file__).parents[1]/'..'/'data'/
            'program_analysis'/module_name).resolve()

    print(get_AST_str_rep(str(filename)))
