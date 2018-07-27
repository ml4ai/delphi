from delphi.program_analysis.printAst import importAst, dump
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
    print('parents[1]:', (Path(__file__).parents[1]))
    filename = (Path(__file__).parents[1]/'..'/'data'/
            'program_analysis'/'pa_crop_yield_v0.1'/module_name).resolve()

    print(get_AST_str_rep(str(filename)))
