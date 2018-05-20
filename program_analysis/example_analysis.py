import printAst
import os
from pathlib import Path


TARGET_PATH = (Path(__file__).parents[0]/'..'/'data'/'program_analysis'/
    'crop_yield.py').resolve()


def run_example():
    tree = printAst.importAst(TARGET_PATH)
    print(printAst.dump(tree, annotate_fields=True, include_attributes=True))


if __name__ == '__main__':
    run_example()
