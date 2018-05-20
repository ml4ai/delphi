import printAst
import os
from pathlib import Path, PurePath


DATA_ROOT = Path(__file__).parents[0]/'..'/'data'
PROGRAM_ANALYSIS_DATA_ROOT = os.path.join(DATA_ROOT, 'program_analysis')
TARGET_PATH = os.path.join(PROGRAM_ANALYSIS_DATA_ROOT, 'crop_yield.py')


def run_example():
    tree = printAst.importAst(TARGET_PATH)
    print(print(printAst.dump(tree, annotate_fields=True, include_attributes=True)))


if __name__ == '__main__':
    print(DATA_ROOT)
    # run_example()
