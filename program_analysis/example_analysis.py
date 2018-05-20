import printAst
import os


DATA_ROOT = '../data'
PROGRAM_ANALYSIS_DATA_ROOT = os.path.join(DATA_ROOT, 'program_analysis')
TARGET_PATH = os.path.join(PROGRAM_ANALYSIS_DATA_ROOT, 'crop_yield.py')


def run_example():
    tree = printAst.importAst(TARGET_PATH)
    print(print(printAst.dump(tree, annotate_fields=True, include_attributes=True)))


if __name__ == '__main__':
    run_example()
