import os
from pygraphviz import AGraph
from typing import Dict
import json

import delphi.program_analysis.scopes as scp
from delphi.program_analysis.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.visualization import visualize

from IPython.display import display, Image

def printScopeTree(scope):
    for node in scope.nodes:
        if isinstance(node, scp.ActionNode):
            print(node.lambda_fn)
    for child in scope.child_scopes:
        printScopeTree(child)


if __name__ == "__main__":
    fortran_file = "crop_yield.f"
    G = ProgramAnalysisGraph.from_fortran_file(fortran_file)
    G.initialize()
    visualize(G, save=True)
