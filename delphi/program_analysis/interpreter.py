from os.path import normpath
from pygraphviz import AGraph
from typing import Dict
import json

import delphi.program_analysis.scopes as scp
import delphi.program_analysis.ProgramAnalysisGraph as pag
from delphi.visualization import visualize
import delphi.program_analysis.autoTranslate.lambdas as lambdas

from IPython.display import display, Image


def printScopeTree(scope):
    for node in scope.nodes:
        if isinstance(node, scp.ActionNode):
            print(node.lambda_fn)
    for child in scope.child_scopes:
        printScopeTree(child)


if __name__ == "__main__":
    dbn_json_file = "autoTranslate/pgm.json"
    scope = scp.Scope.from_json(normpath(dbn_json_file))
    # printScopeTree(scope)

    A = scope.to_agraph()
    pgraph = pag.ProgramAnalysisGraph.from_agraph(A, lambdas)
    petpt_graph = "petpt-pa-graph"
    pgraph.initialize()
    visualize(pgraph, save_to_dot=petpt_graph, show_values=True)

    # B = AGraph("{}.dot".format(petpt_graph))
    # B.draw("{}.png".format(petpt_graph), prog="dot")
