import os
from delphi.GrFN.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.visualization import visualize
from delphi.GrFN.scopes import Scope


def printScopeTree(scope):
    for node in scope.nodes:
        if isinstance(node, Scope.ActionNode):
            print(node.lambda_fn)
    for child in scope.child_scopes:
        printScopeTree(child)


if __name__ == "__main__":
    fortran_file = "data/PETPT.for"
    A = Scope.from_fortran_file(fortran_file).to_agraph()
    A.draw("PETPT.pdf", prog="dot")
    # G.initialize()
    # visualize(G, save=True)
