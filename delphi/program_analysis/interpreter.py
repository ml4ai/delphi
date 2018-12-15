import os
from delphi.program_analysis.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.visualization import visualize

def printScopeTree(scope):
    for node in scope.nodes:
        if isinstance(node, scp.ActionNode):
            print(node.lambda_fn)
    for child in scope.child_scopes:
        printScopeTree(child)


if __name__ == "__main__":
    fortran_file = "../../tests/data/crop_yield.f"
    G = ProgramAnalysisGraph.from_fortran_file(fortran_file)
    print(G.input_variables)
    # G.initialize()
    # visualize(G, save=True)
