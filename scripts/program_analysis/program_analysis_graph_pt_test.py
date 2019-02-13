from delphi.GrFN.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.visualization import visualize


filepath = "../../tests/data/PETPT.for"
G = ProgramAnalysisGraph.from_fortran_file(filepath)
print(G.nodes)
print([n for n, d in G.out_degree() if d == 0])
print([n for n, d in G.in_degree() if d == 0])
visualize(G, save=True)
