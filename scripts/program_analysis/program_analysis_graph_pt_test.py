from delphi.GrFN.ProgramAnalysisGraph import ProgramAnalysisGraph
from delphi.visualization import visualize


filepath = "../../tests/data/crop_yield.f"
G = ProgramAnalysisGraph.from_fortran_file(filepath)
G.initialize()
print(G.nodes)
print([n for n, d in G.out_degree() if d == 0])
print([n for n, d in G.in_degree() if d == 0])
# visualize(G, save=True)
