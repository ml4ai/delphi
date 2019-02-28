from delphi.GrFN.scopes import Scope
from delphi.visualization import visualize
from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork


filepath = "../../tests/data/PETPT.for"
G = GroundedFunctionNetwork.from_fortran_file(filepath)
print(G.nodes)
print(G.edges)
# print(G.nodes)
# print([n for n, d in G.out_degree() if d == 0])
# print([n for n, d in G.in_degree() if d == 0])
# visualize(G, save=True)
