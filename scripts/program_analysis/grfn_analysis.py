import importlib
import json

from delphi.visualization import visualize
from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork


# filepath = "../../tests/data/PETPT.for"
# G = GroundedFunctionNetwork.from_fortran_file(filepath)

lambdas = importlib.__import__("PETPT_lambdas")
pgm = json.load(open("PETPT.json", "r"))
G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
print(G.nodes)
print(G.edges)

# print([n for n, d in G.out_degree() if d == 0])
# print([n for n, d in G.in_degree() if d == 0])
visualize(G, save=True, filename="petpt.pdf")
