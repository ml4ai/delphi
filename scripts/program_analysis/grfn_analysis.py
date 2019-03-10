import importlib
import json

from delphi.visualization import visualize
from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork


# filepath = "../../tests/data/PETPT.for"
# G = GroundedFunctionNetwork.from_fortran_file(filepath)

lambdas = importlib.__import__("PETPT_lambdas")
pgm = json.load(open("PETPT.json", "r"))
G = GroundedFunctionNetwork.from_dict(pgm, lambdas)
# print(G)
values = {name: 1 for name in G.inputs}
G.run(values)
# visualize(G, save=True, filename="petpt.pdf")
