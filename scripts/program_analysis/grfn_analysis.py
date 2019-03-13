import importlib
import json
import time

import numpy as np
import torch

# from delphi.visualization import visualize
from delphi.GrFN.GroundedFunctionNetwork import GroundedFunctionNetwork


lambdas = importlib.__import__("PETPT_lambdas")
pgm = json.load(open("PETPT.json", "r"))
G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

start = time.time()
for i in range(10000):
    values = {name: np.random.rand() for name in G.inputs}
    G.run(values)
    G.clear()
print(f"Full compute time: {1000 * (time.time() - start):2.4f}ms")


lambdas = importlib.__import__("PETPT_numpy_lambdas")
pgm = json.load(open("PETPT_numpy.json", "r"))
G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

values = {
    "petpt::msalb_0": np.random.rand(1000000000),
    "petpt::srad_0": np.random.randint(1, 100, size=1000000000),
    "petpt::tmax_0": np.random.randint(30, 40, size=1000000000),
    "petpt::tmin_0": np.random.randint(10, 15, size=1000000000),
    "petpt::xhlai_0": np.random.rand(1000000000)
}
result = G.run(values)
print(f"Numpy Final result: {result}")
print(f"Numpy Size of final result: {result.shape}")


lambdas = importlib.__import__("PETPT_torch_lambdas")
pgm = json.load(open("PETPT_numpy.json", "r"))
G = GroundedFunctionNetwork.from_dict(pgm, lambdas)

t_values = {k: torch.Tensor(v) for k, v in values.items()}
if torch.cuda.is_available():
    t_values = {k: v.cuda() for k, v in t_values.items()}
result = G.run(t_values)
print(f"Torch Final result: {result}")
print(f"Torch Size of final result: {result[0].size()}")

# visualize(G, save=True, filename="petpt.pdf")
# visualize(G, save=True, filename="petpt_numpy.pdf")
