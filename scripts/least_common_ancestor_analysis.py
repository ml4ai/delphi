from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import inspect
import  Levenshtein
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN


def get_basename(node_name):
    (_, _, _, _, basename, _) = node_name.split("::")
    return basename


PNO_GrFN = GrFN.from_fortran_file(f"../tests/data/program_analysis/PETPNO.for")
PEN_GrFN = GrFN.from_fortran_file(f"../tests/data/program_analysis/PETPEN.for")

# Use basenames for variable comparison because the two GrFNs will have those in common
PNO_nodes = [
    d["basename"]
    for n, d in PNO_GrFN.nodes(data=True)
    if d["type"] == "variable"
]
PEN_nodes = [
    d["basename"]
    for n, d in PEN_GrFN.nodes(data=True)
    if d["type"] == "variable"
]
# print(PNO_nodes)

shared_nodes = list(set(PNO_nodes).intersection(set(PEN_nodes)))
# Make a map so we can access the original variable names from the basenames
PNO_input_map = {get_basename(node): node for node in PNO_GrFN.inputs}
PEN_input_map = {get_basename(node): node for node in PEN_GrFN.inputs}

PNO_inputs = list(PNO_input_map.keys())
PEN_inputs = list(PEN_input_map.keys())

# Find both sets of shared inputs
PNO_shared_inputs = list(set(PNO_inputs).intersection(set(shared_nodes)))
PEN_shared_inputs = list(set(PEN_inputs).intersection(set(shared_nodes)))

print(PNO_shared_inputs)
print(PEN_shared_inputs)

# Reverse the graph so that LCA analysis will work
mock_PNO_GrFN = nx.DiGraph()
mock_PNO_GrFN.add_edges_from([(dst, src) for src, dst in PNO_GrFN.edges])

mock_PEN_GrFN = nx.DiGraph()
mock_PEN_GrFN.add_edges_from([(dst, src) for src, dst in PEN_GrFN.edges])

# Pick some shared inputs for LCA analysis
PNO_input1, PNO_input2 = (
    PNO_input_map[PNO_shared_inputs[0]],
    PNO_input_map[PNO_shared_inputs[1]],
)
print(PNO_input1, PNO_input2)

# Because of the layout of a GrFN the returned node will always be the LCA function node
LCA = lowest_common_ancestor(mock_PNO_GrFN, PNO_input1, PNO_input2)
# print(LCA)

###############################################################


# Get the actual function code for this node via the attribute "lambda_fn"

LCA_test1 = lowest_common_ancestor(mock_PNO_GrFN,'PETPNO::@global::petpno::0::tmax::-1',  'PETPNO::@global::petpno::0::srad::-1')
print(LCA_test1)

LCA_test2 = lowest_common_ancestor(mock_PEN_GrFN,'PETPEN::@global::petpen::0::tmax::-1', 'PETPEN::@global::petpen::0::srad::-1')
print(LCA_test2)

for x, y in PNO_GrFN.node(data='lambda_fn'):
    if y is not None and x == LCA_test1:
        common_function1 = y

for x, y in PEN_GrFN.node(data='lambda_fn'):
    if y is not None and x == LCA_test2:
        common_function2 = y

print(common_function1)
print(common_function2)

# Use inspect.getsourcelines() to grab the stringified lambda functions

lambda_fn1 = inspect.getsourcelines(common_function1)[0][1].split('return')[-1].split('\n')[0]
print(lambda_fn1)

lambda_fn2 = inspect.getsourcelines(common_function2)[0][1].split('return')[-1].split('\n')[0]
print(lambda_fn2)

# Compare the stringified lambda functions with a string based levenshtein distance

print(Levenshtein.distance(lambda_fn1, lambda_fn2))  # Result is 0
print(Levenshtein.distance(lambda_fn1.upper(), lambda_fn2)) # Result is 14


# Recover the differences between the strings with dynamic programming

def editdistDP(fn1, fn2, m, n):
    
    DP = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        for  j  in range(n+1):
            
            if i == 0:
                DP[i][j] = j

            elif j == 0:
                DP[i][j] = i

            elif fn1[i-1] == fn2[j-1]:
                DP[i][j] = DP[i-1][j-1]

            else:
                DP[i][j] = 1 + min(DP[i][j-1], DP[i-1][j], DP[i-1][j-1])


    return DP[m][n]


print(editdistDP(lambda_fn1.upper(), lambda_fn2, len(lambda_fn1), len(lambda_fn2)))
