from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor
import networkx as nx
import matplotlib.pyplot as plt

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

# Pick some shared inputs for LCA analysis
PNO_input1, PNO_input2 = (
    PNO_input_map[PNO_shared_inputs[0]],
    PNO_input_map[PNO_shared_inputs[1]],
)
print(PNO_input1, PNO_input2)

# Because of the layout of a GrFN the returned node will always be the LCA function node
# TODO Khan: Get the actual function code for this node via the attribute "lambda_fn"
# TODO Khan: use inspect.getsourcelines() to grab the stringified lambda functions
# TODO Khan: compare the stringified lambda functions with a string based levenshtein distance
# TODO Khan: recover the differences between the strings with dynamic programming
LCA = lowest_common_ancestor(mock_PNO_GrFN, PNO_input1, PNO_input2)
print(LCA)
