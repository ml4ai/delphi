import sys
import importlib
import networkx as nx

from delphi.GrFN.networks import GroundedFunctionNetwork

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

print('Running demo_generate_grfn.py')
data_dir = "scripts/SIR_Demo/"
sys.path.insert(0, data_dir)
model_file = "SIR-Gillespie-SD__alt"
json_file = f"{model_file}_GrFN.json"
lambdas = importlib.__import__(f"{model_file}_lambdas")

# grfn = GroundedFunctionNetwork.from_fortran_file("scripts/SIR_Demo/SIR-simple.f")
grfn = GroundedFunctionNetwork.from_json_and_lambdas(json_file, lambdas)
agraph = grfn.to_agraph()
agraph.draw('SIR-gillespie_alt.pdf', prog='dot')


CAG = grfn.to_CAG()
model_nodes = [
    "S0",
    "I0",
    "R0",
    "gamma",
    "rho",
    "beta",
    "n_S",
    "n_I",
    "n_R",
    "t",
    "dt",
    "rateInfect",
    "rateRecover",
    "totalRates"
]

all_nodes = list(CAG.nodes())
print(all_nodes)
solver_nodes = list(set(all_nodes) - set(model_nodes))

A = nx.nx_agraph.to_agraph(CAG)
A.graph_attr.update({"dpi": 227, "fontsize": 20, "fontname": "Menlo", "rankdir": "TB"})
A.node_attr.update(
    {
        "shape": "rectangle",
        "color": "#650021",
        "style": "rounded",
        "fontname": "Menlo",
    }
)
A.add_nodes_from(model_nodes)
A.add_nodes_from(solver_nodes)
A.add_subgraph(
    model_nodes,
    name="cluster_model",
    label="model",
    style="bold, rounded",
    color="black"
)
A.add_subgraph(
    solver_nodes,
    name="cluster_solver",
    label="Solver",
    style="bold, rounded",
    color="black"
)
A.edge_attr.update({"color": "#650021", "arrowsize": 0.5})
A.draw('SIR-gillespie-CAG_alt.pdf', prog='dot')

# -----------------------------------------------------------------------------
