import sys
import importlib

from delphi.GrFN.networks import GroundedFunctionNetwork

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

print('Running demo_generate_grfn.py')
data_dir = "scripts/SIR_Demo/"
sys.path.insert(0, data_dir)
model_file = "SIR-Gillespie-SD"
json_file = f"{model_file}_GrFN.json"
lambdas = importlib.__import__(f"{model_file}_lambdas")

# grfn = GroundedFunctionNetwork.from_fortran_file("scripts/SIR_Demo/SIR-simple.f")
grfn = GroundedFunctionNetwork.from_json_and_lambdas(json_file, lambdas)
agraph = grfn.to_agraph()
agraph.draw('SIR-gillespie.pdf', prog='dot')
# CAG = grfn.to_CAG_agraph()
# CAG.draw('SIR-gillespie-CAG.pdf', prog='dot')

# -----------------------------------------------------------------------------
