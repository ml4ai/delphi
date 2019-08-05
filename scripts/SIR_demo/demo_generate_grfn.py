import sys

from delphi.GrFN.networks import GroundedFunctionNetwork

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

print('Running demo_generate_grfn.py')
data_dir = "scripts/SIR_Demo/"
sys.path.insert(0, "scripts/SIR_Demo/")

grfn = GroundedFunctionNetwork.from_fortran_file("scripts/SIR_Demo/SIR-simple.f")
agraph = grfn.to_agraph()
agraph.draw('SIR-gillespie.pdf', prog='dot')
CAG = grfn.to_CAG_agraph()
CAG.draw('SIR-gillespie-CAG.pdf', prog='dot')

# -----------------------------------------------------------------------------
