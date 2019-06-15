from delphi.GrFN.networks import GroundedFunctionNetwork

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

grfn = GroundedFunctionNetwork.from_fortran_file('DiscreteSIR-simple.f')
agraph = grfn.to_agraph()
agraph.draw('graph.pdf', prog='dot')

# -----------------------------------------------------------------------------
