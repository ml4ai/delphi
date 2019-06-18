from delphi.GrFN.networks import GroundedFunctionNetwork

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

print('Running demo_generate_grfn.py')

source_fortran_file = 'DiscreteSIR-noarrays.f'

print(f'    source_fortran_file: {source_fortran_file}')

grfn = GroundedFunctionNetwork.from_fortran_file(source_fortran_file)
agraph = grfn.to_agraph()
agraph.draw('graph.pdf', prog='dot')

# -----------------------------------------------------------------------------
