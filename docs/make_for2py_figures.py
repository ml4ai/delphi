from delphi.GrFN.networks import GroundedFunctionNetwork

G = GroundedFunctionNetwork.from_fortran_src("""\
      subroutine relativistic_energy(e, m, c, p)

      implicit none

      real e, m, c, p
      e = sqrt((p**2)*(c**2) + (m**2)*(c**4))

      return
      end subroutine relativistic_energy"""
)
A = G.to_agraph()
A.draw("relativistic_energy_grfn.png", prog="dot")
