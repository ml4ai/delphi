from delphi import GroundedFunctionNetwork

with open("relativistic_energy.f", "w") as f:
    f.write("""\
      subroutine relativistic_energy(e, m, c, p)

      implicit none

      real e, m, c, p
      e = sqrt((p**2)*(c**2) + (m**2)*(c**4))

      return
      end subroutine func""")

G = GroundedFunctionNetwork.from_fortran_file("relativistic_energy.f")
A = G.to_agraph()
A.draw("relativistic_energy_grfn.png", prog="dot")
