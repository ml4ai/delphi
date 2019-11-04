C     Fortranification of AMIDOL's SIR-Gillespie.py
********************************************************************************
      program main
      use mod_gillespie
      integer, parameter :: S0 = 500, I0 = 10, R0 = 0, Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      call gillespie(S0, I0, R0, MeanS, MeanI, MeanR, VarS, VarI, VarR)

      end program main
