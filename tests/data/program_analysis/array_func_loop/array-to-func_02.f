C   File: array-to-func_02.f
C   This test is an example of passing arrays to a function (subroutine)
C   as arguments twice. In other words, the arrays are declared in the
C   main and passed to subroutine (function) gillespie to
C   update_mean_var.
C
C   The code was sampled from SIR-Gillespie.f

      subroutine update_mean_var(meanS, varS, k, n, runs)
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: meanS, varS
      integer k, n, runs

      meanS(k) = meanS(k) + (n - meanS(k))/(runs+1)
      varS(k) = varS(k) + runs/(runs+1) * (n-meanS(k))*(n-meanS(k))

      return
      end subroutine update_mean_var

      subroutine gillespie(meanS, varS)
      integer, parameter :: Tmax = 100, k = 0, n = 0, runs = 50
      double precision, dimension(0:Tmax) :: meanS, varS

      meanS(k) = 100
      varS(k) = 200
      call update_mean_var(meanS, varS, k, n, runs)

      end subroutine gillespie

      program main
      implicit none

      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: meanS, varS

      call gillespie(meanS, varS)

      end program main
