C   File: array-to-func_01.f
C   This test is an example of passing arrays to a function (subroutine)
C   as arguments once.
C
C   The code was sampled from SIR-Gillespie.f

      function update_mean_var(meanS, varS, k, n, runs)
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: meanS, varS
      integer k, n, runs

      meanS(k) = meanS(k) + (n - meanS(k))/(runs+1)
      varS(k) = varS(k) + runs/(runs+1) * (n-meanS(k))*(n-meanS(k))

      return
      end function update_mean_var

      program main
      implicit none

      integer, parameter :: Tmax = 100, k = 0, n = 0, runs = 50
      double precision, dimension(0:Tmax) :: meanS, varS

      call update_mean_var(meanS, varS, k, n, runs)

      end program main
