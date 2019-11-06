C   File: array-to-func_06.f
C   This test is an example of passing arrays to a function (subroutine)
C   then pass the array to another function under the nested loop.
C
C   The code was sampled from SIR-Gillespie.f

      subroutine update_mean_var(MeanS, VarS, k, n, runs)
      integer, parameter :: Tmax = 100
      integer k, n, runs
      double precision, dimension(0:Tmax) :: MeanS, VarS

      MeanS(k) = MeanS(k) + (n - MeanS(k))/(runs+1)
      VarS(k) = VarS(k) + runs/(runs+1) * (n-MeanS(k))*(n-MeanS(k))

      return
      end subroutine update_mean_var

      subroutine gillespie(MeanS, VarS)
      integer, parameter :: Tmax = 100
      integer runs, sample, i, j

      double precision, dimension(0:Tmax) :: MeanS, VarS
      double precision, dimension(0:Tmax) :: samples

      do i = 0, Tmax
        samples(i) = i
      end do

      do runs = 0, Tmax
        j = 0
        do while (j .le. Tmax)
          call update_mean_var(MeanS, VarS, j, sample, runs)
          j = j + 1
        end do
        sample = samples(runs)
      end do

      end subroutine gillespie

      program main
      implicit none

      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, VarS

      call gillespie(MeanS, VarS)

      end program main
