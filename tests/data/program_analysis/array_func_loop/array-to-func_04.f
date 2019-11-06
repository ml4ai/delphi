C   File: array-to-func_04.f
C   This test is an example of passing arrays to a function (subroutine)
C   as arguments once and initialize the array under a nested loop.
C
C   The code was sampled from SIR-Gillespie.f

      subroutine update_mean_var(meanS, varS)
      integer, parameter :: Tmax = 100

      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      integer i, j

      do j = 0, Tmax
          do i = 0, Tmax    ! Initialize the mean and variance arrays
             MeanS(i) = 0
             MeanI(i) = 0.0
             MeanR(i) = 0.0

             VarS(i) = 0.0
             VarI(i) = 0.0
             VarR(i) = 0.0
          end do
      end do

      return
      end subroutine update_mean_var

      program main
      implicit none

      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      call update_mean_var(meanS, varS)

      end program main
