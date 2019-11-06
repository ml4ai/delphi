C   File: array-to-func_05.f
C   This test is an example of passing arrays to a function (subroutine)
C   as arguments twice and initialize the array under single and nested 
C   loop.
C
C   The code was sampled from SIR-Gillespie.f

      subroutine update_mean_var(MeanS, MeanI, MeanR, VarS, VarI, VarR)
      integer, parameter :: Tmax = 100

      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      integer i, j

      do j = 0, Tmax
             MeanS(j) = 0
             MeanI(j) = 0.0
             MeanR(j) = 0.0
          do i = 0, Tmax    ! Initialize the mean and variance arrays
             VarS(i) = 0.0
             VarI(i) = 0.0
             VarR(i) = 0.0
          end do
      end do

      return
      end subroutine update_mean_var

      subroutine gillespie(MeanS, MeanI, MeanR, VarS, VarI, VarR)
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      call update_mean_var(MeanS, MeanI, MeanR, VarS, VarI, VarR)

      end subroutine gillespie

      program main
      implicit none

      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      call gillespie(MeanS, MeanI, MeanR, VarS, VarI, VarR)

      end program main
