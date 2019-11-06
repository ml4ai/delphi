C   File: array-loop_01.f
C   This test is an example of having an array under the nested
C   loop and initialize to zero.
C
C   The code was sampled from SIR-Gillespie.f
      program main
      implicit none

      integer, parameter :: Tmax = 100
      integer i, j

      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR
      integer, dimension(0:Tmax) :: samples

      do j = 0, Tmax
          do i = 0, Tmax    ! Initialize the mean and variance arrays
             MeanS(i) = 0
             MeanI(i) = 0.0
             MeanR(i) = 0.0

             VarS(i) = 0.0
             VarI(i) = 0.0
             VarR(i) = 0.0

             samples(i) = i
          end do
      end do

      end program main
