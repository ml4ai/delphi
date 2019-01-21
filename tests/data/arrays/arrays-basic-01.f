C File: arrays-basic-01.f
C This program has a simple 1-D array with the default lower bound of 1.

      program main
      implicit none

C     array is a 1-D array of integers with an implicit lower-bound = 1
C     and an upper bound of 10
      integer, dimension(10) :: array    
      integer :: i

      do i = 1, 10
          array(i) = i*i
      end do

      do i = 1, 10
          print *, array(i)
      end do

      end program main
