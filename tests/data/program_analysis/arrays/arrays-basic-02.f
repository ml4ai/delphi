C File: arrays-basic-02.f
C This program has a simple 1-D array with an explicit lower bound.

      program main
      implicit none

C     array is a 1-D array of integers with an explicit lower-bound = -5
C     and an upper bound of 5
      integer, dimension(-5:5) :: array
      integer :: i

      do i = -5, 5
          array(i) = i*i
      end do

      do i = -5, 5
          write (*,10) i, array(i)
      end do

 10   format(I5,X,I5)

      stop
      end program main
