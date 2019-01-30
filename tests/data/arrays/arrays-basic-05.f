C File: arrays-basic-05.f
C This program has a 2-D array with the explicit lower bounds.

      program main
      implicit none

C     A is a 2-D array (5x5) of integers with explicit lower-bounds.
      integer, dimension(-3:1,-4:0) :: A
      integer :: i, j

      do i = -3, 1
        do j = -4, 0
          A(i,j) = i+j     ! initialize the array
        end do
      end do

      ! write out the array
      do i = -3, 1
          write (*,10) A(-1,-4), A(i,-3), A(i,-2), A(i,-1), A(i,0)
      end do

 10   format(5(I5,X))

      stop
      end program main
