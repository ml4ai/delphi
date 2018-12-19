C File: arrays-sect-08.f
C Illustrates the use of array sections using the values of another
C array to specify the array elements being referenced.  In this
C example the target elements referenced are non-contiguous.

      program main
      implicit none

      integer, dimension (5) :: A
      integer, dimension (3) :: B = (/1, 3, 5/)

      integer :: i

      do i = 1, 5
          A(i) = 0
      end do

      write (*,10) A(1), A(2), A(3), A(4), A(5)

 10   format(5(I5,X))


C     Use of a more complex array section to redefine all elements of A
C     This example uses an array as the index.
      A(B) = 17

      write (*,10) A(1), A(2), A(3), A(4), A(5)

      stop
      end program main

