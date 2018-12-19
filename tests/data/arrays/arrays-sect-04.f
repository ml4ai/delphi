C File: arrays-sect-04.f
C Illustrates the use of array sections using an implicit do loop to 
C non-contiguous array elements

      program main
      implicit none

      integer, dimension (5) :: A
      integer :: i

      do i = 1, 5
          A(i) = 0
      end do

      write (*,10) A(1), A(2), A(3), A(4), A(5)

 10   format(5(I5,X))


C     Use of a more complex array section to redefine alternating elements of A
C     This example uses an implicit do loop.
      A(1:5:2) = 17

      write (*,10) A(1), A(2), A(3), A(4), A(5)

      stop
      end program main

