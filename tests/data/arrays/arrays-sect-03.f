C File: arrays-sect-03.f
C Illustrates the use of array sections using an implicit do loop to 
C contiguous array elements

      program main
      implicit none

      integer, dimension (5) :: A
      integer :: i

      do i = 1, 5
          A(i) = i*i
      end do

      write (*,10) A(1), A(2), A(3), A(4), A(5)

 10   format(5(I5,X))


C     Use of an array section to redefine the middle three elements of A
C     This example uses an implicit do loop.
      A(2:4) = 17

      write (*,10) A(1), A(2), A(3), A(4), A(5)

      stop
      end program main

