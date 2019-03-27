C File: arrays-sect-02.f
C Illustrates the use of array sections using explicit indices to 
C non-contiguous array elements

      program main
      implicit none

      integer, dimension (5) :: A
      integer :: i

      do i = 1, 5
          A(i) = i*i
      end do

      write (*,10) A(1), A(2), A(3), A(4), A(5)

 10   format(5(I5,X))


C     Use of an array section to redefine alternate elements of A.
C     This example uses explicit indices.
      A((/1,3,5/)) = 17

      write (*,10) A(1), A(2), A(3), A(4), A(5)

      stop
      end program main

