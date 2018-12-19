C File: arrays-sect-09.f
C Illustrates the use of array sections using implicit do loops to access both 
C the source and destination of each assignment.

      program main
      implicit none

      integer, dimension (5) :: A
      integer, dimension(10) :: B
      integer :: i

      do i = 1, 5
          A(i) = 0
          B(i) = 11*i
          B(i+5) = 7*i+6
      end do

      write (*,10) A(1), A(2), A(3), A(4), A(5)

 10   format(5(I5,X))


C     Use of a more complex array section to redefine alternating elements of A
C     This example uses implicit do loops to access both the source and
C     destination elements.
      A(1:5:2) = B(2:8:3)

      write (*,10) A(1), A(2), A(3), A(4), A(5)

      stop
      end program main

