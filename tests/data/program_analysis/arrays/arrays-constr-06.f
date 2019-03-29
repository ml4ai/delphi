C File: arrays-constr-06.f
C This program has a simple 1-D array with the default lower bound of 1.
C It shows the use of an array constructor to initialize the array.

      program main
      implicit none

C     A is a 1-D array of integers with an implicit lower-bound = 1
C     and an upper bound of 12.  
      integer, dimension(12) :: A
      integer, dimension(12) :: B = (/12,11,10,9,8,7,6,5,4,3,2,1/)
      integer, dimension(12) :: C = (/1,3,5,7,9,11,13,15,17,19,21,23/)
      integer :: i

C     array constructor III -- implied do loop, nested array expressions
C     In this example, the array expression consists of nested array
C     references.  The inner expression B(1:12) gives the sequence of
C     values of B's elements in the order B(1), B(2), ..., B(12), i.e.:
C
C         12, 11, 10, ..., 3, 2, 1
C
C     The outer expression uses this sequence of values to index into
C     the array C, giving the sequence of values of C's elements in the
C     order C(12), C(11), C(10), ..., C(3), C(2), C(1).  The sequence of
C     values assigned to the elements of A is therefore:
C
C         23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1

      A = (/ (C(B(1:12))) /)

      do i = 1, 12
          write (*,10) A(i)
      end do

 10   format(I5)

      stop
      end program main
