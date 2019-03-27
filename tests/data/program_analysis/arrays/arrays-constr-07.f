C File: arrays-constr-07.f
C This program has a simple 1-D array with the default lower bound of 1.
C It shows the use of an array constructor to initialize the array.

      program main
      implicit none

C     A, B, C are 1-D arrays of integers with implicit lower-bound = 1
C     and upper bound of 12.  
      integer, dimension(12) :: A
      integer, dimension(12) :: B = (/12,11,10,9,8,7,6,5,4,3,2,1/)
      integer, dimension(12) :: C = (/1,3,5,7,9,11,13,15,17,19,21,23/)
      integer :: i

C     array constructor III -- implied do loop, nested array expressions
C     This example is very similar to arrays-constr-06.f with the
C     difference that the iteration for the inner array expression goes
C     from 12 down to 1 (start value = 12, end value = 1, step = -1).
C     The inner expression B(12:1:-1) therefore gives the sequence of
C     values of B's elements in the order B(12), B(11), ..., B(1), i.e.:
C
C         1, 2, 3, ..., 10, 11, 12
C
C     The outer expression uses this sequence of values to index into
C     the array C, giving the sequence of values of C's elements in the
C     order C(1), C(2), C(3), ..., C(10), C(11), C(12).  The sequence of
C     values assigned to the elements of A is therefore:
C
C         1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23

      A = (/ (C(B(12:1:-1))) /)

      do i = 1, 12
          write (*,10) A(i)
      end do

 10   format(I5)

      stop
      end program main
