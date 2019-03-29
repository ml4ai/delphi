C File: arrays-constr-05.f
C This program has a simple 1-D array with the default lower bound of 1.
C It shows the use of an array constructor to initialize the array.

      program main
      implicit none

C     arr is a 1-D array of integers with an implicit lower-bound = 1
C     and an upper bound of 12.  
      integer, dimension(12) :: arr 
      integer, dimension(7) :: idx = (/37, 43, 59, 67, 73, 79, 83/)
      integer :: i

C     array constructor III -- implied do loop, multiple expressions
C     computed in each loop iteration.
C     In this example, the iteration variable i takes on values 1, 3, 5, 7
C     (initial value = 1, final value = 7, stride = 2).  The list of
C     values on the RHS of the assignment consists of the values of the
C     expressions given -- i.e.: 7*i, 11*i-1, idx(i) -- for each such
C     value of i:
C
C         i = 1  ==>   7, 10, 37
C         i = 3  ==>  21, 32, 59
C         i = 5  ==>  35, 54, 73
C         i = 7  ==>  49, 76, 83
C
C     So the sequence of values assigned to the elements of arr is:
C
C         7, 10, 37, 21, 32, 59, 35, 54, 73, 49, 76, 83

      arr = (/ (7*i, 11*i-1,  idx(i), i = 1, 7, 2) /)

      do i = 1, 12
          write (*,10) arr(i)
      end do

 10   format(I5)

      stop
      end program main
