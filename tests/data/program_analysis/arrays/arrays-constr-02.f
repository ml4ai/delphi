C File: arrays-constr-02.f
C This program has a simple 1-D array with the default lower bound of 1.
C It shows the use of an array constructor to initialize the array.

      program main
      implicit none

C     arr and X are 1-D arrays of integers with an implicit lower-bound = 1
C     and upper bound of 10.  
      integer, dimension(10) :: X = (/11,22,33,44,55,66,77,88,99,110/)
      integer, dimension(10) :: arr 
      integer :: i

C     array constructor I -- array expression
      arr = (/ 10, 20, X(3:8), 90, 100/)

      do i = 1, 10
          write (*,10) arr(i)
      end do

 10   format(I5)

      stop
      end program main
