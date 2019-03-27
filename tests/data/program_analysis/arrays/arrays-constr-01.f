C File: arrays-constr-01.f
C This program has a simple 1-D array with the default lower bound of 1.
C It shows the use of an array constructor to initialize the array.

      program main
      implicit none

C     arr is a 1-D array of integers with an implicit lower-bound = 1
C     and an upper bound of 10.  
      integer, dimension(10) :: arr
      integer :: i

C     array constructor I -- scalar expression
      arr = (/11,22,33,44,55,66,77,88,99,110/)

      do i = 1, 10
          write (*,10) arr(i)
      end do

 10   format(I5)

      stop
      end program main
