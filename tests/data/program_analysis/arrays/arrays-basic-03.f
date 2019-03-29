C File: arrays-basic-03.f
C This program has a simple 1-D array with the default lower bound of 1
C that is accessed in a non-sequential order using a second array.

      program main
      implicit none

C     array and idx are 1-D arrays of integers with an implicit 
C     lower-bound = 1 upper bound = 10
      integer, dimension(10) :: array, idx 
      integer :: i

      do i = 1, 10
          array(i) = i*i    ! array() holds the values
      end do

      do i = 1, 5
          idx(i) = 2*i      ! idx() is used to index into array()
          idx(i+5) = 2*i-1
      end do

      do i = 1, 10
          write (*,10) array(idx(i))
      end do

 10   format(I5)

      stop
      end program main
