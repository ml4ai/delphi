C File: arrays-basic-06.f
C This program has a 3-D array with mixed explicit and implicit lower bounds.

      program main
      implicit none
      save

C     A is a 3-D array (5x5x5) of integers with mixed explicit and implicit
C     lower-bounds.
      integer, dimension(-3:1,5,10:14) :: A
      integer :: i, j, k

      do i = -3, 1
        do j = 1,5
          do k = 10, 14
            A(i,j,k) = i+j+k     ! initialize the array
          end do
        end do
      end do

      ! write out the array
      do i = -3, 1
        do j = 1,5
          write (*,10) 
     &       A(i,j,10), A(i,j,11), A(i,j,12), A(i,j,13), A(i,j,14)
        end do
      end do

 10   format(5(I5,X))

      stop
      end program main
