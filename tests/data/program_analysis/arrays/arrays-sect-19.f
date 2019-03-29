C File: arrays-sect-19.f
C Illustrates array slicing

      program main
      implicit none

      integer, dimension (5,5) :: A
      integer :: i, j

      do i = 1, 5
          do j = 1, 5
              A(i,j) = 11*(i+j)
          end do
      end do

      do i = 1, 5
          write (*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5)
      end do
      write(*,11)

 10   format(5(I5,X))
 11   format('')

C      Modifying elements in a sub-array of A
       A( :, 3:5 ) = 999

      do i = 1, 5
          write (*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5)
      end do

      stop
      end program main

