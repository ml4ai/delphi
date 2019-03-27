C File: arrays-basic-07.f
C Illustrates: transposition of a 2-D matrix

      program main
      implicit none

      integer, dimension(3,5) :: A
      integer, dimension(5,3) :: B
      integer :: i, j

C     Initialize matrix A
      do i = 1,3
          do j = 1,5
              A(i,j) = (i*j)+(i+j)
          end do
      end do

C     Transpose A into B
      do i = 1,3
          do j = 1,5
              B(j,i) = A(i,1)
          end do
      end do

      stop
      end program main
