C File: array2d-slice-07.f
C Illustrates array slicing

      program main
      implicit none

      integer, dimension (5,5) :: A
      integer, dimension(3) :: K = (/ 1, 3, 5 /)
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

C      Modifying elements in the slice of A defined by K
       A(2, K ) = 999

      do i = 1, 5
          write (*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5)
      end do
      write(*,11)

C     Modifying K
      do i = 1, 3
          K(i) = (K(i)+3)/2
      end do 

C      Modifying elements in the slice of A defined by K
       A(4, K ) = -1

      do i = 1, 5
          write (*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5)
      end do

      stop
      end program main

