C File: arrays-sect-21.f
C Illustrates array slicing

      program main
      implicit none

      integer, dimension (5,-2:2) :: A
      integer :: i, j

      A = -1

      do i = 1, 5
          write (*,10) A(i,-2), A(i,-1), A(i,0), A(i,1), A(i,2)
      end do
      write(*,11)

 10   format(5(I5,X))
 11   format('')

C      Modifying elements in two slices of A
       A( 3, ::2) = 555	! A(2,-2)  A(2,0)  A(2,2), A(4,-2), A(4,0), A(4,2)

       do i = 1, 5
          write (*,10) A(i,-2), A(i,-1), A(i,0), A(i,1), A(i,2)
      end do

      stop
      end program main

