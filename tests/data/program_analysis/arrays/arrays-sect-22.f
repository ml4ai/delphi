C File: arrays-sect-22.f
C Illustrates array slicing

      program main
      implicit none

      integer, dimension (5,-2:2) :: A
      integer, dimension (5) :: B = (/-95,1,-23,45,3/)
      integer :: i, j

      A = -1

      do i = 1, 5
          write (*,10) A(i,-2), A(i,-1), A(i,0), A(i,1), A(i,2)
      end do
      write(*,11)

 10   format(5(I5,X))
 11   format('')

C      Modifying elements in two slices of A
       A(B(2::3), ::2) = 555	! A(1,-2)  A(1,0)  A(1,2), A(3,-2), A(3,0), A(3,2)

       do i = 1, 5
          write (*,10) A(i,-2), A(i,-1), A(i,0), A(i,1), A(i,2)
      end do

      stop
      end program main

