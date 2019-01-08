C File: arrays-sect-12.f
C Illustrates the use of two-dimensional array sections.

      program main
      implicit none

      integer, dimension (4,6) :: A
      integer, dimension(3) :: C = (/1,3,4/)
      integer :: i, J

      do i = 1, 4
         do j = 1, 6
            A(i,j) = i*i+j*j
         end do
      end do

      do i = 1,4
         write (*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5), A(i,6)
      end do
      write(*,11)

 10   format("BEFORE: ", 6(I5,X))
 11   format('')
 12   format("AFTER:  ", 6(I5,X))

C     A two-dimensional array section.  The elements selected are:
C
C        A(1,1), A(1,3), A(1,4)
C        A(4,1), A(4,3), A(4,4)
      
      A((/1,4/),C) = -1

      do i = 1,4
         write (*,12) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5), A(i,6)
      end do

      stop
      end program main

