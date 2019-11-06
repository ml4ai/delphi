      program simple_test
      implicit none
      integer, PARAMETER :: limit = 5
      integer            :: x

      x = 0
      do while (.true.)
         IF (x == limit)  EXIT
         WRITE(*,10)  x
         x = x + 1
      ENDDO

 10   format('x = ', i3)
      end program simple_test