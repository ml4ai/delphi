C     This program tests initialization of variables
      program main
      implicit none

      integer :: x = 12345
      real :: y = 3.1416

      write (*,10) x, y
 10   format (I6,X,f8.4)
      stop
      end program main
 
