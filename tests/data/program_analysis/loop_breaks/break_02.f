C     exit_test.f
C     A simple example of cycle with break within the loop.

      program odd_number
      implicit none
      integer :: n, k, day, x
      n = 19
      k = 0
      do 20 day=1,31
        k = k + 1
        if (k > n) exit
        x = k*2
        write(*,10) k, n, x
 20   enddo

 10   format('k = ', i3, '; n = ', i3, '; x = ', i4)
      end program odd_number
