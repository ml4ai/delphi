C     cycle_02.f
C     A simple example of cycle with break within the loop.
C     Output: i = 1...19 k = 19
C     Source: http://annefou.github.io/Fortran/basics/control.html

      program odd_number
      implicit none
      integer :: N, k
      N = 19
      k = 0
      DO WHILE (.TRUE.)
        k = k + 1
        if (k > N) EXIT
        if (mod(k,2) .eq. 0) CYCLE
        WRITE(*,10) k, N
      ENDDO

 10   FORMAT('k = ', I3, '; N = ', I8)
      end program odd_number
