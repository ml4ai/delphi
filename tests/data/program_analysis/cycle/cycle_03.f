C     cycle_02.f
C     A simple example of multiple cycles with break within the loop.
C     Output: i = 1...19 (Only odd numbers) k = 19
C     Source: http://annefou.github.io/Fortran/basics/control.html

      program odd_number
      implicit none
      integer :: n, k
      n = 19
      k = 0
      do while (.true.)
        k = k + 1
        if (k .eq. 5) cycle
        if (k > n) exit
        if (mod(k,2) .eq. 0) cycle
        write(*,10) k, n
      enddo

 10   format('k = ', i3, '; n = ', i8)
      end program odd_number
