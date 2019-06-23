C     File: goto_05.f
C     A simple program with a single forward conditional goto and
C     a single backward conditional goto.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact

      i = 0
      n = 10
      fact = 0

 111  i = i + 1
      if (fact .ne. 0) goto 222

      fact = fact + 1    ! this line is executed exactly once

 222  fact = fact * i

      write (*, 10) i, fact

      if (i .lt. n) goto 111

      stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
