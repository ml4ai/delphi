C     File: goto_06.f
C     A simple program with a single forward conditional goto and
C     a single backward unconditional goto.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact

      i = 0
      n = 10
      fact = 1

 111  i = i + 1

      if (i .le. n) goto 222

      stop

 222  fact = fact * i

      write (*, 10) i, fact

      goto 111

 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
