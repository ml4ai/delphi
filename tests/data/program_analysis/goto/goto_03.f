C     File: goto_03.f
C     A simple program with a single top-level backward conditional goto.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact

      i = 0
      fact = 1
      n = 10
      
 111  i = i + 1
      fact = fact * i

      write (*, 10) i, fact

      if (i .lt. n) goto 111

      stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
