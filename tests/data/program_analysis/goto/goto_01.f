C     File: goto_01.f
C     A simple program with a single goto at the top level of the program.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact
      n = 10
      fact = 1
      i = 0
      
 111  i = i + 1
      fact = fact * i

      write (*, 10) i, fact
      if (i .eq. n) then
         stop
      endif

      goto 111

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
