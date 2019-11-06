C     File: goto_07.f
C     A simple program with multiple gotos at the top level of the program.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact

      goto 222
      
 333  fact = 1
      goto 444
      
 111  i = i + 1
      fact = fact * i

      write (*, 10) i, fact
      if (i .eq. n) then
         stop
      endif

      goto 111

 222  n = 10
      goto 333

 444  i = 0
      goto 111

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
