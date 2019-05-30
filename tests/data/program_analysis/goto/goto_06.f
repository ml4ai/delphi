C     File: goto_06.f
C     A simple program with a goto within an IF statement.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, n, fact
      n = 10
      fact = 1
      i = 0
      
 111  if (i .lt. n) then
         i = i + 1
         goto 222
         
         i = i * 234    ! this stmt is never executed
         
 222     fact = fact * i
         write (*, 10) i, fact
      else
         stop
      endif

      goto 111

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
