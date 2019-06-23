C     File: goto_04.f
C     A simple program with a single forward conditional goto.
C     The program computes and prints out the values of n! for n in [1,10].
      
      program factorial
      implicit none

      integer i, n, fact

      i = 0
      n = 10
      fact = 0

      do i = 1, n
         if (fact .ne. 0) goto 222

         fact = fact + 1    ! this line is executed exactly once

 222     fact = fact * i

         write (*, 10) i, fact
      end do

      stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
