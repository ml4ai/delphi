C     File: goto_08.f
C     A simple program with a single forward conditional goto with 1-level
C     difference in goto and label.
C     The program computes and prints out the values of n! for n in [1,10].
      
      program factorial
      implicit none

      integer i, n, fact

      i = 0
      n = 11
      fact = 0

      do i = 1, n
         if (fact .eq. 0) then
             fact = fact + 1    ! this line is executed exactly once
         endif

         fact = fact * i

         write (*, 10) i, fact

         if (i .ge. 10) goto 222
      end do

 222  stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
