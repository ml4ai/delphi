C     File: goto_07.f
C     A simple program with a goto within a DO loop.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      integer i, j, k, n, fact
      n = 10
      j = 27
      fact = 1
      
      do i = 1, n
         goto 111
         
         j = 33                 ! this stmt is never executed
         
 111     k = i+j-27
         fact = fact * k
         write (*, 10) i, fact
      end do

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
