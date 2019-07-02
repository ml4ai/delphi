C     File: goto_03.f
C     A simple program with a single top-level backward conditional goto.
C     The program computes and prints out the values of n! for n in [1,10].

      program factorial
      implicit none

      logical fact
      fact = .true.

      if (fact) then
          print *, "TRUE"
      endif

      end program factorial
