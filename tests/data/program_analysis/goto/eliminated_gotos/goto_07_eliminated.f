C     File: goto_07.f
C     A simple program with a single forward conditional goto and
C     a single backward unconditional goto.
C     The program computes and prints out the values of n! for n in [1,10].
C
C     GOTO-elimination algorithm credit:
C     Title: Taming control flow: a structured approach to eliminating goto statements
C     Author: Ana M. Erosa and Laurie J. Hendren
C     URL: https://ieeexplore.ieee.org/abstract/document/288377

      program factorial
      implicit none

      integer i, n, fact
      logical label_flag_1

      i = 0
      n = 10
      fact = 1

      label_flag_1 = .true.
      do while (label_flag_1)
          i = i + 1

C         Negate the condition and move all statements
C         until before label 222 statement
          if (i .le. n) then        
              fact = fact * i       

              write (*, 10) i, fact

              label_flag_1 = .true.
          endif

C         Move the label 222 statement to under the if
C         statement with original condition
          if (i .gt. n) then 
              stop 
          endif
      enddo

 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
