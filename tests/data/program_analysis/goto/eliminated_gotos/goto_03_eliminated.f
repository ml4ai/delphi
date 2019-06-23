C     File: goto_03.f
C     A simple program with a single top-level backward conditional goto.
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
      fact = 1
      n = 10

      label_flag_1 = .true.    
      do while (label_flag_1)
          i = i + 1
          fact = fact * i

          write (*, 10) i, fact
          
          label_flag_1 = i .lt. n   ! Using a boolean expression directly instead of if-statement
      enddo

      stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
