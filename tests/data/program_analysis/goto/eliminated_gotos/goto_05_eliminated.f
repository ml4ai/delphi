C     File: goto_05.f
C     A simple program with a single forward conditional goto and
C     a single backward conditional goto.
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
      fact = 0
      
      label_flag_1 = .true.
      do while (label_flag_1)
          i = i + 1
          if (fact .eq. 0) then
              fact = fact + 1    ! this line is executed exactly once
          endif

          fact = fact * i
          write (*, 10) i, fact

          label_flag_1 = i .lt. n
      enddo

      stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
