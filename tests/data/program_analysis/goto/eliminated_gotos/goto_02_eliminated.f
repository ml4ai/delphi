C     File: goto_02.f
C     A simple program with multiple gotos at the top level of the program.
C     The program computes and prints out the values of n! for n in [1,10].
C
C     GOTO-elimination algorithm credit:
C     Title: Taming control flow: a structured approach to eliminating goto statements
C     Author: Ana M. Erosa and Laurie J. Hendren
C     URL: https://ieeexplore.ieee.org/abstract/document/288377

      program factorial
      implicit none

      integer i, n, fact

      logical :: goto_222
      logical :: label_flag_1, label_flag_2
      goto_222 = .true.
      label_flag_1 = .true.
      label_flag_2 = .true.
    
      do while (label_flag_2) 
          if (.not. goto_222) then
              do while (label_flag_1)
                  i = i + 1
                  fact = fact * i

                  write (*, 10) i, fact
                  if (i .eq. n) then
                     stop
                  endif
                  
                  label_flag_1 = .true.
              enddo
          endif

          n = 10
          fact = 1
          i = 0
          goto_222 = .false.

          label_flag_2 = .true.
      enddo      

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
