C     File: goto_07.f
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

      logical :: goto_222, goto_444
C     For label 111 of first goto 111 appeared
      logical :: label_flag_1
C     For label 333
      logical :: label_flag_2
C     For label 111 of second goto 111 appeared
      logical :: label_flag_3
      goto_222 = .true.
      goto_444 = .true.
      label_flag_1 = .true.
      label_flag_2 = .true.
      label_flag_3 = .true.

      do while (label_flag_3)
          if (.not. goto_222) then
              fact = 1
              do while (label_flag_2)
                  if (.not. goto_444) then 
                      do while (label_flag_1)
                          i = i + 1
                          fact = fact * i

                          write (*, 10) i, fact
                          if (i .eq. n) then
                             stop
                          endif

                          label_flag_1 = .true.
                      enddo

                      label_flag_2 = .true.
                  endif

                  if (goto_444) then
                      i = 0
                      goto_444 = .false.
                  endif
              enddo
          endif

          if (goto_222) then
              n = 10
              goto_222 = .false.
          endif

          label_flag_3 = .true.
      enddo

 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
