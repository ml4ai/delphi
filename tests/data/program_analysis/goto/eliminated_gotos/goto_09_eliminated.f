C     File: goto_08.f
C     A program with a single forward unconditional goto.
      
      program factorial
      implicit none

      integer i, n, fact
      logical goto_flag_222

      i = 0
      n = 10
      fact = 0

      goto_flag_222 = .true.

      do i = 1, n
         if (i .lt. 20) then
             if (i .eq. 1) then
                fact = fact + 1
             elseif (i .le. 10) then
                fact = fact * i
             else
                exit
             end if
             write (*, 10) i, fact
         end if
      end do

      if (goto_flag_222) then
         stop
      end if
 10   format('i = ', I3, '; fact = ', I8)
      end program factorial
