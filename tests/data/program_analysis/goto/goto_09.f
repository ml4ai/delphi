C     File: goto_09.f
      
      program factorial
      implicit none

      integer i, n, fact

      i = 0
      n = 10
      fact = 0

      do i = 1, n
         if (i .lt. 20) then
             if (i .eq. 1) then
                fact = fact + 1
             elseif (i .le. 10) then
                fact = fact * i
             else
                goto 222
             end if
             write (*, 10) i, fact
         end if
      end do

 222  stop
 10   format('i = ', I3, '; fact = ', I8)

      end program factorial
