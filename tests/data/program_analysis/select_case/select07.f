      program f
      implicit none
      integer i
      integer, parameter :: Inf = 5
      do i = 1, 10
         select case (i)
         case (:Inf)
            write (*,10) i

         case (Inf+1:)
            write (*,10) -i
         end select
      end do

 10   format(I4)
      end program f