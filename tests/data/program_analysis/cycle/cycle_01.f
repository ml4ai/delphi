C   cycle_01.f
C   A simple example of  cycle usage within a loop.
C   Output: i from 1 to 20 except 5.
C   Source: https://www.tutorialspoint.com/fortran/fortran_cycle.htm

      program cycle_example     
      implicit none      

        integer :: i

        do i = 1, 20          
       
            if (i == 5) then
                cycle
            end if         
            
            write (*, 10) i
        end do  
       
 10   format('i = ', I3)
      end program cycle_example
