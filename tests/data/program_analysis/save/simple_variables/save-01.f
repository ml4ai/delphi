C Testing SAVE statements

      subroutine f(n, x)

      implicit none
      integer n, x, w

      save

      ! If n == 0, set w to 111; else set w to 2*w.  Then set x to w.
      ! The "save" statement causes the value of w to persist across
      ! calls to f().  This means that the sequence of calls
      !
      !    call f(0, a)
      !    call f(1, a)
      !    call f(1, a)
      !
      ! will set a to the values 111, 222, and 444 respectively.
      
      if (n .eq. 0) then
          w = 111
      else
          w = 2*w
      end if

      x = w
      end subroutine f
C ********************************************************************
      program main
      implicit none
      integer a

      call f(0, a)

 10   format("a = ", I5)
      write (*,10) a

      call f(1, a)
      write (*,10) a

      call f(1, a)
      write (*,10) a

      stop
      end program main
