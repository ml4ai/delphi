C Testing SAVE statements.  This program has two different subroutines
C both of which contain SAVE statements and both of which contain a
C local variable W whose value needs to be SAVEd.  This addresses the
C issue of potential name conflicts between SAVEd locals in different
C subprograms.

      subroutine f(n, x)

      implicit none
      integer n, x
      integer, dimension(1) :: w

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
          w(1) = 111
      else
          w(1) = 2*w(1)
      end if

      x = w(1)
      end subroutine f
C ********************************************************************
      subroutine g(n, x)

      implicit none
      integer n, x
      integer, dimension(10) :: w

      save

      ! If n == 0, set w to 999; else set w to w/3.  Then set x to w.
      ! The "save" statement causes the value of w to persist across
      ! calls to g().  This means that the sequence of calls
      !
      !    call g(0, a)
      !    call g(1, a)
      !    call g(1, a)
      !
      ! will set a to the values 999, 333, and 111 respectively.

      if (n .eq. 0) then
          w(1) = 999
      else
          w(1) = w(1)/3
      end if

      x = w(1)
      end subroutine g
C ********************************************************************
      program main
      implicit none
      integer a, b

      call f(0, a)
      call g(0, b)

 10   format("a = ", I5, ";   b = ", I5)
      write (*,10) a, b

      call f(1, a)
      call g(1, b)
      write (*,10) a, b

      call f(1, a)
      call g(1, b)
      write (*,10) a, b

      stop
      end program main
