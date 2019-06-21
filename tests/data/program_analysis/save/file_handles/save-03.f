C Testing SAVE statements.  This program has two different subroutines
C both of which contain SAVE statements and both of which contain a
C local variable W whose value needs to be SAVEd.  This addresses the
C issue of potential name conflicts between SAVEd locals in different
C subprograms.

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
      subroutine g(n, x)

      implicit none
      integer n, x, w

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
          w = 999
      else
          w = w/3
      end if

      x = w
      end subroutine g
C ********************************************************************
      integer function h(flag)

      implicit none
      integer a, b, retval
      logical flag

      if (flag .eqv. .TRUE.) then        ! initialize
         open (2, file = 'INPUT-FILE.txt')
         retval = 0
      else
         read (2, 20) retval
 20      format(I2)
      end if

      h = retval

      end function h
C ********************************************************************
      program main
      implicit none
      integer a, b, status, h

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

      status = h(.true.)
      print *, "First status: ", status

      status = h(.false.)
      print *, "Second status: ", status

      stop
      end program main
