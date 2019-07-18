!     This program tests the handling of line continuations in the preprocessor.
!     This file uses Fortran-90 continuation syntax.

      subroutine myadd(	&
          arg1,	        &
          arg2,         &
          arg3)
      integer arg1, arg2, arg3

      arg3 = arg1 + arg2
      return
      end subroutine myadd

      program main

      implicit none
      integer x, y, z
      x                &
        =              &
          12
      
      y                &
        =              &  
          13

      call             &
          myadd(       &
          x,           &
          y,           &
      z)

      write (*,10) x, y, z
 10   format(3(X,I3))


      END program main
