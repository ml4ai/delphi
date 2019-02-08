C File: derived-types-01.f
C Illustrates simple user-defined types in Fortran

      program main
      implicit none

      type mytype
      integer :: k
      real :: v
      end type mytype

      type (mytype) x, y;

      x%k = 12
      x%v = 3.456

      y%k = 21
      y%v = 4.567

 10   format (2(I5, X, F6.3))
      write (*, 10) x%k, y%v, y%k, x%v

      stop
      end program main
      

      
