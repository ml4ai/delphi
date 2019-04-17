C File: derived-types-04.f
C This program uses a derived type whose fields are themselves derived types.
C The output produced by this program is:
C  123     12   34
C  456     21   45


      program main
      implicit none

      type mytype_123
          integer :: ctr = 123
          integer :: a, b
          integer :: gta = 987
      end type mytype_123

      type mytype_456
          integer :: ctr = 456
          integer :: c, d
      end type mytype_456

      type mytype_123_456
          type (mytype_123) x
          type (mytype_456) y
      end type mytype_123_456

      type (mytype_123_456) var

      var % x % a % j = 12
      var % y % c = 21

      var % x % b = 34
      var % y % d = 45

 10   format (3(I5,2X))
      write (*,10) var%x%ctr, var%x%a, var%x%b
      write (*,10) var%y%ctr, var%y%c, var%y%d

      stop
      end program main
      
