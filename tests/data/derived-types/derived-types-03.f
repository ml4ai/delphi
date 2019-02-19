C File: derived-types-03.f
C This file tests multiple adjacent derived type declarations
C The output produced by this program is:
C  123     4.560
C  246    13.680

      program main
      implicit none

      type mytype_1
          integer:: a
          real :: b
      end type mytype_1

      type mytype_2
          integer:: a
          real :: b
      end type mytype_2

      type (mytype_1) x
      type (mytype_2) y

      x % a = 123
      x % b = 4.56

      y % a = x % a * 2
      y % b = x % b * 3

 10   format(I5,3X, F7.3)
      write (*, 10) x%a, x%b
      write (*, 10) y%a, y%b

      stop
      end program main
