C File: test_module_02.f
C Purpose: Illustrates a Fortran module that defines values for several variables
C    that are used in the program that uses this module.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_02.f     # << this will create a file "mymod2.mod"
C    gfortran test_module_02.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C        1234           2   3.14159989       7753.46875

      MODULE MYMOD2
      IMPLICIT NONE

      INTEGER :: X = 1234, Y = 2
      REAL :: PI = 3.1416
      END MODULE mymod2

      PROGRAM PGM
      USE mymod2      ! << case-insensitive

      write (*,*) X, Y, PI, X*Y*PI

      stop
      end program PGM
