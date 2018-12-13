C File: test_module_03.f
C Purpose: Illustrates the use of multiple Fortran modules to define values 
C    for several variables that are used in the program.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_03.f     # << this will create files "mymod3.mod"
C                                                                 "mymod4.mod"
C                                                                 "mymod5.mod"
C    gfortran test_module_03.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C        1234           2   3.14159989       7753.46875

      MODULE MYMOD3
      IMPLICIT NONE

      INTEGER :: X = 1234
      END MODULE mymod3

      MODULE MYMOD4
      IMPLICIT NONE

      INTEGER :: Y = 2
      END MODULE mymod4

      MODULE MYMOD5
      IMPLICIT NONE

      REAL :: PI = 3.1416
      END MODULE mymod5

      PROGRAM PGM
      USE mymod3
      USE mymod4
      USE mymod5

      write (*,*) X, Y, PI, X*Y*PI

      stop
      end program PGM
