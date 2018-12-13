C File: test_module_01.f
C Purpose: Illustrates a Fortran module that defines a value for a variable
C    that is used in the program that uses this module.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_01.f     # << this will create a file "mymod1.mod"
C    gfortran test_module_01.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C      1234567

      MODULE MYMOD1
      IMPLICIT NONE

      INTEGER :: X = 1234567
      END MODULE mymod1

      PROGRAM PGM
      USE mymod1      ! << case-insensitive

      write (*,*) X

      stop
      end program PGM
