C FORTRAN test file to test CASE constructs
C This file tests an integer with various types of comparisons

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: I = 5
      INTEGER :: X = 40
      INTEGER :: Y
      INTEGER :: Z = 2

      SELECT CASE(I)
        CASE(:3)
          Y = X/4
          WRITE(*,*) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(9:)
          Y = X/10
          WRITE(*,*) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(8)
          Y = X/2
          WRITE(*,*) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(4:7)
          Y = X/8
          WRITE(*,*) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE DEFAULT
          WRITE(*,*) 'Invalid Argument!'

      END SELECT

      END PROGRAM MAIN
