C FORTRAN test file to test CASE constructs
C This file tests an integer with a list of 'or' test cases in a single case

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: I = 5
      INTEGER :: X = 40
      INTEGER :: Y
      INTEGER :: Z = 2

      SELECT CASE(I)
        CASE(:2,5,9:)
          Y = X/4
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(3,4,6:8)
          Y = X/10
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z

      END SELECT

 10   format(A, I2, I2, I4)

      END PROGRAM MAIN
