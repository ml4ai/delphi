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
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(9:)
          Y = X/10
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(8)
          Y = X/2
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE(4:7)
          Y = X/8
          WRITE(*,10) 'The variable is I, A, and Y are: ', I, Y, Y*Z
        CASE DEFAULT
          WRITE(*,20) 'Invalid Argument!'

      END SELECT

 10   format(A, I2, I2, I4)
 20   format(A)

      END PROGRAM MAIN
