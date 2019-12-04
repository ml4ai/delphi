C FORTRAN test file to test CASE constructs
C This file tests one-to-one comparison with characters

      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER*1 :: VAR = 'B'
      INTEGER :: X = 40
      INTEGER :: Y
      INTEGER :: Z = 2

      SELECT CASE(VAR)
        CASE('A')
          Y = X/4
         WRITE(*,10) 'The variable is A and Y is: ', Y, Y*Z
        CASE('G')
          Y = X/10
          WRITE(*,10) 'The variable is G and Y is: ', Y, Y*Z
        CASE DEFAULT
          WRITE(*,20) 'Invalid Argument!'

      END SELECT

 10   format(A, I2, I4)
 20   format(A)

      END PROGRAM MAIN
