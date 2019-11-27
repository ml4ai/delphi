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
          WRITE(*,*) 'The variable is A and Y is: ', Y, Y*Z
        CASE('G')
          Y = X/10
          WRITE(*,*) 'The variable is G and Y is: ', Y, Y*Z
        CASE DEFAULT
          WRITE(*,*) 'Invalid Argument!'

      END SELECT

      END PROGRAM MAIN
