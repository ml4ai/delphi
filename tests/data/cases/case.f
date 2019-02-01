C FORTRAN test file to test CASE constructs

      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER*1 :: VAR = 'A'
      INTEGER :: X = 40
      INTEGER :: Y

      SELECT CASE(VAR)
        CASE('A')
          Y = X/4
          WRITE(*,*) 'The variable is A and Y is: ', Y
        CASE('G')
          Y = X/10
          WRITE(*,*) 'The variable is G and Y is: ', Y

      END SELECT

      END PROGRAM MAIN
