C FORTRAN test file to test CASE constructs
C This file tests an integer with a list of 'or' test cases in a single case

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: Y
      INTEGER :: INC

      DO 20 INC=1,10

        SELECT CASE(INC)

          CASE(:2,5,9:)
            Y = INC*2
            WRITE(*,30) 'The variables INC and Y have values: ', INC, Y
          CASE(3,4,6:8)
            Y = INC*3
            WRITE(*,30) 'The variables INC and Y have values: ', INC, Y

        END SELECT

 20   END DO

 30   FORMAT(A, I2, I4)

      END PROGRAM MAIN
