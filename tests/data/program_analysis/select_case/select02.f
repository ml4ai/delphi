C FORTRAN test file to test CASE constructs
C This file tests an integer with various types of comparisons

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: INC
      INTEGER :: Y

      DO 10 INC=1,10

        SELECT CASE(INC)

          CASE(:3)
            Y = INC*2
            WRITE(*,20) 'The variables I and Y have values: ', INC, Y
          CASE(9:)
            Y = INC*3
            WRITE(*,20) 'The variables I and Y have values: ', INC, Y
          CASE(8)
            Y = INC*4
            WRITE(*,20) 'The variables I and Y have values: ', INC, Y
          CASE(4:7)
            Y = INC*5
            WRITE(*,20) 'The variables I and Y have values: ', INC, Y
          CASE DEFAULT
            WRITE(*,30) 'Invalid Argument!'

        END SELECT

 10   END DO

 20   format(A, I2, I4)
 30   format(A)

      END PROGRAM MAIN
