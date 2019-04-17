      PROGRAM DO_WHILE
      IMPLICIT NONE

      INTEGER MONTH

      MONTH = 1

      DO WHILE (MONTH <= 13)
          IF (MONTH == 13)  EXIT
          PRINT *, "Month: ", MONTH
          MONTH = MONTH + 1
      ENDDO

      END PROGRAM DO_WHILE
