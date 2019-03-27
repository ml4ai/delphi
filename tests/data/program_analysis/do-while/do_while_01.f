      PROGRAM DO_WHILE
      IMPLICIT NONE

      INTEGER MONTH

      MONTH = 1

      DO WHILE (MONTH <= 12)
          PRINT *, "Month: ", MONTH
          MONTH = MONTH + 1
      ENDDO

      END PROGRAM DO_WHILE
