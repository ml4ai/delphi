      PROGRAM NESTED_LOOP      
      IMPLICIT NONE

      INTEGER MONTH, DAY

      MONTH = 1

      DO WHILE (MONTH <= 12)
        DAY = 1
        DO WHILE (DAY <= 7)
          PRINT *, "Month: ", MONTH, " DAY: ", DAY
          DAY = DAY + 1
        ENDDO
        MONTH = MONTH + 1
      ENDDO 

      END PROGRAM NESTED_LOOP
