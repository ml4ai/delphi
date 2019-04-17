      PROGRAM TRIPLE_NESTED
      IMPLICIT NONE

      INTEGER MONTH, DATE, DAY

      MONTH = 1

      DO WHILE (MONTH <= 12)
        DATE = 1
        DO WHILE (DATE <= 7)
          DAY = 1
          DO WHILE (DAY <= DATE)
            IF (DAY == 1) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", SUNDAY"
            ELSE IF (DAY == 2) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", MONDAY"
            ELSE IF (DAY == 3) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", TUESDAY"
            ELSE IF (DAY == 4) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", WEDNESDAY"
            ELSE IF (DAY == 5) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", THURSDAY"
            ELSE IF (DAY == 6) THEN
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", FRIDAY"
            ELSE
              PRINt *, "MONTH: ", MONTH, " DAY: ", DAY, ", SATURDAY"
            END IF
            DAY = DAY + 1
          END DO
          DATE = DATE +1
        END DO
        MONTH = MONTH + 1
      END DO
      END PROGRAM TRIPLE_NESTED
