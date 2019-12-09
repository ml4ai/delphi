      PROGRAM MAIN
      IMPLICIT NONE

      LOGICAL :: COND_1 = .FALSE.
      LOGICAL, PARAMETER :: COND_2 = .FALSE.

      SELECT CASE (COND_1)
        CASE (COND_2)
          WRITE (*,10) 'Logical False'
        CASE DEFAULT
            WRITE (*,10) 'Logical True'
      END SELECT

 10   FORMAT(A)

      END PROGRAM MAIN