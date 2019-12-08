      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: Class = 8

      SELECT CASE (Class)
        CASE (1)
          WRITE(*,*)  'Freshman'
        CASE (2)
          WRITE(*,*)  'Sophomore'
        CASE (3)
          WRITE(*,*)  'Junior'
        CASE (4)
          WRITE(*,*)  'Senior'
        CASE DEFAULT
          WRITE(*,*)  "Hmmmm, I dont know"
      END SELECT
      WRITE(*,*)  'Done'

      END PROGRAM MAIN