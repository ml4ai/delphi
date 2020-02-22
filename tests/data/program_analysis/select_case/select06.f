      PROGRAM MAIN

      CHARACTER(LEN=1) :: c = 'k'
      CHARACTER(LEN=1), PARAMETER :: lessVar = 'd'
      CHARACTER(LEN=1), PARAMETER :: greatVar = 'm'
      CHARACTER(LEN=1), PARAMETER :: equalVar = 'k'

      SELECT CASE (c)
        CASE (equalVar)
          WRITE(*,10)  'Equal to testVar'
        CASE (:lessVar)
          WRITE(*,10)  'Less than testVar'
        CASE (greatVar:)
          WRITE(*,10)  'Greater than testVar'
        CASE DEFAULT
          WRITE(*,10)  'Wut'
      END SELECT

 10   FORMAT(A)

      END PROGRAM MAIN
