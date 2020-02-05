      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER(5) :: ModuleName = "MGMT"
      CHARACTER(6) :: VarName = "TOTIR"
      INTEGER Value
      Logical ERR

      Value = 0
      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('SPAM')
        SELECT CASE (VarName)
        Case ('PG');     Value = 1
        Case ('CEF');    Value = 2
        Case ('CEM');    Value = 3
        Case ('CEO');    Value = 4
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('BIOMAS'); Value = 5
        Case ('CANHT') ; Value = 6
        Case ('CANWH') ; Value = 7
        Case ('DXR57') ; Value = 8
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('MGMT')
        SELECT CASE (VarName)
        Case ('EFFIRR'); Value = 9
        Case ('TOTIR');  Value = 10
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case DEFAULT; ERR = .TRUE.
      END SELECT

      WRITE(*,10) Value

 10   FORMAT(I2)

      END PROGRAM MAIN