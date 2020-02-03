      MODULE TestModule

      INTEGER, PARAMETER :: MaxFiles = 500

      TYPE ControlType
        CHARACTER (len=1)  MESIC
        CHARACTER (len=2)  CROP
        CHARACTER (len=8)  MODEL
        CHARACTER (len=12) FILEX
        CHARACTER (len=30) FILEIO
        CHARACTER (len=102)DSSATP
        INTEGER   DAS
        INTEGER   NYRS
        INTEGER   YRDIF
      END TYPE ControlType

      TYPE SwitchType
        CHARACTER (len=1) FNAME
        CHARACTER (len=1) IDETC
        CHARACTER (len=1) IDETO
        CHARACTER (len=1) IHARI
        CHARACTER (len=1) ISWCHE
        CHARACTER (len=1) ISWPHO
        CHARACTER (len=1) MEEVP
        CHARACTER (len=1) MESOM
        CHARACTER (len=1) METMP !Temperature, EPIC
        CHARACTER (len=1) IFERI
        INTEGER NSWI
      END TYPE SwitchType

!     Data which can be transferred between modules
      Type TransferType
        Type (ControlType) CONTROL
        Type (SwitchType)  ISWITCH
        Type (OutputType)  OUTPUT
        Type (PlantType)   PLANT
        Type (MgmtType)    MGMT
        Type (NiType)      NITR
        Type (OrgCType)    ORGC
        Type (SoilType)    SOILPROP
        Type (SPAMType)    SPAM
        Type (WatType)     WATER
        Type (WeathType)   WEATHER
        TYPE (PDLABETATYPE) PDLABETA
      End Type TransferType
      
!     The variable SAVE_data contains all of the components to be 
!     stored and retrieved.
      Type (TransferType) SAVE_data

      INTERFACE GET
        MODULE PROCEDURE GET_CONTROL
     &                 , GET_ISWITCH 
      END INTERFACE

      CONTAINS

      Subroutine GET_CONTROL (CONTROL_arg)
!     Retrieves CONTROL variable
      IMPLICIT NONE
      Type (ControlType) CONTROL_arg
      Control_arg = SAVE_data % Control
      Return
      End Subroutine GET_CONTROL

      Subroutine GET_ISWITCH (ISWITCH_arg)
!     Retrieves ISWITCH variable
      IMPLICIT NONE
      Type (SwitchType) ISWITCH_arg
      ISWITCH_arg = SAVE_data % ISWITCH
      Return
      End Subroutine GET_ISWITCH

      END MODULE TestModule
