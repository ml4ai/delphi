      MODULE TestModule

      INTEGER, PARAMETER :: MaxFiles = 500

      TYPE ControlType
        CHARACTER (len=1)  MESIC, RNMODE
        CHARACTER (len=2)  CROP
        CHARACTER (len=8)  MODEL, ENAME
        CHARACTER (len=12) FILEX
        CHARACTER (len=30) FILEIO
        CHARACTER (len=102)DSSATP
        INTEGER   DAS, DYNAMIC, FROP, ErrCode, LUNIO, MULTI, N_ELEMS
        INTEGER   NYRS, REPNO, ROTNUM, RUN, TRTNUM
        INTEGER   YRDIF, YRDOY, YRSIM
      END TYPE ControlType

      TYPE SwitchType
        CHARACTER (len=1) FNAME
        CHARACTER (len=1) IDETC, IDETD, IDETG, IDETH, IDETL, IDETN
        CHARACTER (len=1) IDETO, IDETP, IDETR, IDETS, IDETW
        CHARACTER (len=1) IHARI, IPLTI, IIRRI, ISIMI
        CHARACTER (len=1) ISWCHE, ISWDIS, ISWNIT
        CHARACTER (len=1) ISWPHO, ISWPOT, ISWSYM, ISWTIL, ISWWAT
        CHARACTER (len=1) MEEVP, MEGHG, MEHYD, MEINF, MELI, MEPHO
        CHARACTER (len=1) MESOM, MESOL, MESEV, MEWTH
        CHARACTER (len=1) METMP !Temperature, EPIC
        CHARACTER (len=1) IFERI, IRESI, ICO2, FMOPT
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

      INTERFACE PUT
         MODULE PROCEDURE PUT_Control
     &                  , PUT_ISWITCH 
      END INTERFACE

      CONTAINS

      Subroutine GET_CONTROL (CONTROL_arg)
!     Retrieves CONTROL variable
      IMPLICIT NONE
      Type (ControlType) CONTROL_arg
      Control_arg = SAVE_data % Control
      Return
      End Subroutine GET_CONTROL

      Subroutine PUT_CONTROL (CONTROL_arg)
!     Stores CONTROL variable
      IMPLICIT NONE
      Type (ControlType) CONTROL_arg
      SAVE_data % Control = Control_arg
      Return
      End Subroutine PUT_CONTROL

      Subroutine GET_ISWITCH (ISWITCH_arg)
!     Retrieves ISWITCH variable
      IMPLICIT NONE
      Type (SwitchType) ISWITCH_arg
      ISWITCH_arg = SAVE_data % ISWITCH
      Return
      End Subroutine GET_ISWITCH

      Subroutine PUT_ISWITCH (ISWITCH_arg)
!     Stores ISWITCH variable
      IMPLICIT NONE
      Type (SwitchType) ISWITCH_arg
      SAVE_data % ISWITCH = ISWITCH_arg
      Return
      End Subroutine PUT_ISWITCH

      END MODULE TestModule
