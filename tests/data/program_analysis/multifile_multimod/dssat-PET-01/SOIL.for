!=======================================================================
!  COPYRIGHT 1998-2010 The University of Georgia, Griffin, Georgia
!                      University of Florida, Gainesville, Florida
!                      Iowa State University, Ames, Iowa
!                      International Center for Soil Fertility and 
!                       Agricultural Development, Muscle Shoals, Alabama
!                      University of Guelph, Guelph, Ontario
!  ALL RIGHTS RESERVED
!=======================================================================
!  SOIL, Subroutine
!-----------------------------------------------------------------------
!  Soil Processes subroutine.  Calls the following modules:
!     SOILDYN     - integrates soil properties variables
!     WATBAL      - soil water balance
!     SoilN_inorg - inorganic soil N (from NTRANS)
!     SoilPi      - inorganic soil P
!     SoilKi      - inorganic soil K
!     SoilOrg     - Ceres soil organic matter (from NTRANS)
!     CENTURY     - Century soil organic matter
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  11/02/2001 CHP Written
!  04/20/2002 GH  Modified for crop rotations
!  10/24/2005 CHP Put weather variables in constructed variable. 
!  10/25/2005 CHP Removed NTRANS_OLD module
!  02/22/2006 CHP Added tiledrain.
!  03/03/2006 CHP Added tillage (A.Andales & WDBatchelor).
!  03/21/2006 CHP Added mulch effects
!  07/14/2006 CHP Added P model, split inorganic and organic N routines,
!                 move fertilizer and organic matter placement routines
!                 to management module.
!  10/31/2007 CHP Added simple K model.
C=====================================================================

      SUBROUTINE SOIL(CONTROL, ISWITCH, 
     &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
     &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
     &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
     &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
     &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
     &    SomLitC, SomLitE,                               !Output
     &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output

!-----------------------------------------------------------------------
      USE ModuleDefs
      USE FloodModule
      IMPLICIT NONE
      SAVE
!-----------------------------------------------------------------------
!     Interface variables:
!-----------------------------------------------------------------------
!     Input:
      TYPE (ControlType) , INTENT(IN) :: CONTROL
      TYPE (SwitchType)  , INTENT(IN) :: ISWITCH
      REAL               , INTENT(IN) :: ES
      TYPE (FertType)    , INTENT(IN) :: FERTDATA
      Type (ResidueType) , INTENT(IN) :: HARVRES
      REAL               , INTENT(IN) :: IRRAMT
      REAL               , INTENT(IN) :: KTRANS
      TYPE (OrgMatAppType),INTENT(IN) :: OMAData
      REAL, DIMENSION(NL), INTENT(IN) :: PUptake, KUptake
      Type (ResidueType) , INTENT(IN) :: SENESCE
      REAL               , INTENT(IN) :: SRFTEMP 
      REAL, DIMENSION(NL), INTENT(IN) :: ST
      REAL, DIMENSION(NL), INTENT(IN) :: FracRts
      REAL, DIMENSION(NL), INTENT(IN) :: SWDELTX
      TYPE (TillType)    , INTENT(IN) :: TILLVALS
      REAL, DIMENSION(NL), INTENT(IN) :: UNH4, UNO3
      TYPE (WeatherType) , INTENT(IN) :: WEATHER
      REAL               , INTENT(IN) :: XHLAI

      REAL, DIMENSION(NL) :: SomLit 

!     Input/Output:
      REAL, DIMENSION(NL), INTENT(INOUT) :: UPFLOW
      TYPE (FloodNType)   FLOODN
      TYPE (FloodWatType) FLOODWAT
      TYPE (MulchType)    MULCH

!     Output:
      REAL, DIMENSION(NL), INTENT(OUT) :: NH4
      REAL, DIMENSION(NL), INTENT(OUT) :: NO3
      REAL, DIMENSION(NL), INTENT(OUT) :: UPPM
      REAL, DIMENSION(NL), INTENT(OUT) :: SPi_AVAIL, SKi_AVAIL
      REAL               , INTENT(OUT) :: SNOW
      TYPE (SoilType)    , INTENT(OUT) :: SOILPROP
      REAL, DIMENSION(NL), INTENT(OUT) :: SW
      REAL, DIMENSION(NL), INTENT(OUT) :: SWDELTS
      REAL, DIMENSION(NL), INTENT(OUT) :: SWDELTU
      REAL               , INTENT(OUT) :: WINF
      INTEGER            , INTENT(OUT) :: YREND
      REAL, DIMENSION(0:NL) :: SomLitC
      REAL, DIMENSION(0:NL,NELEM) :: SomLitE

!-----------------------------------------------------------------------
!     Local variables:
      CHARACTER*1  MESOM

      INTEGER DYNAMIC

      REAL, DIMENSION(0:NL) :: newCO2 !DayCent
      REAL, DIMENSION(NL) :: DRN
      REAL, DIMENSION(NL) :: SPi_Labile
      REAL, DIMENSION(0:NL) :: LITC, SSOMC
      REAL, DIMENSION(0:NL,NELEM) :: IMM, MNR
      
!     Added for tile drainage:
      REAL TDFC
      INTEGER TDLNO

!-----------------------------------------------------------------------
!     Transfer values from constructed data types into local variables.
      DYNAMIC = CONTROL % DYNAMIC
      MESOM   = ISWITCH % MESOM

      write (*,10) "ENTERING SUBROUTINE SOIL: dynamic = ", dynamic,
     &", output = ", output, ", seasinit = ", seasinit
 10   FORMAT(3(A,I2))
      
C !***********************************************************************
C !     Call Soil Dynamics module 
C !      IF (DYNAMIC < OUTPUT) THEN
C         CALL SOILDYN(CONTROL, ISWITCH, 
C      &    KTRANS, MULCH, SomLit, SomLitC, SW, TILLVALS,   !Input
C      &    WEATHER, XHLAI,                                 !Input
C      &    SOILPROP)                                       !Output
C !      ENDIF
C 
C !     Call WATBAL first for all except seasonal initialization
C       IF (DYNAMIC /= SEASINIT) THEN
C         CALL WATBAL(CONTROL, ISWITCH, 
C      &    ES, IRRAMT, SOILPROP, SWDELTX,                  !Input
C      &    TILLVALS, WEATHER,                              !Input
C      &    FLOODWAT, MULCH, SWDELTU,                       !I/O
C      &    DRN, SNOW, SW, SWDELTS,                         !Output
C      &    TDFC, TDLNO, UPFLOW, WINF)                      !Output
C       ENDIF
C 
C !     Soil organic matter modules
C C      IF (MESOM .EQ. 'P') THEN
C !       Parton (Century-based) soil organic matter module
C C        CALL CENTURY(CONTROL, ISWITCH, 
C C     &    FERTDATA, FLOODWAT, FLOODN, HARVRES, NH4,       !Input
C C     &    NO3, OMADATA, SENESCE, SOILPROP, SPi_Labile,    !Input
C C     &    SRFTEMP, ST, SW, TILLVALS,                      !Input
C C     &    IMM, LITC, MNR, MULCH, SomLit, SomLitC,         !Output
C C     &    SomLitE, SSOMC,                                 !Output
C C     &    newCO2)                                         !for DayCent in SOILNI added by PG
C C      ELSE
C !      ELSEIF (MESOM .EQ. 'G') THEN
C !       Godwin (Ceres-based) soil organic matter module (formerly NTRANS)
C         CALL SoilOrg (CONTROL, ISWITCH, 
C      &    FLOODWAT, FLOODN, HARVRES, NH4, NO3, OMAData,   !Input
C      &    SENESCE, SOILPROP, SPi_Labile, ST, SW, TILLVALS,!Input
C      &    IMM, LITC, MNR, MULCH, newCO2, SomLit, SomLitC, !Output
C      &    SomLitE, SSOMC)                                 !Output
C C      ENDIF
C 
C !     Inorganic N (formerly NTRANS)
C       CALL SoilNi (CONTROL, ISWITCH, 
C      &    DRN, ES, FERTDATA, FLOODWAT, IMM, LITC, MNR,    !Input
C      &    newCO2, SNOW, SOILPROP, SSOMC, ST, SW, TDFC,    !Input
C      &    TDLNO, TILLVALS, UNH4, UNO3, UPFLOW, WEATHER,   !Input
C      &    XHLAI,                                          !Input
C      &    FLOODN,                                         !I/O
C      &    NH4, NO3, UPPM)                                 !Output
C 
C !     Inorganic P
C       CALL SoilPi(CONTROL, ISWITCH, FLOODWAT, 
C      &    FERTDATA, IMM, MNR, PUptake, SOILPROP,          !Input
C      &    FracRts, SW, TillVals,                          !Input
C      &    SPi_AVAIL, SPi_Labile, YREND)                   !Output
C 
C !     Inorganic K
C       CALL SoilKi(CONTROL, ISWITCH, 
C      &    FERTDATA, KUptake, SOILPROP, TILLVALS,          !Input
C      &    SKi_Avail)                                      !Output
C 
C       IF (DYNAMIC == SEASINIT) THEN
C !       Soil water balance -- call last for initialization
C         CALL WATBAL(CONTROL, ISWITCH, 
C      &    ES, IRRAMT, SOILPROP, SWDELTX,                  !Input
C      &    TILLVALS, WEATHER,                              !Input
C      &    FLOODWAT, MULCH, SWDELTU,                       !I/O
C      &    DRN, SNOW, SW, SWDELTS,                         !Output
C      &    TDFC, TDLNO, UPFLOW, WINF)                      !Output
C       ENDIF

!***********************************************************************

!-----------------------------------------------------------------------

      write (*,11) "LEAVING SUBROUTINE SOIL"
 11   FORMAT(A)

      RETURN
      END SUBROUTINE SOIL

!=======================================================================
