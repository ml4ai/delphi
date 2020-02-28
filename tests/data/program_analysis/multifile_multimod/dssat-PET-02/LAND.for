C=======================================================================
C  COPYRIGHT 1998-2014 DSSAT Foundation
C                      University of Florida, Gainesville, Florida
C                      International Fertilizer Development Center
C                      Washington State University 
C  ALL RIGHTS RESERVED
C=======================================================================
C=======================================================================
C  LAND UNIT Module. G.Hoogenboom, J.W.Jones, C.Porter
C-----------------------------------------------------------------------
C  Land Unit Module.  Provides the interface between soil, weather
C  and crops.  Based on the original CROPGRO routine
C=======================================================================
C  REVISION       HISTORY
C  12/01/2001 CHP Written.
C  12/12/2001 GH  Rename to Land
!  10/24/2005 CHP Put weather variables in constructed variable. 
!  02/28/2006 CHP Rename Alt_Plant to Plant, move call to CROPGRO there
!  03/03/2006 CHP Added tillage (A.Andales & WDBatchelor).
!  03/21/2006 CHP Added mulch effects
!  10/31/2007 CHP Added simple K model.

C=======================================================================
      SUBROUTINE LAND(CONTROL, ISWITCH, 
     &                YRPLT, MDATE, YREND)
      
C-----------------------------------------------------------------------
      USE ModuleDefs      
      USE FloodModule      
C      USE CsvOutput   ! VSH 

      IMPLICIT NONE
      SAVE
C-----------------------------------------------------------------------
C     Crop, Experiment, Command line Variables
C-----------------------------------------------------------------------
      CHARACTER*2  CROP
      CHARACTER*6  ERRKEY
      PARAMETER   (ERRKEY = 'LAND  ')
      CHARACTER*8  MODEL
      CHARACTER*30 FILEIO
      
C-----------------------------------------------------------------------
C     Date / Timing / Sequencing Variables
C-----------------------------------------------------------------------
      INTEGER      DYNAMIC, YRSIM, YRDOY

C-----------------------------------------------------------------------
C     Input and Output Handling
C-----------------------------------------------------------------------
      CHARACTER*1  IDETS, IPLTI
      CHARACTER*78 MSG(2)

C-----------------------------------------------------------------------
C     Weather module Variables
C-----------------------------------------------------------------------
      TYPE (WeatherType)  WEATHER

C-----------------------------------------------------------------------
C     Soil Processes Module Variables 
C-----------------------------------------------------------------------
      REAL SNOW, WINF
      REAL, DIMENSION(NL) :: NH4, NO3, SPi_Avail, SKi_Avail, ST, UPPM
      REAL, DIMENSION(NL) :: SW, SWDELTS, UPFLOW
      TYPE (SoilType) SOILPROP    !type defined in ModuleDefs
      TYPE (FloodWatType) FLOODWAT
      TYPE (FloodNType)   FloodN
      TYPE (MulchType)    MULCH
!     Needed for ORYZA-Rice
      REAL, DIMENSION(0:NL) :: SomLitC
      REAL, DIMENSION(0:NL,NELEM) :: SomLitE

C-----------------------------------------------------------------------
C     Soil - Plant - Atmosphere Module Variables
C-----------------------------------------------------------------------
      REAL EO, EOP, ES, SRFTEMP, TRWUP
      REAL SWDELTU(NL), SWDELTX(NL) !, RWU(NL)
!     Needed for CaneGro_SA
      REAL EOS, EP, TRWU
!     Calculated by ORYZA-Rice
      REAL UH2O(NL)
!     Needed for SALUS
      REAL RWU(NL)

C-----------------------------------------------------------------------
C     PLANT Module Variables
C-----------------------------------------------------------------------
      INTEGER MDATE
      INTEGER STGDOY(20)
      REAL CANHT, EORATIO, NSTRES, PORMIN, PSTRES1, RWUMX
      REAL XHLAI, XLAI
      REAL KSEVAP, KTRANS
      REAL, Dimension(NL) :: PUptake, RLV, FracRts, UNH4, UNO3, KUptake
      Type (ResidueType) HARVRES  !type defined in ModuleDefs
      Type (ResidueType) SENESCE  
      
C-----------------------------------------------------------------------
C     Operations Management Module Variables 
C-----------------------------------------------------------------------
      TYPE (TillType) TILLVALS
      INTEGER YREND, YRPLT
      REAL IRRAMT
      REAL, DIMENSION(2) :: HARVFRAC   !Harvest & byproduct fractions
      TYPE (FertType) FERTDATA         !Fertilizer application
      TYPE (OrgMatAppType)OMAData      !Organic matter application

C-----------------------------------------------------------------------
!!     Temporary timer function
!!     Date / time variables
!      INTEGER DATE_TIME(8)
!!      date_time(1)  The 4-digit year  
!!      date_time(2)  The month of the year  
!!      date_time(3)  The day of the month  
!!      date_time(4)  The time difference with respect to Coordinated Universal Time (UTC) in minutes  
!!      date_time(5)  The hour of the day (range 0 to 23) - local time  
!!      date_time(6)  The minutes of the hour (range 0 to 59) - local time  
!!      date_time(7)  The seconds of the minute (range 0 to 59) - local time  
!!      date_time(8)  The milliseconds of the second (range 0 to 999) - local time  
!      REAL TIME0, TIME1, TIME_START DELTA_TIME
C-----------------------------------------------------------------------

C     Define constructed variable types based on definitions in
C     ModuleDefs.for.
      TYPE (ControlType) CONTROL
      TYPE (SwitchType)  ISWITCH

C     Transfer values from constructed data types into local variables.
      CROP    = CONTROL % CROP
      DYNAMIC = CONTROL % DYNAMIC
      FILEIO  = CONTROL % FILEIO
      MODEL   = CONTROL % MODEL
      YRDOY   = CONTROL % YRDOY
      YRSIM   = CONTROL % YRSIM

      IPLTI   = ISWITCH % IPLTI

      WRITE (*,10) "ENTERING SUBROUTINE LAND: dynamic =", DYNAMIC,
     &     ", runinit =", RUNINIT, ", seasinit =", SEASINIT
      WRITE (*,10) "                          rate =", rate,
     &     ", seasend =", seasend, ", integr =", integr
 10   FORMAT(3(A,X,I2))

C***********************************************************************
C***********************************************************************
C     Run Initialization - Called once per simulation
C***********************************************************************
      IF (DYNAMIC .EQ. RUNINIT) THEN
C-----------------------------------------------------------------------
!!     Temporary timer function
!      !Get initial time
!      CALL DATE_AND_TIME (VALUES=DATE_TIME)
!!     Convert time to seconds
!      TIME0 = DATE_TIME(7) 
!     &      + DATE_TIME(8) / 1000.  
!     &      + DATE_TIME(6) * 60.  
!     &      + DATE_TIME(5) * 3600.
!      TIME_START = TIME0
C-----------------------------------------------------------------------
C     Read switches from FILEIO
C-----------------------------------------------------------------------
C      CALL IPIBS (CONTROL, ISWITCH, CROP, IDETS, MODEL)

C-----------------------------------------------------------------------
C     Read input parameters for weather routines
C-----------------------------------------------------------------------
       CALL WEATHR(CONTROL, ISWITCH, WEATHER, YREND)

C-----------------------------------------------------------------------
C     Read initial soil data 
C-----------------------------------------------------------------------
      CALL SOIL(CONTROL, ISWITCH, 
     &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
     &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
     &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
     &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
     &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
     &    SomLitC, SomLitE,                               !Output
     &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output

C-----------------------------------------------------------------------
C     Read initial soil-plant-atmosphere data
C-----------------------------------------------------------------------
      CALL SPAM(CONTROL, ISWITCH,
     &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
     &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
     &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
     &    FLOODWAT, SWDELTU,                              !I/O
     &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
     &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output

C C-----------------------------------------------------------------------
C C     Read initial plant module data
C C-----------------------------------------------------------------------
C       CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C 
C C-----------------------------------------------------------------------
C C     Initialize summary.out information
C C-----------------------------------------------------------------------
C       CALL OPSUM (CONTROL, ISWITCH, YRPLT)
C 
C C*********************************************************************** 
C C*********************************************************************** 
C C     SEASONAL INITIALIZATION
C C*********************************************************************** 
       ELSEIF (DYNAMIC .EQ. SEASINIT) THEN

C C-----------------------------------------------------------------------
C C     Call WEATHR for initialization - reads first day of weather
C C     data for use in soil N and soil temp initialization.
C C-----------------------------------------------------------------------
C       CALL WEATHR(CONTROL, ISWITCH, WEATHER, YREND)
C 
C C-----------------------------------------------------------------------
C C     Set planting date, adjust operations dates for seasonal or 
C C     sequenced runs.
C C-----------------------------------------------------------------------
C       CALL MGMTOPS(CONTROL, ISWITCH, 
C      &    FLOODWAT, HARVRES, NSTRES, SOILPROP, ST,        !Input 
C      &    STGDOY, SW, WEATHER,                            !Input
C      &    YREND, FERTDATA, HARVFRAC, IRRAMT,              !Output
C      &    MDATE, OMADATA, TILLVALS, YRPLT)                !Output
C 
C C-----------------------------------------------------------------------
C       IF (YRPLT < YRSIM .AND. CROP /= 'FA' .AND.
C      &    INDEX('AF', IPLTI) == 0) THEN
C           CALL ERROR(ERRKEY,2,' ',0)
C       ENDIF
C 
C C-----------------------------------------------------------------------
C C     Seasonal initialization for soil-plant-atmosphere processes
C !     chp moved this before SOIL, so soil temp is available 
C C-----------------------------------------------------------------------
C       CALL SPAM(CONTROL, ISWITCH,
C      &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
C      &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
C      &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
C      &    FLOODWAT, SWDELTU,                              !I/O
C      &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
C      &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output
C 
C C-----------------------------------------------------------------------
C C     Seasonal initialization for soil processes
C C-----------------------------------------------------------------------
C       CALL SOIL(CONTROL, ISWITCH, 
C      &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
C      &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
C      &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
C      &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
C      &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
C      &    SomLitC, SomLitE,                               !Output
C      &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output
C 
C C-----------------------------------------------------------------------
C C     Initialize PLANT routines (including phenology and pest)
C C-----------------------------------------------------------------------
C       CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C C-----------------------------------------------------------------------
C C     Initialize summary output file - possible output from 
C C     various modules.
C C-----------------------------------------------------------------------
C       IF (IDETS .EQ. 'Y' .OR. IDETS .EQ. 'A') THEN
C         CALL OPSUM (CONTROL, ISWITCH, YRPLT)
C       ENDIF
C 
C C***********************************************************************
C C***********************************************************************
C C     DAILY RATE CALCULATIONS
C C***********************************************************************
       ELSE IF (DYNAMIC .EQ. RATE) THEN
C-----------------------------------------------------------------------
C     Call WEATHER Subroutine to input weather data and to
C     calculate hourly radiation and air temperature values
C     Note: First day of weather has already been read by 
C       initialization call to WEATHR.
C-----------------------------------------------------------------------
      CALL WEATHR(CONTROL, ISWITCH, WEATHER, YREND)

C C-----------------------------------------------------------------------
C C     Call Operations Management module to determine today's 
C C     applications of irrigation, tillage, etc.
C C-----------------------------------------------------------------------
C       CALL MGMTOPS(CONTROL, ISWITCH, 
C      &    FLOODWAT, HARVRES, NSTRES, SOILPROP, ST,        !Input 
C      &    STGDOY, SW, WEATHER,                            !Input
C      &    YREND, FERTDATA, HARVFRAC, IRRAMT,              !Output
C      &    MDATE, OMADATA, TILLVALS, YRPLT)                !Output
C 
C C-----------------------------------------------------------------------
C C     Call Soil processes module to determine today's rates of 
C C     change of soil properties.
C C-----------------------------------------------------------------------
C       CALL SOIL(CONTROL, ISWITCH, 
C      &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
C      &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
C      &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
C      &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
C      &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
C      &    SomLitC, SomLitE,                               !Output
C      &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output
C 
C C-----------------------------------------------------------------------
C C     Call Soil-plant-atmosphere module to determine today's
C C     rates of evapotranspiration.
C C-----------------------------------------------------------------------
C       CALL SPAM(CONTROL, ISWITCH,
C      &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
C      &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
C      &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
C      &    FLOODWAT, SWDELTU,                              !I/O
C      &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
C      &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output
C 
C C-----------------------------------------------------------------------
C C     Call PLANT Subroutine to calculate crop growth and
C C     development rates.
C C     Skip plant growth and development routines for fallow runs
C C-----------------------------------------------------------------------
C       IF (CROP .NE. 'FA' .AND. 
C      &    YRDOY .GE. YRPLT .AND. YRPLT .NE. -99) THEN
C         CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C       ENDIF

          WRITE (*,11) "EXITING LAND AFTER 'IF (DYNAMIC .EQ. RATE)'"
          CALL EXIT(0)
C 
C C***********************************************************************
C C     DAILY INTEGRATION 
C C***********************************************************************
       ELSE IF (DYNAMIC .EQ. INTEGR) THEN
C C***********************************************************************
C C     Integrate soil state variables
C C-----------------------------------------------------------------------
C       CALL SOIL(CONTROL, ISWITCH, 
C      &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
C      &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
C      &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
C      &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
C      &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
C      &    SomLitC, SomLitE,                               !Output
C      &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output
C 
C C-----------------------------------------------------------------------
C C     Compute cumulative totals for soil-plant-atmosphere processes
C C-----------------------------------------------------------------------
C       CALL SPAM(CONTROL, ISWITCH,
C      &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
C      &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
C      &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
C      &    FLOODWAT, SWDELTU,                              !I/O
C      &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
C      &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output
C 
C C-----------------------------------------------------------------------
C C     Call Plant module to integrate daily plant processes and update
C C     plant state variables.
C C-----------------------------------------------------------------------
C       IF (CROP .NE. 'FA' .AND. 
C      &        YRDOY .GE. YRPLT .AND. YRPLT .NE. -99) THEN
C         CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C       ENDIF
C 
C C-----------------------------------------------------------------------
C C     Call Operations Management module to check for harvest end, 
C C     accumulate variables.
C C-----------------------------------------------------------------------
C       CALL MGMTOPS(CONTROL, ISWITCH, 
C      &    FLOODWAT, HARVRES, NSTRES, SOILPROP, ST,        !Input 
C      &    STGDOY, SW, WEATHER,                            !Input
C      &    YREND, FERTDATA, HARVFRAC, IRRAMT,              !Output
C      &    MDATE, OMADATA, TILLVALS, YRPLT)                !Output
C 
C C***********************************************************************
C C***********************************************************************
C C     Daily Output
C C***********************************************************************
       ELSE IF (DYNAMIC .EQ. OUTPUT) THEN

       CALL WEATHR(CONTROL, ISWITCH, WEATHER, YREND)
 
       CALL SOIL(CONTROL, ISWITCH, 
     &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
     &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
     &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
     &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
     &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
     &    SomLitC, SomLitE,                               !Output
     &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output
 
C         CALL SPAM(CONTROL, ISWITCH,
C      &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
C      &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
C      &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
C      &    FLOODWAT, SWDELTU,                              !I/O
C      &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
C      &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output
C 
C C-----------------------------------------------------------------------
C C     Call plant module for daily printout.
C C-----------------------------------------------------------------------
C         IF (CROP .NE. 'FA') THEN
C           CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C         ENDIF
C 
C         CALL MGMTOPS(CONTROL, ISWITCH, 
C      &    FLOODWAT, HARVRES, NSTRES, SOILPROP, ST,        !Input 
C      &    STGDOY, SW, WEATHER,                            !Input
C      &    YREND, FERTDATA, HARVFRAC, IRRAMT,              !Output
C      &    MDATE, OMADATA, TILLVALS, YRPLT)                !Output
C 
C C*********************************************************************** 
C C***********************************************************************
C C     Seasonal Output
C C*********************************************************************** 
C       ELSE IF (DYNAMIC .EQ. SEASEND) THEN
C C     Call WEATHER module to close current weather file 
C       CALL WEATHR(CONTROL, ISWITCH, WEATHER, YREND)
C 
C C     Print seasonal summaries and close files.
C       CALL SOIL(CONTROL, ISWITCH, 
C      &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
C      &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
C      &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
C      &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
C      &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
C      &    SomLitC, SomLitE,                               !Output
C      &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output
C 
C       CALL SPAM(CONTROL, ISWITCH,
C      &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
C      &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
C      &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
C      &    FLOODWAT, SWDELTU,                              !I/O
C      &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
C      &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output
C 
C       CALL PLANT(CONTROL, ISWITCH, 
C      &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
C      &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
C      &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
C      &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
C      &    IRRAMT,                                         !Input
C      &    FLOODN,                                         !I/O
C      &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
C      &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
C      &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
C      &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output
C 
C !     Call management operations module for seasonal printout.
C       CALL MGMTOPS(CONTROL, ISWITCH, 
C      &    FLOODWAT, HARVRES, NSTRES, SOILPROP, ST,        !Input 
C      &    STGDOY, SW, WEATHER,                            !Input
C      &    YREND, FERTDATA, HARVFRAC, IRRAMT,              !Output
C      &    MDATE, OMADATA, TILLVALS, YRPLT)                !Output
C 
C C-----------------------------------------------------------------------
C C     Seasonal Output
C C     Call end of season and summary output subroutines
C C-----------------------------------------------------------------------
C       CALL OPSUM (CONTROL, ISWITCH, YRPLT)
C 
C !!     Temporary timer function
C !      CALL DATE_AND_TIME (VALUES=DATE_TIME)
C !      
C !!     Convert time to seconds
C !      TIME1 = DATE_TIME(7) 
C !     &      + DATE_TIME(8) / 1000.  
C !     &      + DATE_TIME(6) * 60.  
C !     &      + DATE_TIME(5) * 3600.
C !      DELTA_TIME = TIME1 - TIME0
C !      WRITE(200,'(1X,"RUN ",I3,3X,F10.3)') RUN, DELTA_TIME
C !      TIME0 = TIME1
C 
C       IF (CONTROL % ERRCODE > 0) THEN
C         WRITE(MSG(1),'(A,I8)') "End of run ", CONTROL % RUN
C         WRITE(MSG(2),'("Simulation ended with error code ",I3)') 
C      &      CONTROL % ERRCODE
C         CALL WARNING(2,'ENDRUN',MSG)
C         CALL INFO(2,'ENDRUN',MSG)
C       ELSE
C         WRITE(MSG(1),'(A,I8)') "Normal end of run ", CONTROL % RUN
C         CALL WARNING(0,'ENDRUN',MSG)
C         CALL INFO(1,'ENDRUN',MSG)
C       ENDIF
C       
C !     VSH
C C      if (SOILPROP % NLAYR > maxnlayers ) then
C C         maxnlayers = SOILPROP % NLAYR
C C      end if 
C C*********************************************************************** 
C C***********************************************************************
C C     End of Run
C C*********************************************************************** 
       ELSE IF (DYNAMIC .EQ. ENDRUN) THEN
       CALL SOIL(CONTROL, ISWITCH, 
     &    ES, FERTDATA, HARVRES, IRRAMT, KTRANS,          !Input
     &    KUptake, OMAData, PUptake, SENESCE, SRFTEMP, ST,!Input
     &    FracRts, SWDELTX,TILLVALS, UNH4, UNO3, UPFLOW,  !Input
     &    WEATHER, XHLAI, FLOODN, FLOODWAT, MULCH,        !I/O
     &    NH4, NO3, SKi_AVAIL, SNOW, SPi_AVAIL, SOILPROP, !Output
     &    SomLitC, SomLitE,                               !Output
     &    SW, SWDELTS, SWDELTU, UPPM, WINF, YREND)        !Output

!!     Temporary timer function
!      CALL DATE_AND_TIME (VALUES=DATE_TIME)
!      
!!     Convert time to seconds
!      TIME1 = DATE_TIME(7) 
!     &      + DATE_TIME(8) / 1000.  
!     &      + DATE_TIME(6) * 60.  
!     &      + DATE_TIME(5) * 3600.
!      DELTA_TIME = TIME1 - TIME_START
!      WRITE(200,'(/," Total Time",F10.3)') RUN, DELTA_TIME

C !      VSH CSV outputs
C C       IF (ISWITCH % FMOPT == 'C') THEN
C C          CALL CsvOutputs(CONTROL % MODEL(1:5), CONTROL % N_ELEMS,
C C     & maxnlayers)
C C        END IF 

!***********************************************************************
!***********************************************************************
!     END OF DYNAMIC IF CONSTRUCT
!***********************************************************************
       ENDIF

      WRITE (*,11) "LEAVING SUBROUTINE LAND"
 11   FORMAT(A)

       RETURN
       END SUBROUTINE LAND 
