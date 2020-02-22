C=======================================================================
C  COPYRIGHT 1998-2015 DSSAT Foundation
C                      University of Florida, Gainesville, Florida
C                      International Fertilizer Development Center
C                      Washington State University
C  ALL RIGHTS RESERVED
C=======================================================================
C=======================================================================
C  PLANT, Subroutine
C
C  This routine calls plant growth routines for all non-cropgro crops.
C
C  These plant growth routines are handled differently by the main 
C     program than those of CROPGRO crops.  
C
C     Standardized output values are initialized by this routine and 
C     may or may not be computed by each individual plant growth module. 
C 
C     As additional crops are added, all will be called by this routine 
C     rather than by the main program.  
C-----------------------------------------------------------------------
C  Revision history
C
C  09/23/2001 CHP Written for modular potato model to be incorporated 
C                   into CSM. (PT_SUBSTOR)
C  01/20/2002 WDB/CHP Added Ceres-Maize (MAIZE)
C  01/30/2002 LAH/CHP Added Wheat model (CSCERES)
C  04/17/2002 GH  Redefine module calls based on module name
C  07/15/2002 CHP Added rice module (US)
C  08/06/2002 WDB/CHP Added Ceres-Millet and Ceres-Sorghum
C  03/12/2003 CHP Changed senescence variable to composite (SENESCE)
C                   as defined in ModuleDefs.for
C  06/19/2003 CHP Added KTRANS (= KCAN + 0.15) for export to 
C                   transpiration routine.
C  07/08/2003 CHP Export KSEVAP for use in soil evaporation routine. 
C  10/08/2004 CHP Added GetPut_Iswitch call to push switch information
C                   into constructed variable which is accessible to
C                   all modules.
C  10/08/2004 CHP Removed some unused variables.
!  10/24/2005 CHP Put weather variables in constructed variable. 
!  02/27/2006 CHP Changed Alt_Plant to PLANT and moved call to CROPGRO
!                 here.
!  07/13/2006 CHP Added P model to CROPGRO
!  02/06/2007 FSR added CASUPRO sugarcane
!  02/08/2007 LAH,CHP Added CSCRP (cassava) 
!  07/15/2007 CHP, MJ added Canegro
!  09/10/2007 CHP, JIL added sweet corn
C  10/31/2007 US/RO/CHP Added TR_SUBSTOR (taro)
!  10/31/2007 CHP Added simple K model.
C  08/09/2012 GH  Added CSCAS model
!  04/16/2013 CHP/KAD Added SALUS model
!  05/09/2013 CHP/FR/JZW Added N-wheat module
!  06/03/2015 LPM Added CSYCA model (CIAT cassava)
!  05/10/2017 CHP removed SALUS model
!  12/01/2015 WDB added Sugarbeet
!  09/01/2018  MJ modified Canegro interface, IRRAMT added.
C=======================================================================

      SUBROUTINE PLANT(CONTROL, ISWITCH, 
     &    EO, EOP, EOS, EP, ES, FLOODWAT, HARVFRAC,       !Input
     &    NH4, NO3, RWU, SKi_Avail, SomLitC, SomLitE,     !Input
     &    SPi_AVAIL, SNOW, SOILPROP, SRFTEMP, ST, SW,     !Input
     &    TRWU, TRWUP, UPPM, WEATHER, YREND, YRPLT,       !Input
     &    IRRAMT,                                         !Input
     &    FLOODN,                                         !I/O
     &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS,        !Output
     &    KUptake, MDATE, NSTRES, PSTRES1,                !Output
     &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
     &    STGDOY, FracRts, UH2O, UNH4, UNO3, XHLAI, XLAI) !Output

C-----------------------------------------------------------------------
!     The following models are currently supported:
!         'CRGRO' - CROPGRO
!         'CSCER' - CERES Wheat, Barley
!         'CSCRP' - CropSim Wheat, Barley
!         'CSCAS' - CropSim/GumCAS Cassava
!         'CSYCA' - CIAT Cassava model
!         'MLCER' - CERES-Millet 
!         'MZCER' - CERES-Maize
!         'PTSUB' - SUBSTOR-Potato
!         'RICER' - CERES-Rice
!         'SCCAN' - CANEGRO Sugarcane
!         'SCCSP' - CASUPRO Sugarcane
!         'SGCER' - CERES-Sorghum
!         'SWCER' - CERES-Sweet corn
!         'MZIXM' - IXIM Maize
!         'TNARO' - Aroids - Tanier
!         'TRARO' - Aroids - Taro
!         'RIORZ' - IRRI ORYZA Rice model
!         'WHAPS' - APSIM N-wheat
!         'PRFRM' - Perennial forage model
!         'BSCER' - Sugarbeet
C-----------------------------------------------------------------------

C-----------------------------------------------------------------------
! Each plant module must compute SATFAC, SWFAC, and TURFAC
C-----------------------------------------------------------------------
      USE ModuleDefs 
      USE ModuleData
      USE FloodModule    

      IMPLICIT NONE
      SAVE

      CHARACTER*1  MEEVP, RNMODE
      CHARACTER*6  ERRKEY
      PARAMETER (ERRKEY = 'PLANTS')
      CHARACTER*8  MODEL
      CHARACTER*78 MESSAGE(10)    !Up to 10 lines of text to be output

      INTEGER DYNAMIC
      INTEGER RUN !, NVALP0
      INTEGER YREND, MDATE, YRPLT  !, YRSIM, YREMRG
      INTEGER STGDOY(20)

      REAL CANHT, CO2, DAYL, EO, EOP, EORATIO, EOS, EP, ES
      REAL KCAN, KEP, KSEVAP, KTRANS, LAI, NSTRES
      REAL PORMIN, RWUEP1, RWUMX, SRFTEMP, SNOW, IRRAMT
      REAL TMAX, TMIN, TRWU
      REAL TRWUP, TWILEN, XLAI, XHLAI

      REAL, DIMENSION(2)  :: HARVFRAC
      REAL, DIMENSION(NL) :: NH4, NO3, RLV, UPPM, RWU
      REAL, DIMENSION(NL) :: ST, SW, UNO3, UNH4, UH2O

      LOGICAL FixCanht, BUNDED    !, CRGRO
c-----------------------------------------------------------------------
C         Variables needed to run ceres maize.....W.D.B. 12-20-01
      CHARACTER*2 CROP 
      REAL    SRAD

!-----------------------------------------------------------------------
C         Variables to run CASUPRO from Alt_PLANT.  FSR 07-23-03
      REAL PAR, TAVG, TGROAV  !CHP 7/26/04 , TDAY
      REAL TGRO(TS)

	INTEGER, PARAMETER :: CanopyLayers=3
	REAL, DIMENSION(1:NumOfStalks,CanopyLayers) :: LFmntDEF
!     P model
      REAL, DIMENSION(NL) :: PUptake, SPi_AVAIL, FracRts
      REAL PSTRES1

!     K model
      REAL, DIMENSION(NL) :: KUptake, SKi_Avail

!     ORYZA Rice model
      REAL, DIMENSION(0:NL) :: SomLitC
      REAL, DIMENSION(0:NL,NELEM) :: SomLitE
      LOGICAL, PARAMETER :: OR_OUTPUT = .FALSE.

!-----------------------------------------------------------------------
!     Constructed variables are defined in ModuleDefs.
      TYPE (ControlType)  CONTROL
      TYPE (SwitchType)   ISWITCH
      TYPE (SoilType)     SOILPROP
      TYPE (ResidueType)  HARVRES
      TYPE (ResidueType)  SENESCE
      TYPE (FloodWatType) FLOODWAT
      TYPE (FloodNType)   FLOODN
      TYPE (WeatherType)  WEATHER

!     Transfer values from constructed data types into local variables.
      CROP    = CONTROL % CROP
      DYNAMIC = CONTROL % DYNAMIC
      MODEL   = CONTROL % MODEL
      RNMODE  = CONTROL % RNMODE
      RUN     = CONTROL % RUN

      MEEVP  = ISWITCH % MEEVP
      BUNDED = FLOODWAT % BUNDED
      CO2    = WEATHER % CO2   
      DAYL   = WEATHER % DAYL  
      PAR    = WEATHER % PAR  
      SRAD   = WEATHER % SRAD  
      TAVG   = WEATHER % TAVG  
      TGRO   = WEATHER % TGRO  
      TGROAV = WEATHER % TGROAV
      TMAX   = WEATHER % TMAX  
      TMIN   = WEATHER % TMIN  
      TWILEN = WEATHER % TWILEN

!***********************************************************************
!***********************************************************************
      IF (DYNAMIC .EQ. RUNINIT) THEN
!-----------------------------------------------------------------------
!     Non-CROPGRO crops can not use MEPHO = 'L' or MEEVP = 'Z' at
!       this time.  If Species files are modified for these options
!       in the future, we need to make this check on a crop by crop basis.
!     The plant routines do not use these codes, but the SPAM module
!       does and it will bomb when species parameters are not found.
      IF (INDEX(MODEL,'CRGRO') <= 0 .and. index(model,'PRFRM') <= 0 
     &  .AND. ISWITCH % MEPHO .EQ. 'L') THEN
        ISWITCH % MEPHO = 'C'
!       Put ISWITCH data where it can be retreived 
!         by other modules as needed.
        CALL PUT(ISWITCH)

!       Write message to WARNING.OUT file
        WRITE(MESSAGE(1),110) 
        WRITE(MESSAGE(2),120) CROP
        WRITE(MESSAGE(3),130) 
        CALL WARNING(3, ERRKEY, MESSAGE)
      ENDIF

  110 FORMAT('You have specified use of the Leaf-level photosynthesis')
  120 FORMAT('option, which is not available for crop ', A2, '.')
  130 FORMAT('Canopy photosynthesis option will be used.')

C      IF (INDEX(MODEL,'CRGRO') <= 0 .and. index(model,'PRFRM') <= 0 
C     &  .AND. ISWITCH % MEEVP .EQ. 'Z') THEN
C!       Default to Priestly-Taylor potential evapotranspiration
C        ISWITCH % MEEVP = 'R'
C!       Put ISWITCH data where it can be retreived 
C!         by other modules as needed.
C        CALL PUT(ISWITCH)
C
C!       Write message to WARNING.OUT file
C        WRITE(MESSAGE(1),210) 
C        WRITE(MESSAGE(2),220) CROP
C        WRITE(MESSAGE(3),230) 
C        WRITE(MESSAGE(4),240) 
C        CALL WARNING(4, ERRKEY, MESSAGE)
C      ENDIF
C      
C  210 FORMAT('You have specified use of the Zonal evapotranspiration')
C  220 FORMAT('option, which is not available for crop ', A2, '.')
C  230 FORMAT('The Priestly-Taylor potential evapo-transpiration ')
C  240 FORMAT('option will be used.')

!---------------------------------------------------------------------
!     Print warning if "dynamic ET" routine is used - need canopy height
!     Need to modify this crop code list as canopy height routines are added.
      IF (MEEVP .EQ. 'D' .AND. INDEX('RIWHMLMZSGPTBA',CROP) .GT. 0) THEN
        WRITE(MESSAGE(1),310) 
        WRITE(MESSAGE(2),320) 
        WRITE(MESSAGE(3),330) CROP
        WRITE(MESSAGE(4),340) 
        CALL WARNING(4, ERRKEY, MESSAGE)
        !Trigger to set canopy height upon emergence
        FixCanht = .TRUE.
      ENDIF

  310 FORMAT('You are using the Penman-Monteith potential evapo-')
  320 FORMAT('transpiration method, which requires canopy height.')
  330 FORMAT('The ',A2,' crop routine does not calculate this ')
  340 FORMAT('value.  A default value will be used.')

!     Initialize output variables.
!     Each plant routine may or may not re-compute these values.
      CANHT    = 0.0
      EORATIO  = 1.0
      KCAN     = 0.85
      KEP      = 1.0
      KSEVAP   = -99.
      KTRANS   = 1.0
      MDATE    = -99
      NSTRES   = 1.0
      !NVALP0   = 10000
      PORMIN   = 0.02
      RLV      = 0.0
      RWUEP1   = 1.5
      RWUMX    = 0.03
      !SENESCE % CumResWt= 0.0
      !SENESCE % CumResE = 0.0
      STGDOY   = 9999999
      XHLAI    = 0.0
      XLAI     = 0.0
      !YREMRG   = -99
      SENESCE % ResWt  = 0.0
      SENESCE % ResLig = 0.0
      SENESCE % ResE   = 0.0
      UNH4     = 0.0
      UNO3     = 0.0
      UH2O     = 0.0

!***********************************************************************
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. SEASINIT) THEN
!-----------------------------------------------------------------------
!     If this is not a sequenced run, don't use any previously calculated
!       harvest residue.
!     This should be done by each plant routine, but in case not:
      IF (RUN .EQ. 1 .OR. INDEX('QF',RNMODE) .LE. 0) THEN
        HARVRES % RESWT  = 0.0
        HARVRES % RESLig = 0.0
        HARVRES % RESE   = 0.0
!        HARVRES % CumResWt= 0.0
!        HARVRES % CumResE = 0.0
      ENDIF

      KUptake = 0.0

!***********************************************************************
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. RATE) THEN
!-----------------------------------------------------------------------
        SENESCE % ResWt  = 0.0
        SENESCE % ResLig = 0.0
        SENESCE % ResE   = 0.0

!-----------------------------------------------------------------------
      ENDIF   !End of dynamic loop prior to calls to crop models
!-----------------------------------------------------------------------

!***********************************************************************
!***********************************************************************
!     Call crop models for all values of DYNAMIC:
      SELECT CASE (MODEL(1:5))
!-----------------------------------------------------------------------
!     CROPGRO model
      CASE('CRGRO') 
        CALL CROPGRO(CONTROL, ISWITCH, 
     &    EOP, HARVFRAC, NH4, NO3, SOILPROP, SPi_AVAIL,   !Input
     &    ST, SW, TRWUP, WEATHER, YREND, YRPLT,           !Input
     &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS, MDATE, !Output
     &    NSTRES, PSTRES1,                                !Output
     &    PUptake, PORMIN, RLV, RWUMX, SENESCE,           !Output
     &    STGDOY, FracRts, UNH4, UNO3, XHLAI, XLAI)       !Output

C!-----------------------------------------------------------------------
C!     Forage model
C      CASE('PRFRM') 
C      call FORAGE(CONTROL, ISWITCH, 
C     &    EOP, HARVFRAC, NH4, NO3, SOILPROP,              !Input
C     &    ST, SW, TRWUP, WEATHER, YREND, YRPLT,           !Input
C     &    CANHT, EORATIO, HARVRES, KSEVAP, KTRANS, MDATE, !Output
C     &    NSTRES, PSTRES1,                                !Output
C     &    PORMIN, RLV, RWUMX, SENESCE,                    !Output
C     &    STGDOY, UNH4, UNO3, XHLAI, XLAI)                !Output
C
C!     -------------------------------------------------
C!     Wheat and Barley CSCER
C      CASE('CSCER')
C        CALL CSCERES_Interface (CONTROL, ISWITCH,          !Input
C     &     EOP, YREND, NH4, NO3, SNOW, SOILPROP,           !Input
C     &     SRFTEMP, ST, SW, TRWUP, WEATHER, YRPLT, HARVFRAC,!Input
C     &     CANHT, HARVRES, KCAN, KEP, MDATE, NSTRES,PORMIN,!Output
C     &     RLV, RWUMX, SENESCE, STGDOY, UNH4, UNO3, XLAI)  !Output
C
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C          KTRANS = KEP
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     Wheat and Barley CSCRP
C      CASE('CSCRP')
C        CALL CSCRP_Interface (CONTROL, ISWITCH,           !Input
C     &    EOP, ES, NH4, NO3, SNOW, SOILPROP, SRFTEMP,     !Input
C     &    ST, SW, TRWUP, WEATHER, YREND, YRPLT, HARVFRAC, !Input
C     &    CANHT, HARVRES, KCAN, KEP, MDATE, NSTRES,       !Output
C     &    PORMIN, RLV, RWUMX, SENESCE, STGDOY,            !Output
C     &    UNH4, UNO3, XLAI)                               !Output
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C          KTRANS = KEP
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C!     -------------------------------------------------
C!     Cassava CSCAS
C      CASE('CSCAS')
C        CALL CSCAS_Interface (CONTROL, ISWITCH,           !Input
C     &    EOP, ES, NH4, NO3, SOILPROP, SRFTEMP,           !Input
C     &    ST, SW, TRWUP, WEATHER, YREND, YRPLT, HARVFRAC, !Input
C     &    CANHT, HARVRES, KCAN, KEP, MDATE, NSTRES,       !Output
C     &    PORMIN, RLV, RWUMX, SENESCE, STGDOY,            !Output
C     &    UNH4, UNO3, XLAI)                               !Output
C
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C          KTRANS = KEP
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C!     -------------------------------------------------
C!     Cassava CSYCA (CIAT cassava model)
C      CASE('CSYCA')
C        CALL CSYCA_Interface (CONTROL, ISWITCH,           !Input
C     &    EOP, ES, NH4, NO3, SOILPROP, SRFTEMP,           !Input
C     &    ST, SW, TRWUP, WEATHER, YREND, YRPLT, HARVFRAC, !Input
C     &    CANHT, HARVRES, KCAN, KEP, MDATE, NSTRES,       !Output
C     &    PORMIN, RLV, RWUMX, SENESCE, STGDOY,            !Output
C     &    UNH4, UNO3, XLAI)                               !Output
C
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C          KTRANS = KEP
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     APSIM N-wheat WHAPS
C      CASE('WHAPS')
C        CALL WH_APSIM (CONTROL, ISWITCH,              !Input
C     &     EO, EOP, ES, HARVFRAC, NH4, NO3, SKi_Avail,            !Input
C     &     SPi_AVAIL, SNOW,                               !Input
C     &     SOILPROP, SW, TRWUP, WEATHER, YREND, YRPLT,    !Input
C     &     CANHT, HARVRES, KCAN, KEP, KUptake, MDATE,     !Output
C     &     NSTRES, PORMIN, PUptake, RLV,                  !Output
C     &     RWUMX, SENESCE, STGDOY, FracRts,               !Output
C     &     UNH4, UNO3, XLAI, XHLAI, UH2O)               !Output
C
C        IF (DYNAMIC < RATE) THEN
C!          KTRANS = KCAN + 0.15        !Or use KEP here??
C          KTRANS = KEP        !KJB/WDB/CHP 10/22/2003
C          KSEVAP = KEP        
C        ENDIF
C
C!     -------------------------------------------------
C!     Millet 
C      CASE('MLCER')
C        CALL ML_CERES (CONTROL, ISWITCH, 
C     &     CO2, DAYL, EOP, HARVFRAC, NH4, NO3,            !Input
C     &     SNOW, SOILPROP, SRAD, SW, TMAX, TMIN,          !Input
C     &     TRWUP, TWILEN, YREND, YRPLT,                   !Input
C     $     CANHT, HARVRES, MDATE, NSTRES, PORMIN, RLV,    !Output
C     &     RWUMX, SENESCE, STGDOY, UNO3, UNH4, XLAI,      !Output
C     &     KCAN, KEP)                                     !Output
C
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C!          KTRANS = KCAN + 0.15        !Or use KEP here??
C          KTRANS = KEP        !KJB/WDB/CHP 10/22/2003
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     Maize, Sweetcorn
C      CASE('MZCER','MZIXM','SWCER')
C        CALL MZ_CERES (CONTROL, ISWITCH,                  !Input
C     &     EOP, HARVFRAC, NH4, NO3, SKi_Avail,            !Input
C     &     SPi_AVAIL, SNOW,                               !Input
C     &     SOILPROP, SW, TRWUP, WEATHER, YREND, YRPLT,    !Input
C     &     CANHT, HARVRES, KCAN, KEP, KUptake, MDATE,     !Output
C     &     NSTRES, PORMIN, PUptake, RLV, RWUMX, SENESCE,  !Output
C     &     STGDOY, FracRts, UNH4, UNO3, XLAI, XHLAI)      !Output
C
C        IF (DYNAMIC < RATE) THEN
C!          KTRANS = KCAN + 0.15        !Or use KEP here??
C          KTRANS = KEP        !KJB/WDB/CHP 10/22/2003
C          KSEVAP = KEP        
C        ENDIF
C
C!     -------------------------------------------------
C!     Sugarbeet
C      CASE('BSCER')
C        CALL BS_CERES (CONTROL, ISWITCH,              !Input
C     &     EOP, HARVFRAC, NH4, NO3, SKi_Avail,            !Input
C     &     SPi_AVAIL, SNOW,                               !Input
C     &     SOILPROP, SW, TRWUP, WEATHER, YREND, YRPLT,    !Input
C     &     CANHT, HARVRES, KCAN, KEP, MDATE,              !Output
C     &     NSTRES, PORMIN, PUptake, RLV, RWUMX, SENESCE,  !Output
C     &     STGDOY, FracRts,XLAI, XHLAI)          !Output
C
C        IF (DYNAMIC < RATE) THEN
C          KTRANS = KEP        !KJB/WDB/CHP 10/22/2003
C          KSEVAP = KEP        
C        ENDIF
C
C!     -------------------------------------------------
C!     Potato 
C      CASE('PTSUB') 
C        CALL PT_SUBSTOR(CONTROL, ISWITCH, 
C     &    CO2, EOP, HARVFRAC, NH4, NO3, SOILPROP, SRAD,   !Input
C     &    ST, SW, TMAX, TMIN, TRWUP, TWILEN, YREND, YRPLT,!Input
C     &    CANHT, HARVRES, MDATE, NSTRES, PORMIN, RLV,     !Output
C     &    RWUMX, SENESCE, STGDOY, UNH4, UNO3, XLAI)       !Output
C
C        IF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     Rice 
C      CASE('RICER')
C        CALL RICE(CONTROL, ISWITCH,
C     &    CO2, DAYL, EOP, FLOODWAT, HARVFRAC, NH4, NO3,   !Input
C     &    SKi_Avail, SPi_AVAIL,                           !Input
C     &    SOILPROP, SRAD, ST, SW, TMAX, TMIN, TRWUP,      !Input
C     &    TWILEN, YRPLT,                                  !Input
C     &    FLOODN,                                         !I/O
C     &    CANHT, HARVRES, XLAI, KUptake, MDATE, NSTRES,   !Output
C     &    PORMIN, PUptake, RWUEP1, RWUMX,                 !Output
C     &    RLV, SENESCE, STGDOY, FracRts, UNH4, UNO3)      !Output
C
C        IF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     ORYZA2000 Rice 
C      CASE('RIORZ')
C        CALL ORYZA_Interface (CONTROL, ISWITCH,                  !Input
C     &   EOP, FLOODWAT, HARVFRAC, NH4, NO3, SOILPROP,            !Input
C     &   SomLitC, SomLitE,                                       !Input
C     &   ST, SW, TRWUP, UPPM, WEATHER, YRPLT, YREND, OR_OUTPUT,  !Input
C     &   CANHT, HARVRES, KTRANS, KSEVAP, MDATE, NSTRES, PORMIN,  !Output
C     &   RLV, RWUMX, SENESCE, STGDOY, UNH4, UNO3, UH2O, XLAI)    !Output
C
C        IF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!!     -------------------------------------------------
C!!	Generic Salus crop model
C!!	KD 09/14/2009
C!	CASE('SALUS') 
C!	  CALL SALUS(CONTROL, ISWITCH, WEATHER, SOILPROP, ST,         !Input
C!     &  HARVFRAC, YRPLT, EOP, SW, RWU, TRWUP, NH4, NO3, SPi_AVAIL,  !Input
C!     &  KCAN, MDATE, RLV, XHLAI, UNO3, UNH4, PUptake)  	            !Output
C!	  IF (DYNAMIC .EQ. INTEGR) THEN
C!          XLAI = XHLAI
C!        ENDIF
C
C!     -------------------------------------------------
C!     Sugarcane - CANEGRO
C!     :::::::::::::::::::
C!     Matthew Jones, 2006-09-20
C!     :::::::::::::::::::::::::
C      CASE('SCCAN')
C      !  MJ Added IRRAMT July 2015
C      !  MJ Added ES July 2015
C      !  MJ added SATFAC Jan 2018
C        CALL SC_CNGRO (
C     &    CONTROL, ISWITCH,                                   !Input
C     &    CO2, DAYL, EOP, EP, EO, ES, HARVFRAC, NH4, NO3, SNOW,   !Input
C     &    SOILPROP, SRAD, SW, TMAX, TMIN, TRWUP, TRWU, EOS,   !Input
C     &    RWUEP1, TWILEN, YREND, YRPLT, WEATHER, IRRAMT,      !Input
C     $    CANHT, HARVRES, KCAN, KTRANS, MDATE, NSTRES,        !Output
C     &    PORMIN, RLV, RWUMX,SENESCE, STGDOY, UNH4,           !Output
C     &    UNO3, XLAI, XHLAI, EORATIO)                 !Output
C
Cc     Added by MJ, 2007-04-04:
Cc     ::::::::::::::::::::::::
Cc     Total LAI must exceed or be equal to healthy LAI:
C          XLAI = MAX(XLAI, XHLAI)
C
C!     -------------------------------------------------
C!     Sugarcane - CASUPRO
C      CASE('SCCSP')
C        CALL CSP_CASUPRO(CONTROL, ISWITCH, 
C     &    CO2, EOP, EOS, HARVFRAC, NH4, NO3, PAR,           !Input
C     &    SOILPROP, SPi_AVAIL, SW, TAVG, TGRO,              !Input  
C     &    TGROAV, TMIN, TRWUP, WEATHER, YREND, YRPLT,       !Input
C     &    CANHT, EORATIO, HARVRES, KTRANS, LFmntDEF, MDATE, !Output 
C     &    NSTRES, PUptake, PORMIN, RLV, RWUMX, SENESCE,     !Output
C     &    STGDOY, FracRts, UNH4, UNO3, XHLAI, XLAI)         !Output
C
C!     -------------------------------------------------
C!     Sorghum 
C      CASE('SGCER')
C        CALL SG_CERES (CONTROL, ISWITCH, 
C     &     CO2, DAYL, EOP, HARVFRAC, NH4, NO3,                  !Input
C     &     SNOW, SOILPROP, SPi_AVAIL, SRAD, SW, TMAX, TMIN,     !Input
C     &     TRWUP, TWILEN, YREND, YRPLT,                         !Input
C     &     CANHT, HARVRES, MDATE, NSTRES, PORMIN, PUptake,      !Output
C     &     RLV, RWUMX, SENESCE, STGDOY, UNO3, UNH4,             !Ouput
C     &     XLAI, KCAN, KEP, FracRts)                            !Output
C
C        IF (DYNAMIC .EQ. SEASINIT) THEN
C!          KTRANS = KCAN + 0.15        !Or use KEP here??
C          KTRANS = KEP        !KJB/WDB/CHP 10/22/2003
C          KSEVAP = KEP
C        ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     Aroids-taro 
C      CASE('TRARO','TNARO')
C        CALL TR_SUBSTOR(CONTROL, ISWITCH,
C     &    CO2, DAYL, EOP, FLOODWAT, HARVFRAC, NH4, NO3,   !Input
C     &    SOILPROP, SRAD, ST, SW, TMAX, TMIN, TRWUP,      !Input
C     &    YRPLT,                                          !Input
C     &    FLOODN,                                         !I/O
C     &    CANHT, HARVRES, XLAI, MDATE, NSTRES, PORMIN,    !Output
C     &    RWUEP1, RWUMX, RLV, SENESCE, STGDOY, UNH4, UNO3)!Output
C
C        IF (DYNAMIC .EQ. INTEGR) THEN
C          XHLAI = XLAI
C        ENDIF
C
C!     -------------------------------------------------
C!     Pineapple - Aloha model 
C      CASE('PIALO')
C        CALL Aloha_Pineapple(CONTROL, ISWITCH, 
C     &    EOP, HARVFRAC, NH4, NO3, SOILPROP, SW, TRWUP,   !Input
C     &    WEATHER, YRPLT,                                 !Input
C     &    LAI, MDATE, RLV, SENESCE, STGDOY, UNH4, UNO3)   !Output
C
C        XLAI  = LAI
C        XHLAI = LAI 
C!     -------------------------------------------------
      END SELECT

!***********************************************************************
!***********************************************************************
!     Processing after calls to crop models:
!-----------------------------------------------------------------------
      IF (DYNAMIC .EQ. SEASINIT) THEN
!-----------------------------------------------------------------------
! Zero the value of HARVRES composite variable here 
!!!NOTE: At this time, the variable has already been used to 
!     initialize soil properties for this season.
!  This should be done by each plant routine, but in case not:
        HARVRES % RESWT  = 0.0
        HARVRES % RESLig = 0.0
        HARVRES % RESE   = 0.0

!***********************************************************************
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. INTEGR) THEN
!-----------------------------------------------------------------------
!     Set default canopy height upon emergence (or first day with 
!       LAI.  Should actually set these defaults within each
!       crop routine.
        IF (FixCanht .AND. (XLAI .GT. 0.001 .OR. XHLAI .GT. 0.001)) THEN
          CANHT = 0.5   
          FixCanht = .FALSE.
        ENDIF

!***********************************************************************
      ENDIF

!***********************************************************************
      RETURN
      END SUBROUTINE PLANT

!===========================================================================
! Variable listing for Alt_Plant - updated 08/18/2003
! --------------------------------------------------------------------------
! CANHT     Canopy height (m)
! CO2       Atmospheric carbon dioxide concentration (µmol[CO2] / mol[air])
! CONTROL   Composite variable containing variables related to control 
!             and/or timing of simulation.  The structure of the variable 
!             (ControlType) is defined in ModuleDefs.for. 
! CROP      Crop identification code 
! DAYL      Day length on day of simulation (from sunrise to sunset) (hr)
! EOP       Potential plant transpiration rate (mm/d)
! EORATIO   Ratio of increase in EO with increase in LAI (up to LAI=6.0) 
!             for use with FAO-56 Penman reference EO. 
! ERRKEY    Subroutine name for error file 
! FIXCANHT  Logical variable, =TRUE if default canopy height is to be set 
!             by Alt_Plant routine upon emergence 
! FLOODN    Composite variable which contains flood nitrogen mass and 
!             concentrations. Structure of variable is defined in 
!             ModuleDefs.for. (var.)
! FLOODWAT  Composite variable containing information related to bund 
!             management. Structure of variable is defined in ModuleDefs.for. 
! HARVFRAC  Two-element array containing fractions of (1) yield harvested 
!             and (2) by-product harvested (fraction)
! HARVRES   Composite variable containing harvest residue amounts for total 
!             dry matter, lignin, and N amounts.  Structure of variable is 
!             defined in ModuleDefs.for. 
! ISWITCH   Composite variable containing switches which control flow of 
!             execution for model.  The structure of the variable 
!             (SwitchType) is defined in ModuleDefs.for. 
! IRRAMT    Irrigation amount (mm)
! KCAN      Canopy light extinction coefficient for daily PAR, for 
!             equidistant plant spacing, modified when in-row and between 
!             row spacing are not equal 
! KEP       Energy extinction coefficient for partitioning EO to EP 
! KSEVAP    Light extinction coefficient used for computation of soil 
!             evaporation 
! KTRANS    Light extinction coefficient used for computation of plant 
!             transpiration 
! MDATE     Harvest maturity date (YYYYDDD)
! MEEVP     Method of evapotranspiration ('P'=Penman, 'R'=Priestly-Taylor, 
!             'Z'=Zonal) 
! MESSAGE   Text array containing information to be written to WARNING.OUT 
!             file. 
! MODEL     Name of CROPGRO executable file 
! NH4(L)    Ammonium N in soil layer L (µg[N] / g[soil])
! NL        Maximum number of soil layers = 20 
! NO3(L)    Nitrate in soil layer L (µg[N] / g[soil])
! NSTRES    Nitrogen stress factor (1=no stress, 0=max stress) 
! NVALP0    Set to 100,000 in PHENOLOG, used for comparison of times of 
!             plant stages (d)
! PORMIN    Minimum pore space required for supplying oxygen to roots for 
!             optimal growth and function (cm3/cm3)
! RLV(L)    Root length density for soil layer L (cm[root] / cm3[soil])
! RNMODE    Simulation run mode (I=Interactive, A=All treatments, 
!             B=Batch mode, E=Sensitivity, D=Debug, N=Seasonal, Q=Sequence)
! RUN       Change in date between two observations for linear 
!             interpolation 
! RWUEP1    Threshold for reducing leaf expansion compared w/ ratio of 
!             TRWU/EP1 (total potential daily root water uptake/ actual 
!             transpiration) 
! RWUMX     Maximum water uptake per unit root length, constrained by soil 
!             water (cm3[water] / cm [root])
! SENESCE   Composite variable containing data about daily senesced plant 
!             matter. Structure of variable is defined in ModuleDefs.for 
! SNOW      Snow accumulation (mm)
! SOILPROP  Composite variable containing soil properties including bulk 
!             density, drained upper limit, lower limit, pH, saturation 
!             water content.  Structure defined in ModuleDefs. 
! SRAD      Solar radiation (MJ/m2-d)
! ST(L)     Soil temperature in soil layer L (°C)
! STGDOY(I) Day when plant stage I occurred (YYYYDDD)
! SW(L)     Volumetric soil water content in layer L
!            (cm3 [water] / cm3 [soil])
! TMAX      Maximum daily temperature (°C)
! TMIN      Minimum daily temperature (°C)
! TRWUP     Potential daily root water uptake over soil profile (cm/d)
! TWILEN    Daylength from twilight to twilight (h)
! UNH4(L)   Rate of root uptake of NH4, computed in NUPTAK
!            (kg [N] / ha - d)
! UNO3(L)   Rate of root uptake of NO3, computed in NUPTAK (kg [N] / ha -d)
! XHLAI     Healthy leaf area index (m2[leaf] / m2[ground])
! XLAI      Leaf area (one side) per unit of ground area
!            (m2[leaf] / m2[ground])
! YREMRG    Day of emergence (YYYYDDD)
! YREND     Date for end of season (usually harvest date) (YYYYDDD)
! YRPLT     Planting date (YYYYDDD)
!===========================================================================
