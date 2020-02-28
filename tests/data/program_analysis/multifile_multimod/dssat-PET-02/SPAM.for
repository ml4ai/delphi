C=======================================================================
C  COPYRIGHT 1998-2014 DSSAT Foundation
C                      University of Florida, Gainesville, Florida
C                      International Fertilizer Development Center
C                      Washington State University
C  ALL RIGHTS RESERVED
C=======================================================================
C=======================================================================
C  SPAM, Subroutine
C  Calculates soil-plant-atmosphere interface energy balance components.
C-----------------------------------------------------------------------
C  REVISION       HISTORY
C  11/09/2001 WMB/CHP Split WATBAL into WATBAL and SPAM.
C  02/06/2003 KJB/CHP Added KEP, EORATIO inputs from plant routines.
C  06/19/2003 CHP Added KTRANS - used instead of KEP in TRANS.
C  04/01/2004 CHP/US Added Penman - Meyer routine for potential ET
!  10/24/2005 CHP Put weather variables in constructed variable. 
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil 
!                 water effects on albedo)
!  08/25/2006 CHP Add SALUS soil evaporation routine, triggered by new
!                 FILEX parameter MESEV
!  12/09/2008 CHP Remove METMP
C-----------------------------------------------------------------------
C  Called by: Main
C  Calls:     XTRACT, OPSPAM    (File SPSUBS.for)
C             PET     (File PET.for)
C             PSE     (File PET.for)
C             ROOTWU  (File ROOTWU.for)
C             SOILEV  (File SOILEV.for)
C             TRANS   (File TRANS.for)
C=======================================================================

      SUBROUTINE SPAM(CONTROL, ISWITCH,
     &    CANHT, EORATIO, KSEVAP, KTRANS, MULCH,          !Input
     &    PSTRES1, PORMIN, RLV, RWUMX, SOILPROP, SW,      !Input
     &    SWDELTS, UH2O, WEATHER, WINF, XHLAI, XLAI,      !Input
     &    FLOODWAT, SWDELTU,                              !I/O
     &    EO, EOP, EOS, EP, ES, RWU, SRFTEMP, ST,         !Output
     &    SWDELTX, TRWU, TRWUP, UPFLOW)                   !Output

!-----------------------------------------------------------------------
      USE ModuleDefs 
      USE ModuleData
      USE FloodModule

      IMPLICIT NONE
      SAVE

      CHARACTER*1  IDETW, ISWWAT
      CHARACTER*1  MEEVP, MEINF, MEPHO, MESEV, METMP
      CHARACTER*2  CROP
      CHARACTER*6, PARAMETER :: ERRKEY = "SPAM  "
!      CHARACTER*78 MSG(2)

      INTEGER DYNAMIC, L, NLAYR

      REAL CANHT, CO2, SRAD, TAVG, 
     &    TMAX, TMIN, WINDSP, XHLAI, XLAI
      REAL CEF, CEM, CEO, CEP, CES, CET, EF, EM, EO, EP, ES, ET, EVAP, 
     &    TRWU, TRWUP, U
      REAL EOS, EOP, WINF, MSALB, ET_ALB
      REAL XLAT, TAV, TAMP, SRFTEMP
      REAL EORATIO, KSEVAP, KTRANS

      REAL DLAYR(NL), DUL(NL), LL(NL), RLV(NL), RWU(NL),  
     &    SAT(NL), ST(NL), SW(NL), SW_AVAIL(NL), !SWAD(NL), 
     &    SWDELTS(NL), SWDELTU(NL), SWDELTX(NL), UPFLOW(NL)
      REAL ES_LYR(NL)

!     ORYZA model
!     Root water uptake computed by some plant routines (optional)
      REAL UH2O(NL)
      REAL ETRD, ETAE, TRAT, TRATIO

!     Species-dependant variables imported from PLANT module:
      REAL PORMIN, RWUMX

!     Flood management variables:
      REAL FLOOD, EOS_SOIL
      
!     P Stress on photosynthesis
      REAL PSTRES1

!-----------------------------------------------------------------------
!     Define constructed variable types based on definitions in
!     ModuleDefs.for.
      TYPE (ControlType) CONTROL
      TYPE (SoilType) SOILPROP
      TYPE (SwitchType) ISWITCH
      TYPE (FloodWatType) FLOODWAT
      TYPE (MulchType)   MULCH
      TYPE (WeatherType)  WEATHER

!     Transfer values from constructed data types into local variables.
      CROP    = CONTROL % CROP
      DYNAMIC = CONTROL % DYNAMIC

      DLAYR  = SOILPROP % DLAYR
      DUL    = SOILPROP % DUL
      LL     = SOILPROP % LL
      MSALB  = SOILPROP % MSALB
      NLAYR  = SOILPROP % NLAYR
      SAT    = SOILPROP % SAT
      U      = SOILPROP % U

      ISWWAT = ISWITCH % ISWWAT
      IDETW  = ISWITCH % IDETW
      MEEVP  = ISWITCH % MEEVP
      MEINF  = ISWITCH % MEINF
      MEPHO  = ISWITCH % MEPHO
      METMP  = ISWITCH % METMP
      MESEV  = ISWITCH % MESEV

      FLOOD  = FLOODWAT % FLOOD

      CO2    = WEATHER % CO2
      SRAD   = WEATHER % SRAD  
      TAMP   = WEATHER % TAMP  
      TAV    = WEATHER % TAV   
      TAVG   = WEATHER % TAVG  
      TMAX   = WEATHER % TMAX  
      TMIN   = WEATHER % TMIN  
      WINDSP = WEATHER % WINDSP
      XLAT   = WEATHER % XLAT  

      write (*,10) "ENTERING SUBROUTINE SPAM: DYNAMIC = ", DYNAMIC, 
     &     ", RUNINIT = ", RUNINIT, ", SEASINIT = ", SEASINIT,
     &     ", RATE = ", RATE, ", INTEGR = ", INTEGR,
     &     ", SEASEND = ", SEASEND
 10   FORMAT(6(A,X,I2))

!***********************************************************************
!***********************************************************************
!     Run Initialization - Called once per simulation
!***********************************************************************
      IF (DYNAMIC .EQ. RUNINIT) THEN
!-----------------------------------------------------------------------
      IF (MEPHO .EQ. 'L' .OR. MEEVP .EQ. 'Z') THEN
        CALL ETPHOT(CONTROL, ISWITCH,
     &    PORMIN, PSTRES1, RLV, RWUMX, SOILPROP, ST, SW,  !Input
     &    WEATHER, XLAI,                                 !Input
     &    EOP, EP, ES, RWU, TRWUP)                        !Output
      ENDIF
      
      !Initialize ASCE dual KC ET variables (KRT)
      CALL PUT('SPAM', 'REFET', -99.0)
      CALL PUT('SPAM', 'KCB', -99.0)
      CALL PUT('SPAM', 'KE', -99.0)
      CALL PUT('SPAM', 'KC', -99.0)

C !***********************************************************************
C !***********************************************************************
C !     Seasonal initialization - run once per season
C !***********************************************************************
C       ELSEIF (DYNAMIC .EQ. SEASINIT) THEN
C !-----------------------------------------------------------------------
C       EF   = 0.0; CEF = 0.0
C       EM   = 0.0; CEM = 0.0
C       EO   = 0.0; CEO  = 0.0
C       EP   = 0.0; EOP = 0.0; CEP  = 0.0
C       ES   = 0.0; EOS = 0.0; CES  = 0.0
C       ET   = 0.0; CET  = 0.0
C       ES_LYR = 0.0
C       SWDELTX = 0.0
C       TRWU = 0.0
C 
C !     ---------------------------------------------------------
C       IF (meevp .NE.'Z') THEN   !LPM 02dec14 to use the values from ETPHOT
C           SELECT CASE (METMP)
C C          CASE ('E')    !EPIC soil temperature routine
C C            CALL STEMP_EPIC(CONTROL, ISWITCH,  
C C     &        SOILPROP, SW, TAVG, TMAX, TMIN, TAV, WEATHER,   !Input
C C     &        SRFTEMP, ST)                                    !Output
C           CASE DEFAULT  !DSSAT soilt temperature
C            CALL STEMP(CONTROL, ISWITCH,
C      &    SOILPROP, SRAD, SW, TAVG, TMAX, XLAT, TAV, TAMP,!Input
C      &    SRFTEMP, ST)   !Output
C      	   END SELECT
C       ENDIF
C !     ---------------------------------------------------------
C       IF (MEEVP .NE. 'Z') THEN
C         CALL ROOTWU(SEASINIT,
C      &      DLAYR, LL, NLAYR, PORMIN, RLV, RWUMX, SAT, SW,!Input
C      &      RWU, TRWUP)                           !Output
C 
C !       Initialize soil evaporation variables
C         SELECT CASE (MESEV)
C !     ----------------------------
C C        CASE ('R')  !Original soil evaporation routine
C C          CALL SOILEV(SEASINIT,
C C     &      DLAYR, DUL, EOS, LL, SW, SW_AVAIL(1),         !Input
C C     &      U, WINF,                                      !Input
C C     &      ES)                                           !Output
C !     ----------------------------
C !        CASE ('S')  !SALUS soil evaporation routine
C          !CALL ESR_SoilEvap(SEASINIT,
C      &   ! EOS, Pond_EV, SOILPROP, SW, SW_AVAIL,           !Input
C      &   ! ES, SWDELTU, SWAD, UPFLOW)                      !Output
C 
C !     ----------------------------
C         END SELECT
C 
C !       Initialize plant transpiration variables
C         CALL TRANS(DYNAMIC, 
C      &    CO2, CROP, EO, EVAP, KTRANS, TAVG,              !Input
C      &    WINDSP, XHLAI,                                  !Input
C      &    EOP)                                            !Output
C       ENDIF
C 
C       CALL MULCH_EVAP(DYNAMIC, MULCH, EOS, EM)
C 
C !     ---------------------------------------------------------
C       IF (CROP .NE. 'FA') THEN
C         IF (MEPHO .EQ. 'L' .OR. MEEVP .EQ. 'Z') THEN
C           CALL ETPHOT(CONTROL, ISWITCH,
C      &    PORMIN, PSTRES1, RLV, RWUMX, SOILPROP, ST, SW,  !Input
C      &    WEATHER, XLAI,                                 !Input
C      &    EOP, EP, ES, RWU, TRWUP)                        !Output
C         ENDIF
C       ENDIF
C 
C !     Call OPSPAM to open and write headers to output file
C       IF (IDETW .EQ. 'Y') THEN
C         CALL OPSPAM(CONTROL, ISWITCH, FLOODWAT, TRWU,
C      &    CEF, CEM, CEO, CEP, CES, CET, EF, EM, 
C      &    EO, EOP, EOS, EP, ES, ET, TMAX, TMIN, SRAD,
C      &    ES_LYR, SOILPROP)
C       ENDIF
C 
C !     Transfer data to storage routine
C       CALL PUT('SPAM', 'CEF', CEF)
C       CALL PUT('SPAM', 'CEM', CEM)
C       CALL PUT('SPAM', 'CEO', CEO)
C       CALL PUT('SPAM', 'CEP', CEP)
C       CALL PUT('SPAM', 'CES', CES)
C       CALL PUT('SPAM', 'CET', CET)
C       CALL PUT('SPAM', 'EF',  EF)
C       CALL PUT('SPAM', 'EM',  EM)
C       CALL PUT('SPAM', 'EO',  EO)
C       CALL PUT('SPAM', 'EP',  EP)
C       CALL PUT('SPAM', 'ES',  ES)
C       CALL PUT('SPAM', 'ET',  ET)
C 
C !***********************************************************************
C !***********************************************************************
C !     DAILY RATE CALCULATIONS
C !***********************************************************************
C       ELSEIF (DYNAMIC .EQ. RATE) THEN
C !-----------------------------------------------------------------------
C       SWDELTX = 0.0
C !     ---------------------------------------------------------
C       IF (meevp .NE.'Z') THEN  !LPM 02dec14 to use the values from ETPHOT
C           SELECT CASE (METMP)
C C          CASE ('E')    !EPIC soil temperature routine
C C            CALL STEMP_EPIC(CONTROL, ISWITCH,  
C C     &        SOILPROP, SW, TAVG, TMAX, TMIN, TAV, WEATHER,   !Input
C C     &        SRFTEMP, ST)                                    !Output
C           CASE DEFAULT  
C !     7/21/2016 - DSSAT method is default, per GH
C !     CASE ('D')  !DSSAT soil temperature
C         CALL STEMP(CONTROL, ISWITCH,
C      &    SOILPROP, SRAD, SW, TAVG, TMAX, XLAT, TAV, TAMP,!Input
C      &    SRFTEMP, ST)                                    !Output
C           END SELECT
C       ENDIF
C !-----------------------------------------------------------------------
C !     POTENTIAL ROOT WATER UPTAKE
C !-----------------------------------------------------------------------
C       IF (ISWWAT .EQ. 'Y') THEN
C !       Calculate the availability of soil water for use in SOILEV.
C         DO L = 1, NLAYR
C           SW_AVAIL(L) = MAX(0.0, SW(L) + SWDELTS(L) + SWDELTU(L))
C         ENDDO
C 
C !       These processes are done by ETPHOT for hourly (Zonal) energy
C !       balance method.
C         IF (MEEVP .NE. 'Z') THEN
C C       Calculate potential root water uptake rate for each soil layer
C C       and total potential water uptake rate.
C           IF (XHLAI .GT. 0.0) THEN
C             CALL ROOTWU(RATE,
C      &          DLAYR, LL, NLAYR, PORMIN, RLV, RWUMX, SAT, SW,!Input
C      &          RWU, TRWUP)                           !Output
C           ELSE
C             RWU   = 0.0
C             TRWUP = 0.0
C           ENDIF
C 
C !-----------------------------------------------------------------------
C !         POTENTIAL EVAPOTRANSPIRATION
C !-----------------------------------------------------------------------
C           IF (FLOOD .GT. 0.0) THEN
C             ! Set albedo to 0.08 under flooded conditions
C             ! US - change to 0.05 Feb2004
C             ET_ALB = 0.05
C           ELSE
C             ET_ALB = MSALB
C           ENDIF
C 
C           CALL PET(CONTROL, 
C      &      ET_ALB, XHLAI, MEEVP, WEATHER,  !Input for all
C      &      EORATIO, !Needed by Penman-Monteith
C      &      CANHT,   !Needed by dynamic Penman-Monteith
C      &      EO)      !Output
C 
C !-----------------------------------------------------------------------
C !         POTENTIAL SOIL EVAPORATION
C !-----------------------------------------------------------------------
C !         05/26/2007 CHP/MJ Use XLAI instead of XHLAI 
C !         This was important for Canegro and affects CROPGRO crops
C !             only very slightly (max 0.5% yield diff for one peanut
C !             experiment).  No difference to other crop models.
C           CALL PSE(EO, KSEVAP, XLAI, EOS)
C 
C !-----------------------------------------------------------------------
C !         ACTUAL SOIL, MULCH AND FLOOD EVAPORATION
C !-----------------------------------------------------------------------
C !         Initialize soil, mulch and flood evaporation
C           ES = 0.; EM = 0.; EF = 0.; EVAP = 0.0
C           UPFLOW = 0.0; ES_LYR = 0.0
C 
C C!         First meet evaporative demand from floodwater
C C          IF (FLOOD .GT. 1.E-4) THEN
C C            CALL FLOOD_EVAP(XLAI, EO, EF)   
C C            IF (EF > FLOOD) THEN
C C!             Floodwater not enough to supply EOS demand
C C              EOS_SOIL = MIN(EF - FLOOD, EOS)
C C              EF = FLOOD
C C            ELSE
C C              EOS_SOIL = 0.0
C C            ENDIF
C C          ELSE
C C            EOS_SOIL = EOS
C C          ENDIF
C 
C !         Next meet evaporative demand from mulch
C           IF (EOS_SOIL > 1.E-6 .AND. INDEX('RSM',MEINF) > 0) THEN
C             CALL MULCH_EVAP(DYNAMIC, MULCH, EOS_SOIL, EM)
C             IF (EOS_SOIL > EM) THEN
C !             Some evaporative demand leftover for soil
C               EOS_SOIL = EOS_SOIL - EM
C             ELSE
C               EOS_SOIL = 0.0
C             ENDIF
C           ENDIF
C 
C !         Soil evaporation after flood and mulch evaporation
C           IF (EOS_SOIL > 1.E-6) THEN
C             SELECT CASE(MESEV)
C !           ------------------------
C C            CASE ('R')  !Ritchie soil evaporation routine
C C!             Calculate the availability of soil water for use in SOILEV.
C C              DO L = 1, NLAYR
C C                SW_AVAIL(L) = MAX(0.0, SW(L) + SWDELTS(L) + SWDELTU(L))
C C              ENDDO
C C              CALL SOILEV(RATE,
C C     &          DLAYR, DUL, EOS_SOIL, LL, SW,             !Input
C C     &          SW_AVAIL(1), U, WINF,                     !Input
C C     &          ES)                                       !Output
C 
C !           ------------------------
C             CASE DEFAULT ! Sulieman-Ritchie soil evaporation routine is default
C !             Note that this routine calculates UPFLOW, unlike the SOILEV.
C !             Calculate the availability of soil water for use in SOILEV.
C               CALL ESR_SoilEvap(
C      &          EOS_SOIL, SOILPROP, SW, SWDELTS,          !Input
C      &          ES, ES_LYR, SWDELTU, UPFLOW)              !Output
C             END SELECT
C !           ------------------------
C           ENDIF
C 
C !         Total evaporation from soil, mulch, flood
C           EVAP = ES + EM + EF
C 
C !-----------------------------------------------------------------------
C !         Potential transpiration - model dependent
C !-----------------------------------------------------------------------
C           IF (XHLAI > 1.E-6) THEN
C             SELECT CASE (CONTROL % MODEL(1:5))
C             CASE ('RIORZ')    !ORYZA2000 Rice
C !             07/22/2011 CHP/TL replace TRANS with this (from ET2.F90 in ORYZA2000) 
C !             Estimate radiation-driven and wind- and humidity-driven part
C               ETRD = EO * 0.75  !s/b 1st term in FAO energy balance eqn
C               ETAE = EO - ETRD
C               EOP = ETRD*(1. - EXP(-KTRANS*XHLAI)) +ETAE*MIN(2.0, XHLAI)
C               EOP = MAX(0.0, EOP)
C               EOP = MIN(EO, EOP)
C 
C               TRAT = TRATIO(CROP, CO2, TAVG, WINDSP, XHLAI)
C               EOP = EOP * TRAT
C 
C             CASE DEFAULT
C !             For all models except ORYZA
C               CALL TRANS(RATE, 
C      &        CO2, CROP, EO, EVAP, KTRANS, TAVG,          !Input
C      &        WINDSP, XHLAI,                              !Input
C      &        EOP)                                        !Output
C             END SELECT
C             
C           ELSE
C             EOP = 0.0
C           ENDIF
C 
C !-----------------------------------------------------------------------
C !         ACTUAL TRANSPIRATION
C !-----------------------------------------------------------------------
C           IF (XHLAI .GT. 1.E-4 .AND. EOP .GT. 1.E-4) THEN
C             !These calcs replace the old SWFACS subroutine
C             !Stress factors now calculated as needed in PLANT routines.
C             EP = MIN(EOP, TRWUP*10.)
C           ELSE
C             EP = 0.0
C           ENDIF
C         ENDIF
C       ENDIF
C 
C !-----------------------------------------------------------------------
C !     ALTERNATE CALL TO ENERGY BALANCE ROUTINES
C !-----------------------------------------------------------------------
C       IF (CROP .NE. 'FA') THEN
C         IF (MEEVP .EQ. 'Z' .OR.
C      &        (MEPHO .EQ. 'L' .AND. XHLAI .GT. 0.0)) THEN
C           !ETPHOT called for photosynthesis only
C           !    (MEPHO = 'L' and MEEVP <> 'Z')
C           !or for both photosynthesis and evapotranspiration
C           !   (MEPHO = 'L' and MEEVP = 'Z').
C           CALL ETPHOT(CONTROL, ISWITCH,
C      &    PORMIN, PSTRES1, RLV, RWUMX, SOILPROP, ST, SW,  !Input
C      &    WEATHER, XLAI,                                 !Input
C      &    EOP, EP, ES, RWU, TRWUP)                        !Output
C           EVAP = ES  !CHP / BK 7/13/2017
C         ENDIF
C 
C !-----------------------------------------------------------------------
C !       ACTUAL ROOT WATER EXTRACTION
C !-----------------------------------------------------------------------
C         IF (ISWWAT .EQ. 'Y') THEN
C !         Adjust available soil water for evaporation
C           SELECT CASE(MESEV)
C           CASE ('R') !
C             SW_AVAIL(1) = MAX(0.0, SW_AVAIL(1) - 0.1 * ES / DLAYR(1))
C 
C           CASE DEFAULT
C             DO L = 1, NLAYR
C               SW_AVAIL(L) = MAX(0.0,SW_AVAIL(L) -0.1*ES_LYR(L)/DLAYR(L))
C             ENDDO
C           END SELECT
C 
C !         Calculate actual soil water uptake and transpiration rates
C           CALL XTRACT(
C      &      NLAYR, DLAYR, LL, SW, SW_AVAIL, TRWUP, UH2O,  !Input
C      &      EP, RWU,                                      !Input/Output
C      &      SWDELTX, TRWU)                                !Output
C         ENDIF   !ISWWAT = 'Y'
C       ENDIF
C 
C !     Transfer computed value of potential floodwater evaporation to
C !     flood variable.
C       FLOODWAT % EF = EF
C 
C !     Transfer data to storage routine
C       CALL PUT('SPAM', 'EF',  EF)
C       CALL PUT('SPAM', 'EM',  EM)
C       CALL PUT('SPAM', 'EO',  EO)
C       CALL PUT('SPAM', 'EP',  EP)
C       CALL PUT('SPAM', 'ES',  ES)
C       CALL PUT('SPAM', 'EOP', EOP)
C       CALL PUT('SPAM', 'EVAP',EVAP)
C 
C !***********************************************************************
C !***********************************************************************
C !     DAILY INTEGRATION
C !***********************************************************************
C       ELSEIF (DYNAMIC .EQ. INTEGR) THEN
C !-----------------------------------------------------------------------
C       IF (ISWWAT .EQ. 'Y') THEN
C !       Perform daily summation of water balance variables.
C         ET  = EVAP + EP
C         CEF = CEF + EF
C         CEM = CEM + EM
C         CEO = CEO + EO
C         CEP = CEP + EP
C         CES = CES + ES
C C JULY 11 2017, KB AND BK TO GET CUM ET OUT  
C             !    IF (MEEVP .EQ. 'Z') THEN
C             !      CET = CET + EP + ES
C             !    ELSE
C                   CET = CET + ET
C             !    ENDIF
C C KB 
C       ENDIF
C 
C       IF (IDETW .EQ. 'Y') THEN
C         CALL OPSPAM(CONTROL, ISWITCH, FLOODWAT, TRWU,
C      &    CEF, CEM, CEO, CEP, CES, CET, EF, EM, 
C      &    EO, EOP, EOS, EP, ES, ET, TMAX, TMIN, SRAD,
C      &    ES_LYR, SOILPROP)
C       ENDIF
C 
C !     Transfer data to storage routine
C       CALL PUT('SPAM', 'CEF', CEF)
C       CALL PUT('SPAM', 'CEM', CEM)
C       CALL PUT('SPAM', 'CEO', CEO)
C       CALL PUT('SPAM', 'CEP', CEP)
C       CALL PUT('SPAM', 'CES', CES)
C       CALL PUT('SPAM', 'CET', CET)
C       CALL PUT('SPAM', 'ET',  ET)
C 
C !***********************************************************************
C !***********************************************************************
C !     OUTPUT - daily output
C !***********************************************************************
C       ELSEIF (DYNAMIC .EQ. OUTPUT) THEN
C C-----------------------------------------------------------------------
C !     Flood water evaporation can be modified by Paddy_Mgmt routine.
C       EF = FLOODWAT % EF
C 
C !     ---------------------------------------------------------
C       IF (meevp .NE.'Z') THEN  !LPM 02dec14 to use the values from ETPHOT
C           SELECT CASE (METMP)
C C          CASE ('E')    !EPIC soil temperature routine
C C            CALL STEMP_EPIC(CONTROL, ISWITCH,  
C C     &        SOILPROP, SW, TAVG, TMAX, TMIN, TAV, WEATHER,   !Input
C C     &        SRFTEMP, ST)                                    !Output
C           CASE DEFAULT  !DSSAT soilt temperature
C             CALL STEMP(CONTROL, ISWITCH,
C      &        SOILPROP, SRAD, SW, TAVG, TMAX, XLAT, TAV, TAMP,!Input
C      &        SRFTEMP, ST)                                    !Output
C           END SELECT
C       ENDIF
C 
C !      SELECT CASE (METMP)
C !      CASE ('E')    !EPIC soil temperature routine
C !        CALL STEMP_EPIC(CONTROL, ISWITCH,  
C !     &    SOILPROP, SW, TAVG, TMAX, TMIN, TAV, WEATHER,   !Input
C !     &    SRFTEMP, ST)                                    !Output
C !      CASE DEFAULT  
C !!     7/21/2016 - DSSAT method is default, per GH
C !!     CASE ('D')  !DSSAT soil temperature
C !        CALL STEMP(CONTROL, ISWITCH,
C !     &    SOILPROP, SRAD, SW, TAVG, TMAX, XLAT, TAV, TAMP,!Input
C !     &    SRFTEMP, ST)                                    !Output
C !      END SELECT
C !
C       CALL OPSPAM(CONTROL, ISWITCH, FLOODWAT, TRWU,
C      &    CEF, CEM, CEO, CEP, CES, CET, EF, EM, 
C      &    EO, EOP, EOS, EP, ES, ET, TMAX, TMIN, SRAD,
C      &    ES_LYR, SOILPROP)
C 
C       IF (CROP .NE. 'FA' .AND. MEPHO .EQ. 'L') THEN
C         CALL ETPHOT(CONTROL, ISWITCH,
C      &    PORMIN, PSTRES1, RLV, RWUMX, SOILPROP, ST, SW,  !Input
C      &    WEATHER, XLAI,                                 !Input
C      &    EOP, EP, ES, RWU, TRWUP)                        !Output
C       ENDIF
C 
C !      CALL OPSTRESS(CONTROL, ET=ET, EP=EP)
C 
C !***********************************************************************
C !***********************************************************************
C !     SEASEND - seasonal output
C !***********************************************************************
C       ELSEIF (DYNAMIC .EQ. SEASEND) THEN
C C-----------------------------------------------------------------------
C       CALL OPSPAM(CONTROL, ISWITCH, FLOODWAT, TRWU,
C      &    CEF, CEM, CEO, CEP, CES, CET, EF, EM, 
C      &    EO, EOP, EOS, EP, ES, ET, TMAX, TMIN, SRAD,
C      &    ES_LYR, SOILPROP)
C 
C !     ---------------------------------------------------------
C       IF (meevp .NE.'Z') THEN  !LPM 02dec14 to use the values from ETPHOT
C           SELECT CASE (METMP)
C C          CASE ('E')    !EPIC soil temperature routine
C C            CALL STEMP_EPIC(CONTROL, ISWITCH,  
C C     &        SOILPROP, SW, TAVG, TMAX, TMIN, TAV, WEATHER,   !Input
C C     &        SRFTEMP, ST)                                    !Output
C           CASE DEFAULT  !DSSAT soilt temperature
C             CALL STEMP(CONTROL, ISWITCH,
C      &        SOILPROP, SRAD, SW, TAVG, TMAX, XLAT, TAV, TAMP,!Input
C      &        SRFTEMP, ST)                                    !Output
C           END SELECT
C       ENDIF
C 
C       IF (MEPHO .EQ. 'L') THEN
C         CALL ETPHOT(CONTROL, ISWITCH,
C      &    PORMIN, PSTRES1, RLV, RWUMX, SOILPROP, ST, SW,  !Input
C      &    WEATHER, XLAI,                                 !Input
C      &    EOP, EP, ES, RWU, TRWUP)                        !Output
C       ENDIF
C 
C !     Transfer data to storage routine
C       CALL PUT('SPAM', 'CEF', CEF)
C       CALL PUT('SPAM', 'CEM', CEM)
C       CALL PUT('SPAM', 'CEO', CEO)
C       CALL PUT('SPAM', 'CEP', CEP)
C       CALL PUT('SPAM', 'CES', CES)
C       CALL PUT('SPAM', 'CET', CET)
C       CALL PUT('SPAM', 'ET',  ET)
C 
C !      CALL OPSTRESS(CONTROL, ET=ET, EP=EP)

!***********************************************************************
!***********************************************************************
!     END OF DYNAMIC IF CONSTRUCT
!***********************************************************************
      ENDIF
!-----------------------------------------------------------------------

      write (*,11) "LEAVING SUBROUTINE SPAM"
 11   FORMAT(A)

      RETURN
      END SUBROUTINE SPAM

!-----------------------------------------------------------------------
!     VARIABLE DEFINITIONS: (updated 12 Feb 2004)
!-----------------------------------------------------------------------
! CANHT       Canopy height (m)
! CEF         Cumulative seasonal evaporation from floodwater surface (mm)
! CEM         Cumulative evaporation from surface mulch layer (mm)
! CEO         Cumulative potential evapotranspiration (mm)
! CEP         Cumulative transpiration (mm)
! CES         Cumulative evaporation (mm)
! CET         Cumulative evapotranspiration (mm)
! CLOUDS      Relative cloudiness factor (0-1) 
! CO2         Atmospheric carbon dioxide concentration
!              (µmol[CO2] / mol[air])
! CONTROL     Composite variable containing variables related to control 
!               and/or timing of simulation.    See Appendix A. 
! CROP        Crop identification code 
! DLAYR(L)    Thickness of soil layer L (cm)
! DUL(L)      Volumetric soil water content at Drained Upper Limit in soil 
!               layer L (cm3[water]/cm3[soil])
! EF          Evaporation rate from flood surface (mm / d)
! EM          Evaporation rate from surface mulch layer (mm / d)
! EO          Potential evapotranspiration rate (mm/d)
! EOP         Potential plant transpiration rate (mm/d)
! EORATIO     Ratio of increase in potential evapotranspiration with 
!               increase in LAI (up to LAI=6.0) for use with FAO-56 Penman 
!               reference potential evapotranspiration. 
! EOS         Potential rate of soil evaporation (mm/d)
! EP          Actual plant transpiration rate (mm/d)
! ES          Actual soil evaporation rate (mm/d)
! ET          Actual evapotranspiration rate (mm/d)
! FLOOD       Current depth of flooding (mm)
! FLOODWAT    Composite variable containing information related to bund 
!               management. Structure of variable is defined in 
!               ModuleDefs.for. 
! IDETW       Y=detailed water balance output, N=no detailed output 
! ISWITCH     Composite variable containing switches which control flow of 
!               execution for model.  The structure of the variable 
!               (SwitchType) is defined in ModuleDefs.for. 
! ISWWAT      Water simulation control switch (Y or N) 
! KSEVAP      Light extinction coefficient used for computation of soil 
!               evaporation 
! KTRANS      Light extinction coefficient used for computation of plant 
!               transpiration 
! LL(L)       Volumetric soil water content in soil layer L at lower limit
!              (cm3 [water] / cm3 [soil])
! MEEVP       Method of evapotranspiration ('P'=Penman, 
!               'R'=Priestly-Taylor, 'Z'=Zonal) 
! MEPHO       Method for photosynthesis computation ('C'=Canopy or daily, 
!               'L'=hedgerow or hourly) 
! NLAYR       Actual number of soil layers 
! PORMIN      Minimum pore space required for supplying oxygen to roots for 
!               optimal growth and function (cm3/cm3)
! RLV(L)      Root length density for soil layer L (cm[root] / cm3[soil])
! RWU(L)      Root water uptake from soil layer L (cm/d)
! RWUMX       Maximum water uptake per unit root length, constrained by 
!               soil water (cm3[water] / cm [root])
! MSALB       Soil albedo with mulch and soil water effects (fraction)
! SAT(L)      Volumetric soil water content in layer L at saturation
!              (cm3 [water] / cm3 [soil])
! SOILPROP    Composite variable containing soil properties including bulk 
!               density, drained upper limit, lower limit, pH, saturation 
!               water content.  Structure defined in ModuleDefs. 
! SRAD        Solar radiation (MJ/m2-d)
! SRFTEMP     Temperature of soil surface litter (°C)
! ST(L)       Soil temperature in soil layer L (°C)
! SUMES1      Cumulative soil evaporation in stage 1 (mm)
! SUMES2      Cumulative soil evaporation in stage 2 (mm)
! SW(L)       Volumetric soil water content in layer L
!              (cm3 [water] / cm3 [soil])
! SW_AVAIL(L) Soil water content in layer L available for evaporation, 
!               plant extraction, or movement through soil
!               (cm3 [water] / cm3 [soil])
! SWDELTS(L)  Change in soil water content due to drainage in layer L
!              (cm3 [water] / cm3 [soil])
! SWDELTU(L)  Change in soil water content due to evaporation and/or upward 
!               flow in layer L (cm3 [water] / cm3 [soil])
! SWDELTX(L)  Change in soil water content due to root water uptake in 
!               layer L (cm3 [water] / cm3 [soil])
! T           Number of days into Stage 2 evaporation (WATBAL); or time 
!               factor for hourly temperature calculations 
! TA          Daily normal temperature (°C)
! TAMP        Amplitude of temperature function used to calculate soil 
!               temperatures (°C)
! TAV         Average annual soil temperature, used with TAMP to calculate 
!               soil temperature. (°C)
! TAVG        Average daily temperature (°C)
! TMAX        Maximum daily temperature (°C)
! TMIN        Minimum daily temperature (°C)
! TRWU        Actual daily root water uptake over soil profile (cm/d)
! TRWUP       Potential daily root water uptake over soil profile (cm/d)
! U           Evaporation limit (cm)
! WINDSP      Wind speed at 2m (km/d)
! WINF        Water available for infiltration - rainfall minus runoff plus 
!               net irrigation (mm / d)
! XHLAI       Healthy leaf area index (m2[leaf] / m2[ground])
! XLAT        Latitude (deg.)
!-----------------------------------------------------------------------
!     END SUBROUTINE SPAM
!-----------------------------------------------------------------------

