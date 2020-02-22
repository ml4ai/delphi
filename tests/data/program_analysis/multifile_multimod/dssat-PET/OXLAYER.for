C=======================================================================
C  OXLAYER, Subroutine, U. Singh
C  Determines Oxidised Layer chemistry
C-----------------------------------------------------------------------
C  REVISION HISTORY
C             US  Written
C  03/29/2002 CHP modular format
!  03/07/2005 CHP Fixed check for LDF10
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil 
!                 water effects on albedo)
C=======================================================================

      SUBROUTINE OXLAYER (CONTROL,
     &    BD1, ES, FERTDATA, FLOODWAT, LFD10,             !Input
     &    NSWITCH, SNH4, SNO3, SOILPROP, SRAD, ST,        !Input
     &    SW, TMAX, TMIN, UREA, XHLAI,                    !Input
     &    DLTSNH4, DLTSNO3, DLTUREA, OXLAYR,              !I/O
     &    ALI, TOTAML)                                    !Output

      USE ModuleDefs
      USE FloodModule
      IMPLICIT NONE
      SAVE

      CHARACTER*1 RNMODE

      INTEGER  NSWITCH, YRDOY, YRDRY, DYNAMIC
      INTEGER  RUN

      REAL    FPI, BD1, SW1
      REAL    ALGACT,AMLOSS,OXRNTRF
      REAL    ALI
      REAL    OXLT,OXFAC,OXU,OXN3,OXN3C,OXH4,OXH4C
      REAL    OXN,OXNI
      REAL    TOTAML
      INTEGER IHDAY,LFD10, IBP

!     UNINCO: Fertilizer not fully incorporated; if true, ignore oxlayer
      LOGICAL DailyCalc,UNINCO
      REAL    WFP
      REAL    OXMIN3, OXMIN4
      REAL    KG2PPM(NL)
      REAL    SURCEC

!     Local Variables
      INTEGER  IST,IHR,I,K
      REAL     SURAD,OXALI,STI,SWI,AMPES,AK,HTMFAC,MF,TK
      REAL     STEMP,HES,PHSHIFT,TEMPFU,XL,XL2,SWF,UALGCT,OXUHYDR,OXPH1
      REAL     OXUHYDC,OXUHYDM,PHOXUHY,OXPH,OXMINC,OXH3C,OXH3,WIND
      REAL     ALOGHK,HK,OXH3M,OXH3P,AMLOS1,GLOS1,TFACTOR,OXNC,WF2
      REAL     PHFACT,RNTRFI,HOXRNT,ELAG

!     Passed Variables
      INTEGER  FERTYPE
      REAL     UREA(NL),SNH4(NL),SNO3(NL),SRAD,MSALB, XHLAI
      REAL     SW(NL),SAT(NL),ST(NL),DUL(NL),ES,TMIN,TMAX
      REAL     LL(NL),PH(NL),DLAYR(NL)
      REAL     OC(NL)

!     Additions to oxidation layer from today's fertilizer:
      REAL ADDOXH4, ADDOXN3, ADDOXU

      TYPE (ControlType)  CONTROL
      TYPE (SoilType)     SOILPROP
      TYPE (FloodWatType) FloodWat
      TYPE (OxLayerType)  OXLAYR
      TYPE (FertType)     FERTDATA

      REAL DLTUREA(NL),  DLTSNO3(NL),  DLTSNH4(NL)
      REAL DLTOXU, DLTOXH4, DLTOXN3
      REAL TMPUREA, TMPNH4,  TMPNO3
      REAL TMPOXU, TMPOXH4, TMPOXN3

!-----------------------------------------------------------------------
      DYNAMIC = CONTROL % DYNAMIC
      RNMODE  = CONTROL % RNMODE 
      RUN     = CONTROL % RUN    
      YRDOY   = CONTROL % YRDOY  

      YRDRY = FLOODWAT % YRDRY

      DLAYR = SOILPROP % DLAYR
      SURCEC= SOILPROP % CEC(1)
      DUL   = SOILPROP % DUL
      KG2PPM= SOILPROP % KG2PPM
      LL    = SOILPROP % LL
      OC    = SOILPROP % OC
      PH    = SOILPROP % PH
!     SALB  = SOILPROP % SALB
      MSALB  = SOILPROP % MSALB
      SAT   = SOILPROP % SAT

      SW1 = SW(1)

      DailyCalc   = OXLAYR % DailyCalc
      DLTOXH4 = OXLAYR % DLTOXH4
      DLTOXN3 = OXLAYR % DLTOXN3
      DLTOXU  = OXLAYR % DLTOXU
      OXLT    = OXLAYR % OXLT
      OXH4    = OXLAYR % OXH4
      OXN3    = OXLAYR % OXN3
      OXU     = OXLAYR % OXU
      OXMIN3  = OXLAYR % OXMIN3
      OXMIN4  = OXLAYR % OXMIN4
      IBP     = OXLAYR % IBP

      ADDOXH4 = FERTDATA % ADDOXH4
      ADDOXN3 = FERTDATA % ADDOXN3
      ADDOXU  = FERTDATA % ADDOXU
      FERTYPE = FERTDATA % FERTYPE
!      LFD10   = FERTDATA % LFD10
      UNINCO  = FERTDATA % UNINCO

!***********************************************************************
!***********************************************************************
!     Seasonal initialization - run once per season
!***********************************************************************
      IF (DYNAMIC .EQ. SEASINIT) THEN
!-----------------------------------------------------------------------
      FPI = 0.50

      !Initialize for non-sequenced runs
      IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN .EQ. 1) THEN

        OXLT  = 0.50 - 0.1*OC(1)
        OXLT  = MAX(0.001,OXLT)
        OXLAYR % OXLT = OXLT
        OXFAC = 1.0/(BD1*OXLT*1.0E-01)

        OXU   = UREA(1) * KG2PPM(1) / OXFAC
        OXN3  = SNO3(1) * KG2PPM(1) / OXFAC
        OXH4  = SNH4(1) * KG2PPM(1) / OXFAC
        OXNI  = 0.05

        TOTAML = 0.0
        DailyCalc = .TRUE.

        OXPH = PH(1)      !chp 1/27/2004
      ENDIF

      ALGACT = 0.1

!***********************************************************************
!***********************************************************************
!     Daily rate calculations
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. RATE) THEN
!-----------------------------------------------------------------------
      OXFAC = 1.0/(BD1*OXLT*1.0E-01)

      TMPUREA = UREA(1) + DLTUREA(1)
      TMPNO3  = SNO3(1) + DLTSNO3(1)
      TMPNH4  = SNH4(1) + DLTSNH4(1)

      TMPOXU  = OXU  + DLTOXU  + ADDOXU
      TMPOXN3 = OXN3 + DLTOXN3 + ADDOXN3 
      TMPOXH4 = OXH4 + DLTOXH4 + ADDOXH4

      !FAC1 = 1.0 / (BD1 * 1.E-01 * DLAYR(1))
      !NO3_1  = SNO3(1) * FAC1
      !NH4_1  = SNH4(1) * FAC1
      !UPPM_1 = UREA(1) * FAC1

      OXH4C   = TMPOXH4   * OXFAC
      OXN3C   = TMPOXN3   * OXFAC

!     If fertilizer not incorporated, ignore oxidation layer 
!         (UNINCO set in FPLACE)
      IF (.NOT. UNINCO) RETURN

C!     LFD10 - 10 days after last fertilizer application (set in FPLACE)
C      IF (YRDOY < LFD10) THEN
C         DailyCalc = .FALSE.
C      ELSE
C         DailyCalc = .TRUE.
C      ENDIF
C
C      TMPOXU    = AMIN1 (TMPOXU , TMPUREA)
C      TMPOXH4   = AMIN1 (TMPOXH4, TMPNH4)
C      TMPOXN3   = AMIN1 (TMPOXN3, TMPNO3)
C
C!     Compute algal activity - for saturated soil
C!      SURAD = SRAD*(1.0-SALB)*EXP(-0.85*XHLAI)
C      SURAD = SRAD*(1.0-MSALB)*EXP(-0.85*XHLAI)
C      OXALI = 1.0 - EXP(-SURAD/10.0)
C
C      IF (SW(1) .EQ. SAT(1)) THEN
C         ALI = 1.0 - EXP(-SURAD/10.0)
C      ELSE
C         ALI = 1.0
C      ENDIF
C
C      IF (ST(1) .LT. 30.0)  THEN
C         STI = (ST(1)-15.0)*0.1
C      ELSEIF (ST(1) .GE. 30.0) THEN
C         STI = 1.0-(ST(1)-30.0)*0.05
C      ENDIF
C
C      STI = AMIN1 (STI,1.0)
C      SWI = 1.0
C      IF (SW(1) .LT. DUL(1)) THEN
C         SWI = (SAT(1)-DUL(1))/(SAT(1)-SW(1))
C      ENDIF
C
C!     Biological and chemical activity of "OXIDIZED" layer
C      ELAG   = AMIN1(OXNI,FPI,ALI,STI,SWI)
C      ALGACT = ELAG*(4.0-ALGACT)*ALGACT           ! 4->3.5
C      IF (XHLAI .GT. 1.5) THEN
C         ALGACT = AMIN1(ALGACT,ALI)
C      ENDIF
C      ALGACT = AMIN1 (ALGACT,1.00)
C      ALGACT = AMAX1 (ALGACT,0.05)
C
C!     Partition soil evaporation
C!     AMPES(hr) is 0.35*ES(daily) chamged to 0.45
C      AMPES   = 0.38*ES                           ! 0.35->0.38
C      AMLOSS  = 0.0
C      OXRNTRF = 0.0
C      IF (ALGACT .GE. 0.2 .OR. YRDOY .LT. LFD10) THEN
C        DailyCalc = .FALSE.
C      ENDIF
C
C      IF (DailyCalc) THEN
C         IHDAY = 1
C         IST   = 6
C         IHR   = 6
C      ELSE
C         IHDAY = 12
C         IST   = 1
C         IHR   = 12
C      ENDIF
C
C!     Potential Hydrolysis
C!     AK = -1.12+0.31*OC(1)+0.203*PH(1)-0.0355*OC(1)*PH(1)
C      AK = 0.25+0.075*OC(1)
C      AK = AMAX1 (AK,0.25)
C      AK = AMIN1 (AK,1.00)
C
C      DO I = IST, IHR
C         K      = 7 - I
C         IF (I .GT. 6) THEN
C            K = I - 6
C         ENDIF
C         HTMFAC = 0.931+0.114*K-0.0703*K**2+0.0053*K**3
C
C!        Hourly soil surface temp
C         STEMP  = TMIN + HTMFAC*(TMAX+2.0-TMIN)
C         IF (I .EQ. 13) THEN
C            STEMP = TMIN
C         ENDIF
C
C!        Calculate hourly soil evaporation (HES)
C         IF (I .LE. 6 .OR. I .GT. 9) THEN
C            HES = AMPES*SIN(3.141593*FLOAT(I)/12.0)+0.08    ! 10->9
C         ENDIF
C
C         HES = AMAX1 (ES/24.0,ABS(HES))
C
C         ! (1)  Calculate indices for bio-chemical activity
C         IF (I. EQ. 6) THEN
C             OXNI = (OXH4C+OXN3C)/10.0+0.10
C         ENDIF
C         OXNI = AMIN1 (OXNI,1.0)
C
C         ! BIOACT = AMIN1(1.0,5.0*ALGACT)
C         PHSHIFT = 0.75 + 2.0*ALGACT
C
C         SELECT CASE (FERTYPE)
C           CASE (0:3,8,11:50)
C             IF (OXH4C .LE. OXN3C) THEN
C                PHSHIFT = 0.6
C             ENDIF
C           CASE DEFAULT
C             ! Just go on to UREA hydrolysis
C         END SELECT
C
C!        UREA hydrolysis function of surface layer OC or
C!        biological activity, whichever is greater.
C         TEMPFU  = 0.04*STEMP - 0.2
C         TEMPFU  = AMIN1 (TEMPFU,1.0)
C         TEMPFU  = AMAX1 (TEMPFU,0.0)
C         XL      = DUL(1) - LL(1)*0.5
C         XL2     = SW(1)  - LL(1)*0.5
C         MF      = XL2/XL
C         MF      = AMAX1 (MF,0.0)
C         SWF     = MF + 0.20
C         IF (SW(1) .GE. DUL(1)) THEN
C            SWF = 1.0
C         ENDIF
C         SWF     = AMIN1 (SWF,1.0)
C         SWF     = AMAX1 (SWF,0.0)
C         UALGCT  = 0.25*ALGACT
C         OXUHYDR = AMAX1 (AK,UALGCT)*AMIN1 (SWF,TEMPFU)*TMPOXU
C
C!        CHP added this check, but still get negative urea 
C!        need to overhaul this routine - prevent negative values for 
C!        all N species.
C         OXUHYDR = AMIN1(TMPUREA, OXUHYDR)  !CHP 5/4/2010
C
C         TMPOXU     = TMPOXU     - OXUHYDR
C         TMPOXH4    = TMPOXH4    + OXUHYDR
C         TMPUREA = TMPUREA - OXUHYDR
C         TMPNH4  = TMPNH4  + OXUHYDR
C
C         IF (I .LE. 6) THEN
C            OXPH1 = PH(1)+PHSHIFT*SIN(3.1412*FLOAT(I)/12.0)   ! 8->11
C         ENDIF
C
C!        Add effects of UREA hydrolysis on surface layer pH here
C         IF (MF .GT. 1.5 .AND. OXPH1 .LT. 7.2) THEN
C            OXPH1 = 7.2
C         ENDIF
C         IF (OXUHYDR .GT. 0.001) THEN
C            OXUHYDC = OXUHYDR*OXFAC
C            OXUHYDM = OXUHYDC*0.001/14.0          ! (Molar conc.)
C            PHOXUHY = AMIN1 (10.0,-LOG10(OXUHYDM))
C            OXPH    = OXPH1 + OXALI*(10.0-PHOXUHY)/10.0
C         ENDIF
C         OXPH = AMIN1 (OXPH,9.0)
C         OXPH = AMAX1 (OXPH,PH(1))
C
C!        AMMONIA loss routine ... calculate surface layer NH3
C         TK     = STEMP + 273.15
C         OXH4C  = TMPOXH4  * OXFAC
C         OXMINC = OXMIN4* OXFAC
C
C!        CALL AMTHERM (OXH4C,OXH4SC,BD(1),CEC,2,BPOXL,OXMINC)
C!        OXH4SC = AMIN1(OXH4SC/DUL(1),OXH4C)
C         IF (I .LE. 6 .OR. OXH3C .GE. OXH4C) THEN
C            OXH3C = OXH4C/(1.0+10.0**(0.09018+2729.92/TK-OXPH))
C         ENDIF
C
C!        Calculate ammonia (Mass) using inverse of (KG2PPM) OXLT in cm)
C         IF (OXH3C .LE. 0.00001 .AND. OXH3C .GT. 0.0) THEN
C            OXH3C = 0.0
C         ENDIF
C         OXH3    = OXH3C/OXFAC
C         IF (OXH3 .GT. (TMPOXH4-OXMIN4)) THEN
C            OXH3  = TMPOXH4 - OXMIN4
C            OXH3C = OXH3 * OXFAC
C         ENDIF
C
C         WIND    = 7.15*HES           ! 7.15 -> 5.75
C         ALOGHK  = 158.559-8621.06/TK-25.6767*ALOG(TK)+0.035388*TK 
C         HK      = EXP(ALOGHK)
C         OXH3M   = OXH3C*0.001/14.0
C         OXH3P   = 10.0*OXH3M/HK                 
C         IF (OXH3P .GT. 0.0) THEN
C            AMLOS1  = 0.0012*OXH3P+0.0014*WIND+0.00086*OXH3P**2*WIND
C         ENDIF
C         IF (NSWITCH .EQ. 8) THEN
C            AMLOS1 = 0.0
C         ENDIF
C         IF (OXH3P .LE. 0.0) THEN
C            AMLOS1 = 0.0
C         ENDIF
C         GLOS1   = AMLOS1
C         AMLOS1  = AMIN1 (AMLOS1,TMPOXH4-OXMIN4)
C         AMLOSS  = AMLOSS + AMLOS1
C         TOTAML  = TOTAML + AMLOS1
C         TMPOXH4    = TMPOXH4   - AMLOS1
C         OXH4C   = TMPOXH4   * OXFAC
C         TMPNH4  = TMPNH4 - AMLOS1
C
C!        Nitrification section
C         OXN     = TMPOXH4 + TMPOXU + TMPOXN3
C         OXNC    = OXN*OXFAC
C         TFACTOR = EXP(-6572 / TK + 21.4)
C
C         WFP = SW(1) / SAT(1)
C         IF (SW(1) .GT. DUL(1)) THEN
C             WF2 = -2.5 * WFP + 2.55
C         ELSE
C             IF (WFP .GT. 0.4) THEN
C                WF2 = 1.0
C             ELSE
C                WF2 = 3.15 * WFP - 0.1
C             ENDIF
C         ENDIF
C         WF2     = AMAX1 (WF2, 0.0)
C         WF2     = AMIN1 (1.0, WF2)
C         PHFACT  = 0.3*OXPH - 1.8
C         PHFACT  = AMIN1 (PHFACT,1.0)
C         PHFACT  = AMAX1 (PHFACT,0.0)
C         RNTRFI  = TFACTOR*WF2*PHFACT/13.0
C         RNTRFI  = AMIN1 (RNTRFI,0.90)
C         HOXRNT  = RNTRFI*TMPOXH4*OXLT/DLAYR(1)
C         IF (OXMIN4 .GT. TMPOXH4) THEN
C            TMPOXH4   = OXMIN4
C         ENDIF
C         IF (TMPOXH4-HOXRNT .LE. OXMIN4) THEN
C            HOXRNT = TMPOXH4 - OXMIN4
C         ENDIF
C         IF (HOXRNT .GT. -0.00001 .AND. HOXRNT .LT. 0.00001) THEN
C            HOXRNT = 0.0
C         ENDIF
C
C!        HOXRNT  = AMIN1 (HOXRNT,SNH4(1)-SMIN4(1))
C         HOXRNT  = AMIN1 (HOXRNT,TMPOXH4)
C         TMPNH4  = TMPNH4  - HOXRNT
C         TMPNO3  = TMPNO3  + HOXRNT
C         TMPOXH4    = TMPOXH4    - HOXRNT
C         TMPOXN3    = TMPOXN3    + HOXRNT
C         OXRNTRF = OXRNTRF + HOXRNT
C
C!        Recompute equilibria after nitrification/denitrification
C         CALL EQUIL2 (
C     &     BD1, SURCEC, DLAYR, 0.0, IHDAY, 1,          !Input
C     &     OXLT, OXMIN3, OXMIN4, SW1, YRDOY, YRDRY,    !Input
C     &     IBP, 0.0, TMPOXU, TMPUREA)                  !I/O 
C
C         CALL EQUIL2 (
C     &     BD1, SURCEC, DLAYR, 0.0, IHDAY, 2,          !Input
C     &     OXLT, OXMIN3, OXMIN4, SW1, YRDOY, YRDRY,    !Input
C     &     IBP, 0.0, TMPOXN3, TMPNO3)                  !I/O 
C
C         CALL EQUIL2 (
C     &     BD1, SURCEC, DLAYR, 0.0, IHDAY, 3,          !Input
C     &     OXLT, OXMIN3, OXMIN4, SW1, YRDOY, YRDRY,    !Input
C     &     IBP, 0.0, TMPOXH4, TMPNH4)                  !I/O 
C      END DO
C
C!     Surface variables
C      DLTUREA(1) = MAX(0.0, TMPUREA) - UREA(1)
C      DLTSNO3(1) = MAX(0.0, TMPNO3)  - SNO3(1)
C      DLTSNH4(1) = MAX(0.0, TMPNH4)  - SNH4(1)
C
C      DLTOXU  = MAX(0.0, TMPOXU)  - OXU
C      DLTOXH4 = MAX(0.0, TMPOXH4) - OXH4
C      DLTOXN3 = MAX(0.0, TMPOXN3) - OXN3

!***********************************************************************
!***********************************************************************
!     DAILY INTEGRATION (also performed for seasonal initialization)
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. INTEGR) THEN
!-----------------------------------------------------------------------
!     Oxidation layer integration
      OXU  = OXU  + DLTOXU
      OXH4 = OXH4 + DLTOXH4
      OXN3 = OXN3 + DLTOXN3

      OXLAYR % OXU  = OXU 
      OXLAYR % OXH4 = OXH4
      OXLAYR % OXN3 = OXN3

      DLTOXU  = 0.0
      DLTOXH4 = 0.0
      DLTOXN3 = 0.0

!***********************************************************************
!***********************************************************************
!     END OF DYNAMIC IF CONSTRUCT
!***********************************************************************
      ENDIF
!***********************************************************************
      OXLAYR % DailyCalc   = DailyCalc
      OXLAYR % DLTOXH4 = DLTOXH4
      OXLAYR % DLTOXN3 = DLTOXN3
      OXLAYR % DLTOXU  = DLTOXU
      OXLAYR % OXU  = OXU 
      OXLAYR % OXH4 = OXH4
      OXLAYR % OXN3 = OXN3
      OXLAYR % IBP  = IBP

      RETURN
      END SUBROUTINE OXLAYER

