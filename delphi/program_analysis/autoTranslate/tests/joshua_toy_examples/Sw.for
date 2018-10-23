!************************************************************************
!************************************************************************
*     Subroutine SW
!-----------------------------------------------------------------------
*     This subroutine calculates the soil water availability for the plant,
*     considering the rain, runoff, deep percolation (drainage) and water
*     use by the plant (evapotranspiration). It is divided in subroutines
*     that calculate those parameters separately. Daily data from climate
*     comes from WEATHER and daily LAI from PLANT subroutines. SW supplies
*     PLANT with daily soil water factor of availability (SWFAC)
*
!************************************************************************
*
*          LIST OF VARIABLES
*
*     CN      = runoff curve number
*     DATE    = date of irrigation applications (YYDDD)
*     DOY     = day of year
*     DP      = depth of the profile considered in the simulation (cm)
*     DRN     = daily subsurface drainage (mm)
*     DRNp    = daily drainage percentage (fraction of void space)
*     DYN     = dynamic control variable
*     EPa     = actual daily plant transpiration (mm)
*     EPp     = potential plant transpiration (mm)
*     ESa     = daily soil evaporation (mm)
*     ESp     = potential soil evaporation (mm)
*     ETp     = daily potential evapotranspiration (mm)
*     FC      = soil water storage at field capacity (mm)
*     FCp     = water content at field capacity (fraction of void space)
*     INF     = daily infiltration (mm)
*     IRR     = daily irrigation (mm)
*     LAI     = leaf area index (m2/m2)
*     POTINF  = potential infiltration (mm)
*     RAIN    = daily rainfall (mm)
*     ROF     = daily runoff (mm)
*     SRAD    = solar radiation (mj/m2/day)
*     ST      = soil water storage at saturation (mm)
*     STp     = water content saturation (fraction of void space)
*     SWC     = actual soil water storage in the profile (mm)
*     SWC_ADJ = cumulative adjustment factor for soil water content (mm)
*     SWC_INIT= initial soil water content (mm)
*     SWFAC1  = soil water deficit stress factor 
*     SWFAC2  = soil water excess stress factor
*     TDRN    = cumulative vertical drainage (mm)
*     TEPA    = cumulative plant transpiration (mm)
*     TESA    = cumulative soil evaporation (mm)
*     TINF    = cumulative infiltration (mm)
*     TIRR    = cumulative irrigation applied (mm)
*     TRAIN   = cumulative precipitation (mm)
*     TROF    = cumulative runoff (mm)
*     TMAX    = daily maximum temperature (c)
*     TMIN    = daily minimum temperature (c)
*     WP      = soil water storage at wilting point (mm)
*     WPp     = water content at wilting point (fraction of void space)

!************************************************************************

      SUBROUTINE SW(
     &    DOY, LAI, RAIN, SRAD, TMAX, TMIN,               !Input
     &    SWFAC1, SWFAC2,                                 !Output
     &    DYN)                                            !Control

!-----------------------------------------------------------------------
      IMPLICIT NONE 
      SAVE

      INTEGER   DATE, DOY
      REAL      SRAD,TMAX,TMIN,RAIN,SWC,INF,IRR,ROF,ESa,EPa,DRNp
      REAL      DRN,DP,WPp,FCp,STp,WP,FC,ST,ESp,EPp,ETp,LAI
      CHARACTER*10 DYN

      REAL CN, SWFAC1, SWFAC2, POTINF 
      REAL SWC_INIT, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
      REAL TINF, SWC_ADJ

!************************************************************************
!************************************************************************
!     INITIALIZATION
!************************************************************************
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
!************************************************************************
        OPEN(3,FILE='SOIL.INP',STATUS='UNKNOWN')
        OPEN(10,FILE='sw.out',  STATUS='REPLACE')
        OPEN(11,FILE='IRRIG.INP',STATUS='UNKNOWN')
        
        READ(3,10) WPp,FCp,STp,DP,DRNp,CN,SWC
   10   FORMAT(5X,F5.2,5X,F5.2,5X,F5.2,5X,F7.2,5X,F5.2,5X,F5.2,5X,F5.2)
        CLOSE(3)

        WRITE(10,15)
   15   FORMAT('Results of soil water balance simulation:',
     &    /,105X,'Soil',/,73X,'Pot.  Actual  Actual    Soil   Water',
     &    10X,'Excess',/,'  Day   Solar     Max     Min',42X,
     &    'Evapo-    Soil   Plant   Water Content Drought   Water',
     &    /,'   of    Rad.    Temp    Temp    Rain   Irrig  Runoff',
     &    '   Infil   Drain   Trans   Evap.  Trans. content   (mm3/',
     &    '  Stress  Stress',/,' Year (MJ/m2)    (oC)    (oC)    (mm)',
     &    '    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)',
     &    '    (mm)    mm3)  Factor  Factor')

        WP  = DP * WPp * 10.0  
        FC  = DP * FCp * 10.0   
        ST  = DP * STp * 10.0  
        SWC_INIT = SWC

        CALL RUNOFF(POTINF, CN, ROF, 'INITIAL   ')
        CALL STRESS(SWC, DP, FC, ST, WP, SWFAC1, SWFAC2, 
     &                'INITIAL   ')

!     Keep totals for water balance
        TRAIN = 0.0
        TIRR  = 0.0
        TESA  = 0.0
        TEPA  = 0.0
        TROF  = 0.0
        TDRN  = 0.0
        TINF  = 0.0
        SWC_ADJ = 0.0

!************************************************************************
!************************************************************************
!     RATE CALCULATIONS
!************************************************************************
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
!************************************************************************
        READ(11,25) DATE, IRR
   25   FORMAT(I5,2X,F4.1)

        TIRR = TIRR + IRR
        POTINF = RAIN + IRR
        TRAIN  = TRAIN + RAIN
        CALL DRAINE(SWC, FC, DRNp, DRN)

        IF (POTINF. GT. 0.0) THEN
          CALL RUNOFF(POTINF, CN, ROF, 'RATE      ')
          INF = POTINF - ROF
        ELSE
          ROF = 0.0
          INF = 0.0
        ENDIF

!     Potential evapotranspiration (ETp), soil evaporation (ESp) and
!       plant transpiration (EPp)
        CALL ETpS(SRAD,TMAX,TMIN,LAI,ETp)
        ESp = ETp * EXP(-0.7 * LAI)
        EPp = ETp * (1 - EXP(-0.7 * LAI))

!     Actual soil evaporation (ESa), plant transpiration (EPa)
        CALL ESaS(ESp,SWC,FC,WP,ESa)
        EPa = EPp * MIN(SWFAC1, SWFAC2)

!************************************************************************
!************************************************************************
!     INTEGRATION
!************************************************************************
      ELSEIF (INDEX(DYN,'INTEG') .NE. 0) THEN
!************************************************************************
        SWC = SWC + (INF - ESa - EPa - DRN)

        IF (SWC .GT. ST) THEN
          ROF = ROF + (SWC - ST)
          SWC = ST
        ENDIF

        IF (SWC .LT. 0.0) THEN
          SWC_ADJ =  SWC_ADJ - SWC
          SWC = 0.0
        ENDIF

        TINF = TINF + INF
        TESA = TESA + ESA
        TEPA = TEPA + EPA
        TDRN = TDRN + DRN
        TROF = TROF + ROF

        CALL STRESS(SWC, DP, FC, ST, WP, SWFAC1, SWFAC2, 'INTEG     ')

!************************************************************************
!************************************************************************
!     OUTPUT
!************************************************************************
      ELSEIF (INDEX(DYN,'OUTPUT    ') .NE. 0) THEN
!************************************************************************
        WRITE(10,40) DOY, SRAD, TMAX, TMIN, RAIN, IRR, ROF, INF, DRN, 
     &    ETP, ESa, EPa, SWC, SWC/DP, SWFAC1, SWFAC2
   40   FORMAT(I5,3F8.1,9F8.2,3F8.3)

!************************************************************************
!************************************************************************
!     CLOSE
!************************************************************************
      ELSEIF (INDEX(DYN,'CLOSE') .NE. 0) THEN
!************************************************************************
      CALL WBAL(SWC_INIT, SWC, TDRN, TEPA, 
     &    TESA, TIRR, TRAIN, TROF, SWC_ADJ, TINF)
      CLOSE(10) 
      CLOSE(11)

!************************************************************************
!************************************************************************
!     End of dynamic 'IF' construct
!************************************************************************
      ENDIF
!************************************************************************
      RETURN
      END SUBROUTINE SW
!************************************************************************



!************************************************************************
*     Subroutine DRAINE
*     Calculates vertical drainage.
!-----------------------------------------------------------------------
!     Input:  SWC, FC, DRNp
!     Output: DRN
!************************************************************************

      SUBROUTINE DRAINE(SWC,FC,DRNp,DRN)

!-----------------------------------------------------------------------
      IMPLICIT NONE
      SAVE
      REAL SWC, FC, DRN, DRNp
!-----------------------------------------------------------------------

      IF (SWC .GT. FC) THEN
         DRN = (SWC - FC) * DRNp
      ELSE
         DRN = 0
      ENDIF

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE DRAINE
!************************************************************************



!************************************************************************
*     Subroutine ESaS
*     Calculates the actual daily soil evaporation.
!-----------------------------------------------------------------------
*     Input:  SWC, WP, FC, ESp
*     Output: ESa
!************************************************************************

      SUBROUTINE ESaS(ESp,SWC,FC,WP,ESa)     

!-----------------------------------------------------------------------
      IMPLICIT NONE
      SAVE
      REAL a, SWC, WP, FC, ESa, ESp
!-----------------------------------------------------------------------
      IF (SWC .LT. WP) THEN
        a = 0
      ELSEIF (SWC .GT. FC) THEN
        a = 1
      ELSE 
        a = (SWC - WP)/(FC - WP)
      ENDIF
      
      ESa = ESp * a

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE ESAS
!************************************************************************



!************************************************************************
*     Subroutine ETpS
*     Calculates the daily potential evapotranspiration.
!-----------------------------------------------------------------------
*     Input:  LAI, TMAX, TMIN, SRAD
*     Output: ETp    
!************************************************************************
C
*     Local Variables
*     ALB  =  ALBEDO OF CROP-SOIL SURFACE
*     EEQ  =  EQUILIBRIUM EVAPOTRANSPIRATION (mm)
*     Tmed =  ESTIMATED AVERAGE DAILY TEMPERATURE (C)
*     f    =    

!-----------------------------------------------------------------------
      SUBROUTINE ETpS(SRAD,TMAX,TMIN,LAI,ETp)
      
!-----------------------------------------------------------------------
      IMPLICIT NONE  
      SAVE
      REAL    ALB,EEQ,f,Tmed,LAI
      REAL TMAX, TMIN, SRAD, ETP  

!-----------------------------------------------------------------------
      ALB =  0.1 * EXP(-0.7 * LAI) + 0.2 * (1 - EXP(-0.7 * LAI))
      Tmed = 0.6 * TMAX + 0.4 * TMIN
      EEQ = SRAD * (4.88E-03 - 4.37E-03 * ALB) * (Tmed + 29) 

      IF (TMAX .LT. 5) THEN
        f = 0.01 * EXP(0.18 *(TMAX + 20))
      ELSEIF (TMAX .GT. 35) THEN
        f = 1.1 + 0.05 * (TMAX - 35)
      ELSE 
        f = 1.1
      ENDIF

      ETp = f * EEQ
!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE ETPS
!************************************************************************



!************************************************************************
*     Subroutine RUNOFF
*     Calculates the daily runoff.
!************************************************************************
*     Input:  POTINF, CN
*     Output: ROF
!-----------------------------------------------------------------------
*     Local Variables
*     CN = CURVE NUMBER SCS EQUATION
*     S  = WATERSHED STORAGE SCS EQUATION (MM)

!-----------------------------------------------------------------------

      SUBROUTINE RUNOFF(POTINF, CN, ROF, DYN) 

!-----------------------------------------------------------------------
      IMPLICIT NONE  
      SAVE
      CHARACTER*10 DYN
      REAL S, CN
      REAL POTINF, ROF 

!************************************************************************
!************************************************************************
!     INITIALIZATION
!************************************************************************
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
!************************************************************************
        S = 254 * (100/CN - 1)
      
!************************************************************************
!************************************************************************
!     RATE CALCULATIONS
!************************************************************************
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
!************************************************************************
        IF (POTINF .GT. 0.2 * S)  THEN
          ROF = ((POTINF - 0.2 * S)**2)/(POTINF + 0.8 * S)
        ELSE
          ROF = 0
        ENDIF

!************************************************************************
!************************************************************************
!     End of dynamic 'IF' construct
!************************************************************************
      ENDIF
!************************************************************************
      RETURN
      END SUBROUTINE RUNOFF
!************************************************************************



!************************************************************************
*     Sub-subroutine STRESS calculates soil water stresses.
*     Today's stresses will be applied to tomorrow's rate calcs.
!-----------------------------------------------------------------------  
*     Input:  SWC, DP, FC, ST, WP
*     Output: SWFAC1, SWFAC2
!************************************************************************
      SUBROUTINE STRESS(SWC, DP, FC, ST, WP, SWFAC1, SWFAC2, DYN)

!-----------------------------------------------------------------------  
      IMPLICIT NONE
      SAVE
      CHARACTER*10 DYN
      REAL FC, ST, SWC, WP, SWFAC2, SWFAC1
      REAL DP, DWT, WTABLE, THE
      REAL, PARAMETER :: STRESS_DEPTH = 250   !Water table depth below 
                                           !which no stress occurs (mm)
!************************************************************************
!************************************************************************
!     INITIALIZATION
!************************************************************************
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
!************************************************************************
        THE = WP + 0.75 * (FC - WP)   !threshold for drought stress (mm)

      ENDIF
      
!************************************************************************
!************************************************************************
!     INTEGRATION (also done for initialization)
!************************************************************************
      IF (INDEX(DYN,'INTEG') .NE. 0 .OR. INDEX(DYN,'INITIAL') .NE. 0) 
     &    THEN
!************************************************************************
!     Drought stress factor - SWFAC1
!-----------------------------------------------------------------------
        IF (SWC .LT. WP) THEN
          SWFAC1 = 0.0
        ELSEIF (SWC .GT. THE) THEN
          SWFAC1 = 1.0
        ELSE
          SWFAC1 = (SWC - WP) / (THE - WP)
          SWFAC1 = MAX(MIN(SWFAC1, 1.0), 0.0)
        ENDIF

!-----------------------------------------------------------------------
!     Excess water stress factor - SWFAC2 
!-----------------------------------------------------------------------  
        IF (SWC .LE. FC) THEN
          WTABLE = 0.0
          DWT = DP * 10.              !DP in cm, DWT in mm
          SWFAC2 = 1.0
        ELSE
      !FC water is distributed evenly throughout soil profile.  Any
      !  water in excess of FC creates a free water surface
      !WTABLE - thickness of water table (mm)
      !DWT - depth to water table from surface (mm)
          WTABLE = (SWC - FC) / (ST - FC) * DP * 10.
          DWT = DP * 10. - WTABLE     

          IF (DWT .GE. STRESS_DEPTH) THEN
            SWFAC2 = 1.0
          ELSE 
            SWFAC2 = DWT / STRESS_DEPTH
          ENDIF
          SWFAC2 = MAX(MIN(SWFAC2, 1.0), 0.0)
        ENDIF

!************************************************************************
!************************************************************************
!     End of dynamic 'IF' construct
!************************************************************************
      ENDIF
!************************************************************************
      RETURN
      END SUBROUTINE STRESS
!************************************************************************



!************************************************************************
*     Subroutine WBAL
*     Seasonal water balance
!-----------------------------------------------------------------------
!     Input:  SWC, SWC_INIT, TDRN, TEPA, 
!                 TESA, TIRR, TRAIN, TROF
!     Output: None
!************************************************************************

      SUBROUTINE WBAL(SWC_INIT, SWC, TDRN, TEPA, 
     &    TESA, TIRR, TRAIN, TROF, SWC_ADJ, TINF)

!-----------------------------------------------------------------------
      IMPLICIT NONE
      SAVE
      INTEGER, PARAMETER :: LSWC = 21
      REAL SWC, SWC_INIT
      REAL TDRN, TEPA, TESA, TIRR, TRAIN, TROF
      REAL WATBAL, SWC_ADJ, TINF
      REAL CHECK
!-----------------------------------------------------------------------
      OPEN (LSWC, FILE = 'WBAL.OUT', STATUS = 'REPLACE')

      WATBAL = (SWC_INIT - SWC) + (TRAIN + TIRR) - 
!      0.0   =(Change in storage)+  (Inflows)    -
     &            (TESA + TEPA + TROF + TDRN)
!                         (Outflows)

      WRITE(*,100)   SWC_INIT, SWC, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
      WRITE(LSWC,100)SWC_INIT, SWC, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
  100 FORMAT(//, 'SEASONAL SOIL WATER BALANCE', //,
     &       'Initial soil water content (mm):',F10.3,/,
     &       'Final soil water content (mm):  ',F10.3,/,
     &       'Total rainfall depth (mm):      ',F10.3,/,
     &       'Total irrigation depth (mm):    ',F10.3,/,
     &       'Total soil evaporation (mm):    ',F10.3,/,
     &       'Total plant transpiration (mm): ',F10.3,/,
     &       'Total surface runoff (mm):      ',F10.3,/,
     &       'Total vertical drainage (mm):   ',F10.3,/)

      IF (SWC_ADJ .NE. 0.0) THEN
        WRITE(*,110) SWC_ADJ
        WRITE(LSWC,110) SWC_ADJ
  110   FORMAT('Added water for SWC<0 (mm):     ',E10.3,/)
      ENDIF

      WRITE(*   ,200) WATBAL
      WRITE(LSWC,200) WATBAL
  200 FORMAT('Water Balance (mm):             ',F10.3,//)

      CHECK = TRAIN + TIRR - TROF
      IF ((CHECK - TINF) .GT. 0.0005) THEN
        WRITE(*,300) CHECK, TINF, (CHECK - TINF)
        WRITE(LSWC,300) CHECK, TINF, (CHECK - TINF)
  300   FORMAT(/,'Error: TRAIN + TIRR - TROF = ',F10.4,/,
     &           'Total infiltration =         ',F10.4,/,
     &           'Difference =                 ',F10.4)
      ENDIF

      CLOSE (LSWC)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE WBAL



!************************************************************************
!************************************************************************

