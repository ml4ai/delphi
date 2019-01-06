      PROGRAM MAIN
      IMPLICIT NONE
      REAL LAI, SWFAC1, SWFAC2
      REAL SRAD, TMAX, TMIN, PAR, RAIN
      INTEGER DOY,DOYP, endsim
      INTEGER FROP, IPRINT
      CALL OPENF(DOYP, FROP)
      CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'INITIAL   ')
      CALL SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,'INITIAL   ')
      CALL PLANT(DOY, endsim, TMAX, TMIN,PAR, SWFAC1, SWFAC2,LAI,'INITIAL   ')
      DO 500 DOY = 0,1000
        IF (DOY .NE. 0) THEN
          CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'RATE      ')
          CALL SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,'RATE      ')
          IF (DOY .GT. DOYP) THEN
            CALL PLANT(DOY, endsim,TMAX,TMIN,PAR, SWFAC1, SWFAC2,LAI,'RATE      ')
          ENDIF
          CALL SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,'INTEG     ')
          IF (DOY .GT. DOYP) THEN
            CALL PLANT(DOY, endsim, TMAX,TMIN,PAR, SWFAC1, SWFAC2,LAI,'INTEG     ')
          ENDIF
        ENDIF
        IPRINT = MOD(DOY, FROP)
        IF ((IPRINT .EQ. 0) .OR. (endsim .EQ. 1) .OR.(DOY .EQ. DOYP)) THEN
          CALL SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,'OUTPUT    ')
          IF (DOY .GE. DOYP) THEN
            CALL PLANT(DOY, endsim, TMAX,TMIN,PAR, SWFAC1, SWFAC2,LAI,'OUTPUT    ')
          ENDIF
        ENDIF
        IF (ENDSIM .EQ. 1) EXIT
  500 ENDDO
      CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'CLOSE     ') 
      CALL SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,'CLOSE     ')
      CALL PLANT(DOY, endsim, TMAX,TMIN,PAR, SWFAC1, SWFAC2,LAI,'CLOSE     ') 
      STOP
      END PROGRAM MAIN
      SUBROUTINE OPENF(DOYP, FROP)
      IMPLICIT NONE
      INTEGER DOYP, FROP
      OPEN (UNIT=8, FILE='SIMCTRL.INP',STATUS='UNKNOWN')
      READ(8,5) DOYP, FROP
      IF (FROP .LE. 0) FROP = 1
    5 FORMAT(2I6)
      CLOSE(8)
      RETURN
      END SUBROUTINE OPENF
      SUBROUTINE PLANT(DOY, endsim,TMAX,TMIN, PAR, SWFAC1, SWFAC2,LAI,DYN)
      IMPLICIT NONE 
      SAVE 
      REAL E,Fc,Lai, nb,N,PT,Pg, di,PAR
      REAL rm,dwf,int, TMAX,TMIN, p1, sla, LAI
      REAL PD,EMP1,EMP2,Lfmax,dwc, TMN
      REAL dwr,dw,dn,w,wc,wr,wf,tb,intot, dLAI, FL
      INTEGER DOY, endsim, COUNT
      CHARACTER*10 DYN
      REAL SWFAC1, SWFAC2  
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
        endsim = 0
        OPEN (2,FILE='PLANT.INP',STATUS='UNKNOWN')
        OPEN (1,FILE='plant.out',STATUS='REPLACE')
        READ(2,10) Lfmax, EMP2,EMP1,PD,nb,rm,fc,tb,intot,n,lai,w,wr,wc,p1,sla
   10   FORMAT(17(1X,F7.4))
        CLOSE(2)
        WRITE(1,11)
        WRITE(1,12)
   11   FORMAT('Results of plant growth simulation: ')
   12   FORMAT(//,'                Accum',/,'       Number    Temp                                    Leaf',/,'  Day      of  during   Plant  Canopy    Root   Fruit    Area',/,'   of    Leaf  Reprod  Weight  Weight  Weight  weight   Index',/,' Year   Nodes    (oC)  (g/m2)  (g/m2)  (g/m2)  (g/m2) (m2/m2)',/,' ----  ------  ------  ------  ------  ------  ------  ------')
        WRITE(*,11)
        WRITE(*,12)
        COUNT = 0
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
        TMN = 0.5 * (TMAX + TMIN)
        CALL PTS(TMAX,TMIN,PT)
        CALL PGS(SWFAC1, SWFAC2,PAR, PD, PT, Lai, Pg)
        IF (N .LT. Lfmax) THEN
          FL = 1.0
          E  = 1.0
          dN = rm * PT
          CALL LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,dN,p1, sla, dLAI)
          dw = E * (Pg) * PD
          dwc = Fc * dw
          dwr = (1-Fc) * dw
          dwf = 0.0
        ELSE
          FL = 2.0
          IF (TMN .GE. tb .and. TMN .LE. 25) THEN
            di = (TMN-tb)
          ELSE 
            di = 0.0
          ENDIF
          int = int + di
          E = 1.0
          CALL LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,dN,p1, sla, dLAI)
          dw = E * (Pg) * PD
          dwf = dw
          dwc = 0.0
          dwr = 0.0
          dn = 0.0
        ENDIF
      ELSEIF (INDEX(DYN,'INTEG') .NE. 0) THEN
        LAI = LAI + dLAI
        w  = w  + dw
        wc = wc + dwc
        wr = wr + dwr
        wf = wf + dwf
        LAI = MAX(LAI,0.0)
        W   = MAX(W, 0.0)
        WC  = MAX(WC,0.0)
        WR  = MAX(WR,0.0)
        WF  = MAX(WF,0.0)
        N   = N   + dN
        IF (int .GT. INTOT) THEN
          endsim = 1
          WRITE(1,14) doy
   14     FORMAT(/,'  The crop matured on day ',I3,'.')
          RETURN
        ENDIF
      ELSEIF (INDEX(DYN,'OUTPUT') .NE. 0) THEN
        WRITE(1,20) DOY,n,int,w,wc,wr,wf,lai
   20   FORMAT(I5,7F8.2)
        IF (COUNT .EQ. 23) THEN
          COUNT = 0
          WRITE(*,30)
   30     FORMAT(2/)
          WRITE(*,12)
        ENDIF
        COUNT = COUNT + 1
        WRITE(*,20) DOY,n,int,w,wc,wr,wf,lai
      ELSEIF (INDEX(DYN,'CLOSE') .NE. 0) THEN
       CLOSE(1)   
      ENDIF
      RETURN
      END SUBROUTINE PLANT
      SUBROUTINE LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,dN,p1, sla, dLAI)
      IMPLICIT NONE
      SAVE
      REAL PD,EMP1,EMP2,N,nb,dLAI, SWFAC,a, dN, p1,sla
      REAL SWFAC1, SWFAC2, PT, di, FL
      SWFAC = MIN(SWFAC1, SWFAC2)
      IF (FL .EQ. 1.0) THEN 
          a = exp(EMP2 * (N-nb))  
          dLAI = SWFAC * PD * EMP1 * PT * (a/(1+a)) * dN 
      ELSEIF (FL .EQ. 2.0) THEN
          dLAI = - PD * di * p1 * sla 
      ENDIF
      RETURN
      END SUBROUTINE LAIS
      SUBROUTINE PGS(SWFAC1, SWFAC2,PAR, PD, PT, Lai, Pg)
      IMPLICIT NONE 
      SAVE
      REAL PAR, Lai, Pg, PT, Y1
      REAL SWFAC1, SWFAC2, SWFAC,ROWSPC,PD
      SWFAC = MIN(SWFAC1, SWFAC2)
      ROWSPC = 60.0  
      Y1    = 1.5 - 0.768 * ((ROWSPC * 0.01)**2 * PD)**0.1 
      Pg = PT * SWFAC * 2.1 * PAR/PD * (1.0 - EXP(-Y1 * LAI))
      RETURN
      END SUBROUTINE PGS
      SUBROUTINE PTS(TMAX,TMIN,PT)
      IMPLICIT NONE 
      SAVE  
      REAL PT,TMAX,TMIN
      PT = 1.0 - 0.0025*((0.25*TMIN+0.75*TMAX)-26.0)**2
      RETURN
      END SUBROUTINE PTS
      SUBROUTINE SW(DOY, LAI, RAIN, SRAD, TMAX, TMIN,SWFAC1, SWFAC2,DYN)
      IMPLICIT NONE 
      SAVE
      INTEGER   DATE, DOY
      REAL      SRAD,TMAX,TMIN,RAIN,SWC,INF,IRR,ROF,ESa,EPa,DRNp
      REAL      DRN,DP,WPp,FCp,STp,WP,FC,ST,ESp,EPp,ETp,LAI
      CHARACTER*10 DYN
      REAL CN, SWFAC1, SWFAC2, POTINF 
      REAL SWC_INIT, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
      REAL TINF, SWC_ADJ
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
        OPEN(3,FILE='SOIL.INP',STATUS='UNKNOWN')
        OPEN(10,FILE='sw.out',  STATUS='REPLACE')
        OPEN(11,FILE='IRRIG.INP',STATUS='UNKNOWN')
        READ(3,10) WPp,FCp,STp,DP,DRNp,CN,SWC
   10   FORMAT(5X,F5.2,5X,F5.2,5X,F5.2,5X,F7.2,5X,F5.2,5X,F5.2,5X,F5.2)
        CLOSE(3)
        WRITE(10,15)
   15   FORMAT('Results of soil water balance simulation:',/,105X,'Soil',/,73X,'Pot.  Actual  Actual    Soil   Water',10X,'Excess',/,'  Day   Solar     Max     Min',42X,'Evapo-    Soil   Plant   Water Content Drought   Water',/,'   of    Rad.    Temp    Temp    Rain   Irrig  Runoff','   Infil   Drain   Trans   Evap.  Trans. content   (mm3/','  Stress  Stress',/,' Year (MJ/m2)    (oC)    (oC)    (mm)','    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)    (mm)','    (mm)    mm3)  Factor  Factor')
        WP  = DP * WPp * 10.0  
        FC  = DP * FCp * 10.0   
        ST  = DP * STp * 10.0  
        SWC_INIT = SWC
        CALL RUNOFF(POTINF, CN, ROF, 'INITIAL   ')
        CALL STRESS(SWC, DP, FC, ST, WP, SWFAC1, SWFAC2,'INITIAL   ')
        TRAIN = 0.0
        TIRR  = 0.0
        TESA  = 0.0
        TEPA  = 0.0
        TROF  = 0.0
        TDRN  = 0.0
        TINF  = 0.0
        SWC_ADJ = 0.0
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
        READ(11,25) DATE, IRR
   25   FORMAT(I5,2X,F4.1)
        TIRR = TIRR + IRR
        POTINF = RAIN + IRR
        TRAIN  = TRAIN + RAIN
        CALL DRAINE(SWC, FC, DRNp, DRN)
        IF (POTINF .GT. 0.0) THEN
          CALL RUNOFF(POTINF, CN, ROF, 'RATE      ')
          INF = POTINF - ROF
        ELSE
          ROF = 0.0
          INF = 0.0
        ENDIF
        CALL ETpS(SRAD,TMAX,TMIN,LAI,ETp)
        ESp = ETp * EXP(-0.7 * LAI)
        EPp = ETp * (1 - EXP(-0.7 * LAI))
        CALL ESaS(ESp,SWC,FC,WP,ESa)
        EPa = EPp * MIN(SWFAC1, SWFAC2)
      ELSEIF (INDEX(DYN,'INTEG') .NE. 0) THEN
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
      ELSEIF (INDEX(DYN,'OUTPUT    ') .NE. 0) THEN
        WRITE(10,40) DOY, SRAD, TMAX, TMIN, RAIN, IRR, ROF, INF, DRN,ETP, ESa, EPa, SWC, SWC/DP, SWFAC1, SWFAC2
   40   FORMAT(I5,3F8.1,9F8.2,3F8.3)
      ELSEIF (INDEX(DYN,'CLOSE') .NE. 0) THEN
      CALL WBAL(SWC_INIT, SWC, TDRN, TEPA,TESA, TIRR, TRAIN, TROF, SWC_ADJ, TINF)
      CLOSE(10) 
      CLOSE(11)
      ENDIF
      RETURN
      END SUBROUTINE SW
      SUBROUTINE DRAINE(SWC,FC,DRNp,DRN)
      IMPLICIT NONE
      SAVE
      REAL SWC, FC, DRN, DRNp
      IF (SWC .GT. FC) THEN
         DRN = (SWC - FC) * DRNp
      ELSE
         DRN = 0
      ENDIF
      RETURN
      END SUBROUTINE DRAINE
      SUBROUTINE ESaS(ESp,SWC,FC,WP,ESa)     
      IMPLICIT NONE
      SAVE
      REAL a, SWC, WP, FC, ESa, ESp
      IF (SWC .LT. WP) THEN
        a = 0
      ELSEIF (SWC .GT. FC) THEN
        a = 1
      ELSE 
        a = (SWC - WP)/(FC - WP)
      ENDIF
      ESa = ESp * a
      RETURN
      END SUBROUTINE ESAS
      SUBROUTINE ETpS(SRAD,TMAX,TMIN,LAI,ETp)
      IMPLICIT NONE  
      SAVE
      REAL    ALB,EEQ,f,Tmed,LAI
      REAL TMAX, TMIN, SRAD, ETP  
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
      RETURN
      END SUBROUTINE ETPS
      SUBROUTINE RUNOFF(POTINF, CN, ROF, DYN) 
      IMPLICIT NONE  
      SAVE
      CHARACTER*10 DYN
      REAL S, CN
      REAL POTINF, ROF 
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
        S = 254 * (100/CN - 1)
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
        IF (POTINF .GT. 0.2 * S)  THEN
          ROF = ((POTINF - 0.2 * S)**2)/(POTINF + 0.8 * S)
        ELSE
          ROF = 0
        ENDIF
      ENDIF
      RETURN
      END SUBROUTINE RUNOFF
      SUBROUTINE STRESS(SWC, DP, FC, ST, WP, SWFAC1, SWFAC2, DYN)
      IMPLICIT NONE
      SAVE
      CHARACTER*10 DYN
      REAL FC, ST, SWC, WP, SWFAC2, SWFAC1
      REAL DP, DWT, WTABLE, THE
      REAL, PARAMETER :: STRESS_DEPTH = 250
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
        THE = WP + 0.75 * (FC - WP)
      ENDIF
      IF (INDEX(DYN,'INTEG') .NE. 0 .OR. INDEX(DYN,'INITIAL') .NE. 0)THEN
        IF (SWC .LT. WP) THEN
          SWFAC1 = 0.0
        ELSEIF (SWC .GT. THE) THEN
          SWFAC1 = 1.0
        ELSE
          SWFAC1 = (SWC - WP) / (THE - WP)
          SWFAC1 = MAX(MIN(SWFAC1, 1.0), 0.0)
        ENDIF
        IF (SWC .LE. FC) THEN
          WTABLE = 0.0
          DWT = DP * 10.
          SWFAC2 = 1.0
        ELSE
          WTABLE = (SWC - FC) / (ST - FC) * DP * 10.
          DWT = DP * 10. - WTABLE     
          IF (DWT .GE. STRESS_DEPTH) THEN
            SWFAC2 = 1.0
          ELSE 
            SWFAC2 = DWT / STRESS_DEPTH
          ENDIF
          SWFAC2 = MAX(MIN(SWFAC2, 1.0), 0.0)
        ENDIF
      ENDIF
      RETURN
      END SUBROUTINE STRESS
      SUBROUTINE WBAL(SWC_INIT, SWC, TDRN, TEPA,TESA, TIRR, TRAIN, TROF, SWC_ADJ, TINF)
      IMPLICIT NONE
      SAVE
      INTEGER, PARAMETER :: LSWC = 21
      REAL SWC, SWC_INIT
      REAL TDRN, TEPA, TESA, TIRR, TRAIN, TROF
      REAL WATBAL, SWC_ADJ, TINF
      REAL CHECK
      OPEN (LSWC, FILE = 'WBAL.OUT', STATUS = 'REPLACE')
      WATBAL = (SWC_INIT - SWC) + (TRAIN + TIRR) -(TESA + TEPA + TROF + TDRN)
      WRITE(*,100)   SWC_INIT, SWC, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
      WRITE(LSWC,100)SWC_INIT, SWC, TRAIN, TIRR, TESA, TEPA, TROF, TDRN
  100 FORMAT(//, 'SEASONAL SOIL WATER BALANCE', //,'Initial soil water content (mm):',F10.3,/,'Final soil water content (mm):  ',F10.3,/,'Total rainfall depth (mm):      ',F10.3,/,'Total irrigation depth (mm):    ',F10.3,/,'Total soil evaporation (mm):    ',F10.3,/,'Total plant transpiration (mm): ',F10.3,/,'Total surface runoff (mm):      ',F10.3,/,'Total vertical drainage (mm):   ',F10.3,/)
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
  300   FORMAT(/,'Error: TRAIN + TIRR - TROF = ',F10.4,/,'Total infiltration =         ',F10.4,/,'Difference =                 ',F10.4)
      ENDIF
      CLOSE (LSWC)
      RETURN
      END SUBROUTINE WBAL
      SUBROUTINE WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,DYN)
      IMPLICIT NONE 
      SAVE
      REAL SRAD,TMAX,TMIN,RAIN,PAR
      INTEGER DATE
      CHARACTER*10 DYN
      IF (INDEX(DYN,'INITIAL') .NE. 0) THEN
        OPEN (4,FILE='WEATHER.INP',STATUS='UNKNOWN')  
      ELSEIF (INDEX(DYN,'RATE') .NE. 0) THEN
        READ(4,20) DATE,SRAD,TMAX,TMIN,RAIN,PAR
   20   FORMAT(I5,2X,F4.1,2X,F4.1,2X,F4.1,F6.1,14X,F4.1)
        PAR = 0.5 *  SRAD
      ELSEIF (INDEX(DYN,'CLOSE') .NE. 0) THEN
        CLOSE(4) 
      ENDIF
      RETURN
      END SUBROUTINE WEATHR
