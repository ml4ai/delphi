      SUBROUTINE PETASCE(
     &        CANHT, DOY, MSALB, MEEVP, SRAD, TDEW,       !Input
     &        TMAX, TMIN, WINDHT, WINDRUN, XHLAI,         !Input
     &        XLAT, XELEV,                                !Input
     &        EO)                                         !Output

      IMPLICIT NONE

      SAVE

!-----------------------------------------------------------------------
!     MARKER VARIABLES
!     Variables named i_g_n_o_r_e__m_e___* are used to mark points in
!     the program's AST for later association of intra-subprogram comments
      LOGICAL i_g_n_o_r_e__m_e___94, i_g_n_o_r_e__m_e___98,
     & i_g_n_o_r_e__m_e___101, i_g_n_o_r_e__m_e___104,
     & i_g_n_o_r_e__m_e___108, i_g_n_o_r_e__m_e___113,
     & i_g_n_o_r_e__m_e___116, i_g_n_o_r_e__m_e___119,
     & i_g_n_o_r_e__m_e___127, i_g_n_o_r_e__m_e___136,
     & i_g_n_o_r_e__m_e___139, i_g_n_o_r_e__m_e___150,
     & i_g_n_o_r_e__m_e___153, i_g_n_o_r_e__m_e___156,
     & i_g_n_o_r_e__m_e___166, i_g_n_o_r_e__m_e___197,
     & i_g_n_o_r_e__m_e___201, i_g_n_o_r_e__m_e___220,
     & i_g_n_o_r_e__m_e___227, i_g_n_o_r_e__m_e___235,
     & i_g_n_o_r_e__m_e___239, i_g_n_o_r_e__m_e___242


!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL CANHT, MSALB, SRAD, TDEW, TMAX, TMIN, WINDHT, WINDRUN
      REAL XHLAI, XLAT, XELEV
      INTEGER DOY
      CHARACTER*1 MEEVP
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL TAVG, PATM, PSYCON, UDELTA, EMAX, EMIN, ES, EA, FC, FEW, FW
      REAL ALBEDO, RNS, PIE, DR, LDELTA, WS, RA1, RA2, RA, RSO, RATIO
      REAL FCD, TK4, RNL, RN, G, WINDSP, WIND2m, Cn, Cd, KCMAX, RHMIN
      REAL WND, CHT
      REAL SKC, KCBMIN, KCBMAX, KCB, KE, KC

      i_g_n_o_r_e__m_e___94 = .True.
!-----------------------------------------------------------------------

!     ASCE Standardized Reference Evapotranspiration
!     Average temperature (ASCE Standard Eq. 2)
      TAVG = (TMAX + TMIN) / 2.0 !deg C

      i_g_n_o_r_e__m_e___98 = .True.
!     Atmospheric pressure (ASCE Standard Eq. 3)
      PATM = 101.3 * ((293.0 - 0.0065 * XELEV)/293.0) ** 5.26 !kPa

      i_g_n_o_r_e__m_e___101 = .True.
!     Psychrometric constant (ASCE Standard Eq. 4)
      PSYCON = 0.000665 * PATM !kPa/deg C

      i_g_n_o_r_e__m_e___104 = .True.
!     Slope of the saturation vapor pressure-temperature curve
!     (ASCE Standard Eq. 5)                                !kPa/degC
      UDELTA = 2503.0*EXP(17.27*TAVG/(TAVG+237.3))/(TAVG+237.3)**2.0

      i_g_n_o_r_e__m_e___108 = .True.
!     Saturation vapor pressure (ASCE Standard Eqs. 6 and 7)
      EMAX = 0.6108*EXP((17.27*TMAX)/(TMAX+237.3)) !kPa
      EMIN = 0.6108*EXP((17.27*TMIN)/(TMIN+237.3)) !kPa
      ES = (EMAX + EMIN) / 2.0                     !kPa

      i_g_n_o_r_e__m_e___113 = .True.
!     Actual vapor pressure (ASCE Standard Eq. 8)
      EA = 0.6108*EXP((17.27*TDEW)/(TDEW+237.3)) !kPa

      i_g_n_o_r_e__m_e___116 = .True.
!     RHmin (ASCE Standard Eq. 13, RHmin limits from FAO-56 Eq. 70)
      RHMIN = MAX(20.0, MIN(80.0, EA/EMAX*100.0))

      i_g_n_o_r_e__m_e___119 = .True.
!     Net shortwave radiation (ASCE Standard Eq. 16)
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23
      ENDIF
      RNS = (1.0-ALBEDO)*SRAD !MJ/m2/d

      i_g_n_o_r_e__m_e___127 = .True.
!     Extraterrestrial radiation (ASCE Standard Eqs. 21,23,24,27)
      PIE = 3.14159265359
      DR = 1.0+0.033*COS(2.0*PIE/365.0*DOY) !Eq. 23
      LDELTA = 0.409*SIN(2.0*PIE/365.0*DOY-1.39) !Eq. 24
      WS = ACOS(-1.0*TAN(XLAT*PIE/180.0)*TAN(LDELTA)) !Eq. 27
      RA1 = WS*SIN(XLAT*PIE/180.0)*SIN(LDELTA) !Eq. 21
      RA2 = COS(XLAT*PIE/180.0)*COS(LDELTA)*SIN(WS) !Eq. 21
      RA = 24.0/PIE*4.92*DR*(RA1+RA2) !MJ/m2/d Eq. 21

      i_g_n_o_r_e__m_e___136 = .True.
!     Clear sky solar radiation (ASCE Standard Eq. 19)
      RSO = (0.75+2E-5*XELEV)*RA !MJ/m2/d

      i_g_n_o_r_e__m_e___139 = .True.
!     Net longwave radiation (ASCE Standard Eqs. 17 and 18)
      RATIO = SRAD/RSO
      IF (RATIO .LT. 0.3) THEN
        RATIO = 0.3
      ELSEIF (RATIO .GT. 1.0) THEN
        RATIO = 1.0
      END IF
      FCD = 1.35*RATIO-0.35 !Eq 18
      TK4 = ((TMAX+273.16)**4.0+(TMIN+273.16)**4.0)/2.0 !Eq. 17
      RNL = 4.901E-9*FCD*(0.34-0.14*SQRT(EA))*TK4 !MJ/m2/d Eq. 17

      i_g_n_o_r_e__m_e___150 = .True.
!     Net radiation (ASCE Standard Eq. 15)
      RN = RNS - RNL !MJ/m2/d

      i_g_n_o_r_e__m_e___153 = .True.
!     Soil heat flux (ASCE Standard Eq. 30)
      G = 0.0 !MJ/m2/d

      i_g_n_o_r_e__m_e___156 = .True.
!     Wind speed (ASCE Standard Eq. 33)
      WINDSP = WINDRUN * 1000.0 / 24.0 / 60.0 / 60.0 !m/s
      WIND2m = WINDSP * (4.87/LOG(67.8*WINDHT-5.42))
      IF (MEEVP .EQ. 'A') THEN
        Cn = 1600.0
        Cd = 0.38
      ELSE IF (MEEVP .EQ. 'G') THEN
        Cn = 900.0
        Cd = 0.34
      END IF

      i_g_n_o_r_e__m_e___166 = .True.
!     Aerodynamic roughness and surface resistance daily timestep constants
!     (ASCE Standard Table 1)

!      SELECT CASE(MEEVP) !
!        CASE('A') !Alfalfa reference
!          Cn = 1600.0 !K mm s^3 Mg^-1 d^-1
!          Cd = 0.38 !s m^-1
!        CASE('G') !Grass reference
!          Cn = 900.0 !K mm s^3 Mg^-1 d^-1
!          Cd = 0.34 !s m^-1
!      END SELECT

!     FAO-56 dual crop coefficient approach
!     Basal crop coefficient (Kcb)
!     Also similar to FAO-56 Eq. 97
!     KCB is zero when LAI is zero

C The three calls to GET below are commented out and replaced by hard-coded
C assignments to match the Python code that Paul constructed

C      CALL GET('SPAM', 'SKC', SKC)
C      CALL GET('SPAM', 'KCBMIN', KCBMIN)
C      CALL GET('SPAM', 'KCBMAX', KCBMAX)

      SKC = 1.0
      KCBMIN = 1.0
      KCBMAX = 1.0

      IF (XHLAI .LE. 0.0) THEN
         KCB = 0.0
      ELSE
         i_g_n_o_r_e__m_e___197 = .True.
         !DeJonge et al. (2012) equation
         KCB = MAX(0.0,KCBMIN+(KCBMAX-KCBMIN)*(1.0-EXP(-1.0*SKC*XHLAI)))
      ENDIF

      i_g_n_o_r_e__m_e___201 = .True.
      !Maximum crop coefficient (Kcmax) (FAO-56 Eq. 72)
      WND = MAX(1.0,MIN(WIND2m,6.0))
      CHT = MAX(0.001,CANHT)

      IF (MEEVP .EQ. 'A') THEN
        KCMAX = MAX(1.0,KCB+0.05)
      ELSE IF (MEEVP .EQ. 'G') THEN
        KCMAX = MAX((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))
     &                      *(CHT/3.0)**(0.3)),KCB+0.05)
      END IF

!      SELECT CASE(MEEVP)
!        CASE('A') !Alfalfa reference
!            KCMAX = MAX(1.0,KCB+0.05)
!        CASE('G') !Grass reference
!            KCMAX = MAX((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))
!     &                      *(CHT/3.0)**(0.3)),KCB+0.05)
!      END SELECT

      i_g_n_o_r_e__m_e___220 = .True.
      !Effective canopy cover (fc) (FAO-56 Eq. 76)
      IF (KCB .LE. KCBMIN) THEN
         FC = 0.0
      ELSE
         FC = ((KCB-KCBMIN)/(KCMAX-KCBMIN))**(1.0+0.5*CANHT)
      ENDIF

      i_g_n_o_r_e__m_e___227 = .True.
      !Exposed and wetted soil fraction (FAO-56 Eq. 75)
      !Unresolved issue with FW (fraction wetted soil surface).
      !Some argue FW should not be used to adjust demand.
      !Rather wetting fraction issue should be addressed on supply side.
      !Difficult to do with a 1-D soil water model
      FW = 1.0
      FEW = MIN(1.0-FC,FW)

      i_g_n_o_r_e__m_e___235 = .True.
      !Potential evaporation coefficient (Ke) (Based on FAO-56 Eq. 71)
      !Kr = 1.0 since this is potential Ke. Model routines handle stress
      KE = MAX(0.0, MIN(1.0*(KCMAX-KCB), FEW*KCMAX))

      i_g_n_o_r_e__m_e___239 = .True.
      !Potential crop coefficient (Kc) (FAO-56 Eqs. 58 & 69)
      KC = KCB + KE

      i_g_n_o_r_e__m_e___242 = .True.
      !Potential evapotranspiration (FAO-56 Eq. 69)
      EO = EV_TRANSP(UDELTA,RN,G,PSYCON,Cn,TAVG,WIND2m,ES,EA,Cd,KC)

      RETURN
C The four calls to GET below are commented out to match the Python code
C that Paul constructed

C      CALL PUT('SPAM', 'REFET', REFET)
C      CALL PUT('SPAM', 'KCB', KCB)
C      CALL PUT('SPAM', 'KE', KE)
C      CALL PUT('SPAM', 'KC', KC)

!-----------------------------------------------------------------------
      CONTAINS
C     Module MOD_EVAPOTRANSP contains code to compute potential evapotranspiration
C     (FAO-56 Eq. 69)
C     REFET: Standardized reference evapotranspiration (ASCE Standard Eq. 1)
      REAL FUNCTION REFET(UD,RN,G,PCON,Cn,TAVG,W2m,ES,EA,Cd)
          REAL UD, RN, G, PCON, Cn, TAVG, W2m, ES, EA, Cd
          REAL RVAL

          RVAL = 0.408*UD*(RN-G)+PCON*(Cn/(TAVG+273.0))* W2m*(ES-EA)
          RVAL = RVAL/(UD+PCON*(1.0+Cd*W2m)) !mm/d
          REFET = MAX(0.0001, RVAL)
      END FUNCTION REFET

C     EV_TRANSP: potential evapotranspiration (FAO-56 Eq. 69)
      REAL FUNCTION EV_TRANSP(UD,RN,G,PCON,Cn,TAVG,W2m,ES,EA,Cd,KC)
          REAL UD, RN, G, PCON, Cn, TAVG, W2m, ES, EA, Cd, KC
          REAL EVTRANSP

          EVTRANSP = KC * REFET(UD,RN,G,PCON,Cn,TAVG, W2m,ES,EA,Cd)

          EV_TRANSP = MAX(EVTRANSP,0.0001)
      END FUNCTION EV_TRANSP

      END SUBROUTINE PETASCE
!-------------------------------------------------------------------

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER DOY
      REAL EO

      REAL, PARAMETER :: TMAX = 30.0
      REAL, PARAMETER :: TMIN = 20.0
      REAL, PARAMETER :: WINDRUN = 10.0
      REAL, PARAMETER :: XHLAI = 0.8
      REAL, PARAMETER :: XLAT = 40.0
      CHARACTER*1, PARAMETER :: MEEVP = 'A'

      REAL :: CANHT = 10.0
      REAL :: MSALB = 0.9
      REAL :: SRAD = 20.41
      REAL :: TDEW = 25.0
      REAL :: WINDHT = 10.0
      REAL :: XELEV = 100.0

      DO 123 DOY = 0, 500, 50
          CALL PETASCE(
     &        CANHT, DOY, MSALB, MEEVP, SRAD, TDEW,       !Input
     &        TMAX, TMIN, WINDHT, WINDRUN, XHLAI,         !Input
     &        XLAT, XELEV,                                !Input
     &        EO)                                         !Output

          WRITE (*,*) EO

          CANHT = CANHT + 1.0
          MSALB = MSALB * 1.2
          SRAD = SRAD + 0.02
          TDEW = TDEW + 2.0
          WINDHT = WINDHT * 1.2
          XELEV = XELEV + 500.0

 123   ENDDO

      STOP
      END PROGRAM MAIN

C=======================================================================
