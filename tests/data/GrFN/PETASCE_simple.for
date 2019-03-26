      SUBROUTINE PETASCE(
     &        CANHT, DOY, MSALB, MEEVP, SRAD, TDEW,       !Input
     &        TMAX, TMIN, WINDHT, WINDRUN, XHLAI,         !Input
     &        XLAT, XELEV,                                !Input
     &        EO)                                         !Output

      IMPLICIT NONE

      SAVE

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
      REAL REFET, SKC, KCBMIN, KCBMAX, KCB, KE, KC
!-----------------------------------------------------------------------

!     ASCE Standardized Reference Evapotranspiration
!     Average temperature (ASCE Standard Eq. 2)
      TAVG = (TMAX + TMIN) / 2.0 !deg C

!     Atmospheric pressure (ASCE Standard Eq. 3)
      PATM = 101.3 * ((293.0 - 0.0065 * XELEV)/293.0) ** 5.26 !kPa

!     Psychrometric constant (ASCE Standard Eq. 4)
      PSYCON = 0.000665 * PATM !kPa/deg C

!     Slope of the saturation vapor pressure-temperature curve
!     (ASCE Standard Eq. 5)                                !kPa/degC
      UDELTA = 2503.0*EXP(17.27*TAVG/(TAVG+237.3))/(TAVG+237.3)**2.0

!     Saturation vapor pressure (ASCE Standard Eqs. 6 and 7)
      EMAX = 0.6108*EXP((17.27*TMAX)/(TMAX+237.3)) !kPa
      EMIN = 0.6108*EXP((17.27*TMIN)/(TMIN+237.3)) !kPa
      ES = (EMAX + EMIN) / 2.0                     !kPa

!     Actual vapor pressure (ASCE Standard Eq. 8)
      EA = 0.6108*EXP((17.27*TDEW)/(TDEW+237.3)) !kPa

!     RHmin (ASCE Standard Eq. 13, RHmin limits from FAO-56 Eq. 70)
      RHMIN = MAX(20.0, MIN(80.0, EA/EMAX*100.0))

!     Net shortwave radiation (ASCE Standard Eq. 16)
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23
      ENDIF
      RNS = (1.0-ALBEDO)*SRAD !MJ/m2/d

!     Extraterrestrial radiation (ASCE Standard Eqs. 21,23,24,27)
      PIE = 3.14159265359
      DR = 1.0+0.033*COS(2.0*PIE/365.0*DOY) !Eq. 23
      LDELTA = 0.409*SIN(2.0*PIE/365.0*DOY-1.39) !Eq. 24
      WS = ACOS(-1.0*TAN(XLAT*PIE/180.0)*TAN(LDELTA)) !Eq. 27
      RA1 = WS*SIN(XLAT*PIE/180.0)*SIN(LDELTA) !Eq. 21
      RA2 = COS(XLAT*PIE/180.0)*COS(LDELTA)*SIN(WS) !Eq. 21
      RA = 24.0/PIE*4.92*DR*(RA1+RA2) !MJ/m2/d Eq. 21

!     Clear sky solar radiation (ASCE Standard Eq. 19)
      RSO = (0.75+2E-5*XELEV)*RA !MJ/m2/d

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

!     Net radiation (ASCE Standard Eq. 15)
      RN = RNS - RNL !MJ/m2/d

!     Soil heat flux (ASCE Standard Eq. 30)
      G = 0.0 !MJ/m2/d

!     Wind speed (ASCE Standard Eq. 33)
      WINDSP = WINDRUN * 1000.0 / 24.0 / 60.0 / 60.0 !m/s
      WIND2m = WINDSP * (4.87/LOG(67.8*WINDHT-5.42))
C      HACK: adding initialized values due to possible undefined behavior
      Cn = 0.0
      Cd = 0.0
      IF (MEEVP .EQ. 'A') THEN
        Cn = 1600.0
        Cd = 0.38
      ELSE IF (MEEVP .EQ. 'G') THEN
        Cn = 900.0
        Cd = 0.34
      END IF

!     Standardized reference evapotranspiration (ASCE Standard Eq. 1)
      REFET =0.408*UDELTA*(RN-G)+PSYCON*(Cn/(TAVG+273.0))*WIND2m*(ES-EA)
      REFET = REFET/(UDELTA+PSYCON*(1.0+Cd*WIND2m)) !mm/d
      REFET = MAX(0.0001, REFET)

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

!     Values set from paper inspection
      SKC = 0.8
      KCBMIN = 0.3
      KCBMAX = 1.2

      IF (XHLAI .LE. 0.0) THEN
         KCB = 0.0
      ELSE
         !DeJonge et al. (2012) equation
         KCB = MAX(0.0,KCBMIN+(KCBMAX-KCBMIN)*(1.0-EXP(-1.0*SKC*XHLAI)))
      ENDIF

      !Maximum crop coefficient (Kcmax) (FAO-56 Eq. 72)
      WND = MAX(1.0,MIN(WIND2m,6.0))
      CHT = MAX(0.001,CANHT)

C     HACK: Another one like the above
      KCMAX = 0.5           ! HACK: Guarantees greater that KCBMIN
      IF (MEEVP .EQ. 'A') THEN
        KCMAX = MAX(1.0,KCB+0.05)
      ELSE IF (MEEVP .EQ. 'G') THEN
        KCMAX = MAX((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))
     &                      *(CHT/3.0)**(0.3)),KCB+0.05)
      END IF

      !Effective canopy cover (fc) (FAO-56 Eq. 76)
      IF (KCB .LE. KCBMIN) THEN
         FC = 0.0
      ELSE
         FC = ((KCB-KCBMIN)/(KCMAX-KCBMIN))**(1.0+0.5*CANHT)
      ENDIF

      !Exposed and wetted soil fraction (FAO-56 Eq. 75)
      !Unresolved issue with FW (fraction wetted soil surface).
      !Some argue FW should not be used to adjust demand.
      !Rather wetting fraction issue should be addressed on supply side.
      !Difficult to do with a 1-D soil water model
      FW = 1.0
      FEW = MIN(1.0-FC,FW)

      !Potential evaporation coefficient (Ke) (Based on FAO-56 Eq. 71)
      !Kr = 1.0 since this is potential Ke. Model routines handle stress
      KE = MAX(0.0, MIN(1.0*(KCMAX-KCB), FEW*KCMAX))

      !Potential crop coefficient (Kc) (FAO-56 Eqs. 58 & 69)
C      KC = KCB + KE
C     NOTE: the above is dead code

      !Potential evapotranspiration (FAO-56 Eq. 69)
      EO = (KCB + KE) * REFET

      EO = MAX(EO,0.0001)

      RETURN
      END SUBROUTINE PETASCE
!-------------------------------------------------------------------
