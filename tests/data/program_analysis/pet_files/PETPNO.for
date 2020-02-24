C=======================================================================
C  PETPNO, Subroutine, N.B. Pickering
C  Calculates FAO-24 Penman potential evapotranspiration (without
C  correction)--grass reference.
!-----------------------------------------------------------------------
C  REVISION HISTORY
C  02/19/1992 NBP Written
C  11/04/1993 NBP Renamed routine PET to PETPEN.  Changed XLAI to XHLAI
C               Added XHLAI .LE. 0.0 statement.
C  05/13/1994 NBP Converted all vapor pressures to Pa.  Rearranged.
C  09/16/1994 NBP Added limits to prevent EO and ES (PE) < 0.
C  10/17/1997 CHP Updated for modular format.
C  09/01/1999 GH  Incorporated into CROPGRO
C  05/06/2002 WMB Fixed Stefan-Boltzmann constant
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!-----------------------------------------------------------------------
!  Called from:   PET
!  Calls:         None
C=======================================================================
      SUBROUTINE PETPNO(
     &    CLOUDS, MSALB, SRAD, TAVG, TDEW,                !Input
     &    TMAX, TMIN, WINDSP, XHLAI,                      !Input
     &    EO)                                             !Output
!-----------------------------------------------------------------------
      IMPLICIT NONE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL CLOUDS, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSYCON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VHCAIR, VPD, WFNFAO, DAIR, RT
      REAL SHAIR, PATM, SBZCON

      PARAMETER (SHAIR = 1005.0)
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZCON=4.093E-9)   !(MJ/m2/d)
      PARAMETER (SBZCON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUNCTION SUBROUTINES:
      REAL VPSLOP, VPSAT      !Found in file HMET.for

C-----------------------------------------------------------------------
C     Compute air properties.

C      ESAT = VPSAT(VPSLOP(VPSAT(TMIN))) + VPSLOP(VPSLOP(VPSAT(TMAX)))

      LHVAP = (2501.0-2.373*TAVG) * 1000.0                 ! J/kg
      PSYCON = SHAIR * PATM / (0.622*LHVAP)                ! Pa/K
      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0               ! Pa
      EAIR = VPSAT(TDEW)                                   ! Pa
      VPD = ESAT - EAIR                                    ! Pa
      S = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0                ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                             ! N.m/mol
      DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VHCAIR = DAIR * SHAIR    !not used                      ! J/m3

C     Convert windspeed to 2 m reference height.
!     Do this conversion in WEATHR and send out 2m windspeed
!     CHP 11/26/01
!      WIND2 = WINDSP * (2.0/WINDHT)**0.2

C     Calculate net radiation (MJ/m2/d).  Constants for RADB from
C     Jensen et al (1989) for semi-humid conditions.  The value 0.005
C     converts the value 0.158 from kPa to Pa.

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
      RADB = SBZCON * TK4 * (0.4 - 0.005 * SQRT(EAIR)) *
     &        (1.1 * (1. - CLOUDS) - 0.1)
      RNET= (1.0-ALBEDO)*SRAD - RADB

C     Compute ETP using the FAO wind function.  The multipliers for WNDFAO
C     are 1000 times smaller than in Jensen et al (1979) to convert VPD in
C     Pa to kPa. Equation for RNETMG converts from MJ/m2/d to mm/day.

!      WFNFAO = 0.0027 * (1.0+0.01*WIND2)
      WFNFAO = 0.0027 * (1.0+0.01*WINDSP)
      RNETMG = (RNET-G) / LHVAP * 1.0E6
      EO = (S*RNETMG + PSYCON*WFNFAO*VPD) / (S+PSYCON)
!###  EO = MAX(EO,0.0)   !gives error in DECRAT_C
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPNO


C=======================================================================
C  VPSAT, Real Function, N.B. Pickering, 4/1/90
C  Calculates saturated vapor pressure of air (Tetens, 1930).
!-----------------------------------------------------------------------
!  Called by: CANPET, HMET, VPSLOP, PETPEN
!  Calls:     None
!-----------------------------------------------------------------------
C  Input : T (C)
C  Output: VPSAT (Pa)
C=======================================================================
      REAL FUNCTION VPSAT(T)

      IMPLICIT NONE
      REAL T

      VPSAT = 610.78 * EXP(17.269*T/(T+237.30))

      RETURN
      END FUNCTION VPSAT
C=======================================================================
! VPSAT Variables
!-----------------------------------------------------------------------
! T     Air temperature (oC)
! VPSAT Saturated vapor pressure of air (Pa)
C=======================================================================



C=======================================================================
C  VPSLOP, Real Function, N.B. Pickering, 4/1/90
C  Calculates slope of saturated vapor pressure versus temperature curve
C  using Classius-Clapeyron equation (see Brutsaert, 1982 p. 41)
!-----------------------------------------------------------------------
!  Called by: ETSOLV, PETPEN, TRATIO
!  Calls:     VPSAT
!-----------------------------------------------------------------------
C  Input : T (C)
C  Output: VPSLOP
C=======================================================================
      REAL FUNCTION VPSLOP(T)

      IMPLICIT NONE

      REAL T,VPSAT

C     dEsat/dTempKel = MolWeightH2O * LatHeatH2O * Esat / (Rgas * TempKel^2)
      VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)

      RETURN
      END FUNCTION VPSLOP
C=======================================================================
! VPSLOP variables
!-----------------------------------------------------------------------
! T      Air temperature (oC)
! VPSAT  Saturated vapor pressure of air (Pa)
! VPSLOP Slope of saturated vapor pressure versus temperature curve
C=======================================================================
