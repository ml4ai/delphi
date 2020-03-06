C=======================================================================
C  PETPEN, Subroutine, N.B. Pickering
C  Calculates FAO-56 Penman-Monteith potential evapotranspiration, exactly
C  grass reference, with place for optional Kc, need this Kc in species.
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
C  01/15/2003 KJB discarded old Penman FAO-24 (it is much too stressful)
C     replaced with Penman FAO-56, exactly grass reference, with explicit
C     LAI 2.88, height 0.12 m, Rs = 100/(0.5*2.88)
C  01/15/2003 KJB:  THREE ISSUES TO ADDRESS:
C  1) USING FIXED ALBEDO, BECAUSE THAT IS HOW REFERENCE IS DEFINED
C  2)  PRESENTLY USING A LOCKED-IN VALUE OF 1.1 TO GIVE KC OF 1.1
C  I WOULD LIKE TO SEE OPTION OF SPECIES INPUT OF KC=1.1 TO 1.3
C  3) WINDHT WAS IN OLD, APPARENTLY 2.0, NO LONGER HERE.  ???
C  02/06/2003 KJB/CHP Added EORATIO as input from plant routines.
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!  09/19/2006 SSJ Fixed error in REFHT calc as noted below.
!  08/25/2011 CHP Use measured vapor pressure (VAPR), if available
!-----------------------------------------------------------------------
!  Called from:   PET
!  Calls:         None
C=======================================================================
      SUBROUTINE PETPEN(
     &    CLOUDS, EORATIO, MSALB, SRAD, TAVG, TDEW,       !Input
     &    TMAX, TMIN, VAPR, WINDSP, WINDHT, XHLAI,        !Input
     &    EO)                                             !Output
!-----------------------------------------------------------------------
      IMPLICIT NONE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL CLOUDS, EORATIO, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI, WINDSP_M
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSYCON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VHCAIR, VPD, DAIR, RT, ET0, KC, WINDHT, VAPR
      REAL SHAIR, PATM, SBZCON
      REAL k, d, REFHT, Zom, Zoh, ra, rl, rs    !added for PenMon
!      REAL alt_RADB, Tprev
!      INTEGER THREEDAYCOUNT
!      REAL    THREEDAYAVG(3)

!     PARAMETER (WINDHT = 2.0)
C     PARAMETER (SHAIR = 1005.0)
      PARAMETER (SHAIR = 0.001005)  !changed for PenMon to MJ/kg/K
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZCON=4.093E-9)  !(MJ/m2/d)
      PARAMETER (SBZCON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUNCTION SUBROUTINES:
      REAL VPSLOP, VPSAT      !Found in file HMET.for

C-----------------------------------------------------------------------
C     Compute air properties.
      LHVAP = (2501.0-2.373*TAVG) * 1000.0                ! J/kg
C     PSYCON = SHAIR * PATM / (0.622*LHVAP)               ! Pa/K
      PSYCON = SHAIR * PATM / (0.622*LHVAP) * 1000000     ! Pa/K

!     Previous code:
      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0              ! Pa
      EAIR = VPSAT(TDEW)                                  ! Pa

!     If actual vapor pressure is available, use it.
      IF (VAPR > 1.E-6) THEN
        EAIR = VAPR * 1000.
      ENDIF

      VPD = MAX(0.0, ESAT - EAIR)                         ! Pa
      S = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0               ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                         ! N.m/mol
      DAIR = 0.028966*(PATM-0.387*EAIR)/RT                ! kg/m3
C BAD DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VHCAIR = DAIR * SHAIR    !not used                  ! J/m3

C     Convert windspeed to 2 m reference height.
!     Do this conversion in WEATHR and send out 2m windspeed
!     CHP 11/26/01
!      WIND2 = WINDSP * (2.0/WINDHT)**0.2

C       Calculate aerodynamic resistance (ra).
C       ra (d/m) = {ln[zm-d/zom]*ln[zh-d/zoh]}/(k^2*uz)
C       zm = ht.wind measurement (m), zh = ht.humidity measurement (m),
C       zom,zoh=rooughness length of momentum, heat and vapor x-fer (m)
C       k=von Karman's constant 0.41, uz=WINDSP @ z m/d,
C       d = zero plane displacement height (m)

        REFHT = 0.12                 !arbitrary for testing PenMon
        WINDSP_M = WINDSP*(1000.)     !Converts km/d to m/d
        k = 0.41                     !von Karman's constant

!       was 2/3, which (for integers) results in zero!!
!       SSJ 9/19/2006 added the decimals
        !d = (2/3)*REFHT
        d = (2./3.)*REFHT

        Zom = 0.123*REFHT
        Zoh = 0.1*Zom
        ra = (LOG((WINDHT-d)/Zom)*LOG((WINDHT-d)/Zoh))/((k**2)*WINDSP_M)

C       Calculate surface resistance (rs).
C       rs = rl/LAIactive       rs (s m^-1),
C       rl = bulk stomatal resistance of the well-illuminated leaf (s m^-1)

        rl = 100           !value assummed from FAO grass reference
        rs = rl/(0.5*2.88) !0.5*XHLAI assumes half of LA is contributing
C                          !  to heat/vapor transfer
        rs = rs/86400      !converts (s m^-1 to d/m)

C     Calculate net radiation (MJ/m2/d).  By FAO method 1990. EAIR is divided
C       by 1000 to convert Pa to KPa.

c     MJ, 2007-04-11
c     --------------
c     There appears to be no support for soil heat flux (G), apart
c     from the variable already existing; it is just always set to
c     0, for some reason.
c     Here is the (improved) CANEGRO method for calculating G
c     (Allen, R.G. et al 1989,
c     'Operational Estimates of Reference Evapotranspiration',
c     Agronomy Journal Vol. 81, No. 4),
c     http://www.kimberly.uidaho.edu/water/papers/evapotranspiration/
c                   Allen_Operational_Estimates_Reference_ET_1989.pdf
c     :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
c         3-day sum of average temperature:
c          IF (THREEDAYCOUNT .LT. 1) THEN
c             Initialise
c              THREEDAYCOUNT = 1
c              THREEDAYAVG   = Tavg
c          ELSE IF (THREEDAYCOUNT .GE. 3) THEN
c              THREEDAYCOUNT = 1
c          ELSE
c              THREEDAYCOUNT = THREEDAYCOUNT + 1
c          ENDIF
c          THREEDAYAVG(THREEDAYCOUNT) = Tavg
c          Tprev = SUM(THREEDAYAVG)/3.
c          G = (Tavg-Tprev) * 0.38
c     --------------
c     MJ, 2007-04-12:
c     :::::::::::::::
c     FAO suggests that G be set to 0.  Oh well.
c     ------------------------------------------

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
C  KJB NOTE THAT REFERENCE IS ALWAYS ALBEDO FIXED TO 0.23,  OLD PEN VARIED
C  THE ALBEDO WITH LAI.  WHAT DO WE WANT?  IS THIS PART OF THE REASON THAT
C  KC IS NEEDED WITH THE REFERENCE FORMULATION?
C       ALBEDO = 0.23-(0.23-SALB)*EXP(-0.75*XHLAI)
        ALBEDO = 0.23
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
C
C     BELOW WAS THE OLD PENMAN, DIFFERENT CLOUDS METHOD, EAIR CHG IS GOOD
C     RADB = SBZCON * TK4 * (0.4 - 0.005 * SQRT(EAIR)) *
C    &        (1.1 * (1. - CLOUDS) - 0.1)
C

      RADB = SBZCON * TK4 * (0.34 - 0.14 * SQRT(EAIR/1000)) *
     &        (1.35 * (1. - CLOUDS) - 0.35)

      RNET= (1.0-ALBEDO)*SRAD - RADB

C     Compute EO using Penman-Montieth

      RNETMG = (RNET-G)
C     !MJ/m2/d
        ET0 = ((S*RNETMG + (DAIR*SHAIR*VPD)/ra)/(S+PSYCON*(1+rs/ra)))
C     !Converts MJ/m2/d to mm/d
        ET0 = ET0/ (LHVAP / 1000000.)
        IF (XHLAI .LE. 6.0) THEN
        XHLAI = XHLAI
        ELSE
        XHLAI = 6.0
        ENDIF
C   KJB LATER, NEED TO PUT VARIABLE IN PLACE OF 1.1
!      KC=1.0+(1.1-1.0)*XHLAI/6.0
      KC=1.0+(EORATIO-1.0)*XHLAI/6.0
      EO=ET0*KC
C     EO=ET0
        EO = MAX(EO,0.0)
!###  EO = MAX(EO,0.0)   !gives error in DECRAT_C
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPEN

!-----------------------------------------------------------------------
!     PETPEN VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! CLOUDS  Relative cloudiness factor (0-1)
! DAIR
! EAIR    Vapor pressure at dewpoint (Pa)
! EO      Potential evapotranspiration rate (mm/d)
! ESAT    Vapor pressure of air (Pa)
! G       Soil heat flux density term (MJ/m2/d)
! LHVAP   Latent head of water vaporization (J/kg)
! PATM     = 101300.0
! PSYCON  Psychrometric constant (Pa/K)
! RADB    Net outgoing thermal radiation (MJ/m2/d)
! RNET    Net radiation (MJ/m2/d)
! RNETMG  Radiant energy portion of Penman equation (mm/d)
! RT
! S       Rate of change of saturated vapor pressure of air with
!           temperature (Pa/K)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! SBZCON   Stefan Boltzmann constant = 4.903E-9 (MJ/m2/d)
! SHAIR    = 1005.0
! SRAD    Solar radiation (MJ/m2-d)
! TAVG    Average daily temperature (�C)
! TDEW    Dewpoint temperature (�C)
! TK4     Temperature to 4th power ((oK)**4)
! TMAX    Maximum daily temperature (�C)
! TMIN    Minimum daily temperature (�C)
! Tprev   3-day sum of average temperature:
! VHCAIR
! VPD     Vapor pressure deficit (Pa)
! VPSAT   Saturated vapor pressure of air (Pa)
! VPSLOP  Calculates slope of saturated vapor pressure versus
!           temperature curve (Pa/K)
! WFNFAO  FAO 24 hour wind function
! WIND2   Windspeed at 2m reference height. (km/d)
! WINDSP  Wind speed at 2m (km/d)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPEN
!-----------------------------------------------------------------------


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
