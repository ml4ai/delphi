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

C      REAL FUNCTION VPSAT(T)
C
C      IMPLICIT NONE
C      REAL T
C
C      VPSAT = 610.78 * EXP(17.269*T/(T+237.30))
C
C      RETURN
C      END FUNCTION VPSAT
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

C      REAL FUNCTION VPSLOP(T)
C
C      IMPLICIT NONE
C
C      REAL T,VPSAT
C
C      dEsat/dTempKel = MolWeightH2O * LatHeatH2O * Esat / (Rgas * TempKel^2)
C
C      VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)
C
C      RETURN
C      END FUNCTION VPSLOP
C=======================================================================
! VPSLOP variables
!-----------------------------------------------------------------------
! T      Air temperature (oC)
! VPSAT  Saturated vapor pressure of air (Pa)
! VPSLOP Slope of saturated vapor pressure versus temperature curve
C=======================================================================


C=======================================================================
C  PETDYN Subroutine, K. J. BOOTE, F. SAU, M. BOSTIC
C  Calculates PENMAN-MONTEITH potential evapotranspiration
C  using dynamic CANHT, LAI, along with wind effects on Ra, Rs
C  Steiner approach for Ra recommended, but FAO and Lhomme commented out
C  Sunlit LAI effect on Rs is recommended, but rl/(0.5*LAI) would work
C  Weighting used for Ra and Rs between soil and crop.  Need to changee
C  two constants (HTS and zos) if you change from sunlit LAI to other.
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  11/19/01 TO 1/15/02  Written By Boote, Sau, McNair
C  01/15/03 Moved from V3.5 trial to V4.0  by K. J. Boote
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)

!  Called from:   PET
!  Calls:         None
C=======================================================================
      SUBROUTINE PETDYN(
     &    CANHT, CLOUDS, MSALB, SRAD, TAVG, TDEW,         !Input
     &    TMAX, TMIN, WINDSP, XHLAI,                      !Input
     &    EO)                                             !Output
C  Calculates Penman-Monteith evapotranspiration
!-----------------------------------------------------------------------
      IMPLICIT NONE
      SAVE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL CLOUDS, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI, WINDSP_M
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSYCON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VHCAIR, VPD, DAIR, RT
      REAL SHAIR, PATM, SBZCON
      REAL k,DFAO, CANHT, ZOMF, ZOHF, ra, rl, rs, RAERO !add for PenDyn
      REAL ZCROP,DCROP,ZOMC,ZOVC,WIND2C,RASOIL,HTS,DLH,ZOLH
      REAL MAXHT, rb, AC, AS_MOD, zos, RTOT                  !add for PenDyn
C     PARAMETER (SHAIR = 1005.0)
      PARAMETER (SHAIR = 0.001005)  !changed for PenDyn to MJ/kg/K
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZCON=4.093E-9)  !(MJ/m2/d)
      PARAMETER (SBZCON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUNCTION SUBROUTINES:
      REAL VPSLOP_TMAX, VPSLOP_TMIN, VPSAT_TMAX, VPSAT_TMIN, VPSAT_TDEW

C-----------------------------------------------------------------------
C     Compute air properties.
      LHVAP = (2501.0-2.373*TAVG) * 1000.0                 ! J/kg
C     PSYCON = SHAIR * PATM / (0.622*LHVAP)                ! Pa/K
      PSYCON = SHAIR * PATM / (0.622*LHVAP) * 1000000     ! Pa/K

      VPSAT_TMAX = 610.78 * EXP(17.269*TMAX/(TMAX+237.30))
      VPSAT_TMIN = 610.78 * EXP(17.269*TMIN/(TMIN+237.30))
      VPSAT_TDEW = 610.78 * EXP(17.269*TDEW/(TDEW+237.30))

      ESAT = (VPSAT_TMAX+VPSAT_TMIN) / 2.0               ! Pa
      EAIR = VPSAT_TDEW                                   ! Pa

      VPD = ESAT - EAIR                                    ! Pa

      VPSLOP_TMAX = 18.0 * (2501.0-2.373*TMAX) * VPSAT_TMAX / (8.314*(TMAX+273.0)**2)
      VPSLOP_TMIN = 18.0 * (2501.0-2.373*TMIN) * VPSAT_TMIN / (8.314*(TMIN+273.0)**2)
      
      S = (VPSLOP_TMAX+VPSLOP_TMIN) / 2.0                ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                             ! N.m/mol
      DAIR = 0.028966*(PATM-0.387*EAIR)/RT                    ! kg/m3
C BAD DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VHCAIR = DAIR * SHAIR    !not used                      ! J/m3

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

      WINDSP_M = WINDSP*(1000.)/86400.          !Converts km/d to m/s
      k = 0.41                                  !von Karman's constant

      IF (CANHT .LE. 0.10) THEN
        ZCROP = 2.0 + 0.10
C       Next 3 are Steiner et al. coefficients, used for Steiner Ra
        DCROP = 0.75 * 0.10
        ZOMC = 0.25 * (0.10 - DCROP)
        ZOVC = 0.1 * ZOMC
        DFAO = 2. * 0.10 / 3.0
        ZOMF = 0.123*0.10
        ZOHF = 0.1*ZOMF

      ELSE
        ZCROP = 2.0 + CANHT
        DCROP = 0.75 * CANHT
        ZOMC = 0.25 * (CANHT - DCROP)
        ZOVC = 0.1 * ZOMC
        DFAO= 2.0 * CANHT / 3.0
        ZOMF = 0.123*CANHT
        ZOHF = 0.1*ZOMF
      ENDIF

C     LHOMME ET AL. AG & FOR. MET. 104:119.  2000.
C     Combined effects of LAI and crop height on Ra
C     cd = 0.2 (in eq below), where X=0.2*LAI
C     Zolh up to X<0.2 (or LAI=1), then X 0.2 to 1.5 (LAI=7.5)
C     Actually should have a cap at LAI 7.5 or less.

      DLH = 1.1*MAX(0.10,CANHT)*LOG(1.+(0.2*XHLAI)**0.25)

      IF (XHLAI .LT. 1.0) THEN
        ZOLH = 0.01+0.3*MAX(0.10,CANHT)*(0.2*XHLAI)**0.5
      ELSE
C        ELSEIF (XHLAI .LT. 7.5)
        ZOLH = 0.3*MAX(0.10,CANHT)*(1.0-DLH/MAX(0.10,CANHT))
      ENDIF

C  Concept of Ra, always for 2 m above crop height, from Steiner et al
C       Agron. J. 83:240.  1991.  Also, needs wind speed adjusted, up to
C       10 m, then back down to exactly 2 m above crop height.
C       Needs z defined at 2 m above crop, or z = 2.0 + CANHT
C       Grass assumed 0.10 m, its d is 0.075, its Zom is 0.00625

      WIND2C = WINDSP_M * LOG((10.-0.075)/0.00625) *
     &                      LOG((ZCROP-DCROP)/ZOMC) /
     &         (LOG((10.-DCROP)/ZOMC)*LOG((2.-0.075)/0.00625))

C       Steiner Ra
      ra = ( (LOG((ZCROP-DCROP)/ZOMC)*LOG((ZCROP-DCROP)/ZOVC))
     &       /((k**2)*WIND2C) )  /86400

C       Standard FAO Ra
C       ra = ( (LOG((ZCROP-DFAO)/ZOMF)*LOG((ZCROP-DFAO)/ZOHF))
C    &       /((k**2)*WIND2C) )  /86400

C       Lhomme Ra
C       ra = ( (LOG((ZCROP-DLH)/ZOLH)*LOG((ZCROP-DLH)/(0.1*ZOLH)))
C    &       /((k**2)*WIND2C) )  /86400

C      NOW, COMPUTING Ra for bare soil and Rs for bare soil
C      For bare soil Ra, with an effective height of 0.40 m
C      Uses standard FAO eq, windsp for 2 m height.  Not for soil Rs
C      HTS = 0.13, for SUNLIT LAI FORM.  HTS = 0.25 for 0.5*LAI FORM.

      HTS = 0.13
C      HTS = 0.25
      RASOIL = (  (LOG((2.0-2*HTS/3.)/(0.123*HTS))
     &    *LOG((2.0-2*HTS/3.)/(0.1*0.123*HTS)))/((k**2)*WINDSP_M))/86400

CWMB    BOUNDARY LAYER RESISTANCE (rb) FOR BARE SOIL FROM (JAGTAP AND JONES, 1989)
C       zos = roughness ht. of soil (m), MAXHT = maximum plant height (m)
C       MAXHT is a dummy argument to get right Rs from soil.  Not real
C       This is wet surface resistance, Rs-soil, to match up with Rs-crop
C       Do not want WIND2C, as this one acts most for bare soil, no crop
C
C       For Sunlit LAI for Rc, use zos = 0.01
C       For 0.5*LAI for Rc, need zos = 0.03
      zos = 0.01
C       zos = 0.03
      MAXHT = 1.0
      rb=((log(2.0/zos)*log((0.83*MAXHT)/zos))/
     &            ((k**2)*WINDSP_M))/86400
C

C       Using K = 0.5 everywhere possible
        AC = 1-exp(-0.50*XHLAI)
        AS_MOD = 1 - AC

      RAERO = AC*RA + AS_MOD*RASOIL
C     Calculate surface resistance (rs).
C     rs = rl/LAIactive       rs (s m^-1),
C     rl = bulk stomatal resistance of the well-illuminated leaf (s m^-1)

      rl = 100                !value assummed from FAO grass reference
      IF (XHLAI .GE. 0.1) THEN
C          rs = rl/(0.5*XHLAI)
        rs = rl/((1/0.5)*(1.0-EXP(-0.5*XHLAI)))       !SUNLIT LAI form
      ELSE
        rs = rl/(0.5*0.1)
      ENDIF

      rs = rs/86400           !converts (s m^-1 to d/m)

      RTOT = AC*rs + AS_MOD*rb

C     Calculate net radiation (MJ/m2/d).  By FAO method 1990. EAIR is divided
C       by 1000 to convert Pa to KPa.

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
C     I THINK THIS K VALUE SHOULD BE 0.5, NEAR THEORETICAL OF 0.5 KJB
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
      RADB = SBZCON * TK4 * (0.34 - 0.14 * SQRT(EAIR/1000)) *
     &        (1.35 * (1. - CLOUDS) - 0.35)
      RNET= (1.0-ALBEDO)*SRAD - RADB

C     Compute EO using Penman-Montieth

      RNETMG = (RNET-G)
C     !MJ/m2/d
      EO=((S*RNETMG + (DAIR*SHAIR*VPD)/RAERO)/(S+PSYCON*(1+RTOT/RAERO)))
C     !Converts MJ/m2/d to mm/d
        EO = EO/ (LHVAP / 1000000.)
!###  EO = MAX(EO,0.0)   !gives error in DECRAT_C
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETDYN

!     PETPEN VARIABLES:  Nearly same as PETPEN above
