C=======================================================================
C  PETPT, Subroutine, J.T. Ritchie
C  Calculates Priestly-Taylor potential evapotranspiration
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  ??/??/19?? JR  Written
C  11/04/1993 NBP Modified
C  10/17/1997 CHP Updated for modular format.
C  09/01/1999 GH  Incorporated into CROPGRO
!  07/24/2006 CHP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!-----------------------------------------------------------------------
!  Called by:   WATBAL
!  Calls:       None
C=======================================================================
      SUBROUTINE PETPT(
     &    MSALB, SRAD, TMAX, TMIN, XHLAI,                 !Input
     &    EO)                                             !Output

!-----------------------------------------------------------------------
      IMPLICIT NONE

!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL MSALB, SRAD, TMAX, TMIN, XHLAI
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL ALBEDO, EEQ, SLANG, TD

!-----------------------------------------------------------------------
!     Should use TAVG here -- we have it from WEATHER variable!
!     SSJ 9/18/2006
!     TD = TAVG
!     JWJ 2/15/2007 - Can't use TAVG unless coefficients in EEQ
!         equation are recalibrated.  Keep TD calc as it was
!         developed.
      TD = 0.60*TMAX+0.40*TMIN

      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      SLANG = SRAD*23.923
      EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)
      EO = EEQ*1.1

      IF (TMAX .GT. 35.0) THEN
        EO = EEQ*((TMAX-35.0)*0.05+1.1)
      ELSE IF (TMAX .LT. 5.0) THEN
        EO = EEQ*0.01*EXP(0.18*(TMAX+20.0))
      ENDIF

!###  EO = MAX(EO,0.0)   !gives error in DECRAT_C
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPT
!-----------------------------------------------------------------------
!     PETPT VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! EEQ     Equilibrium evaporation (mm/d)
! EO      Potential evapotranspiration rate (mm/d)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! SLANG   Solar radiation
! SRAD    Solar radiation (MJ/m2-d)
! TD      Approximation of average daily temperature (�C)
! TMAX    Maximum daily temperature (�C)
! TMIN    Minimum daily temperature (�C)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPT
C=======================================================================
