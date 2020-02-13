C=======================================================================
C  WTDEPT, Subroutine
C  Determines water table depth
!  Allows perched water table (uses highest free water surface in profile)
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  01/06/1997 GH  Written
!  10/20/1997 CHP Modified for modular format.
!  03/17/2001 CHP Allows water table between top and bottom of layer
!                 (no longer a step function).  Uses highest free water
!                 surface.
!  05/23/2007 CHP Start at bottom of profile and determine 1st unsaturated
!                   layer.  Add factor to decrease step function appearance.
C-----------------------------------------------------------------------
C  Called by: WATBAL
C  Calls    : None
C=======================================================================
      SUBROUTINE WTDEPT(
     &    NLAYR, DLAYR, DS, DUL, SAT, SW,                 !Input
     &    WTDEP)                                          !Output

!     ------------------------------------------------------------------
      USE ModuleDefs     !Definitions of constructed variable types,
                         ! which contain control information, soil
                         ! parameters, hourly weather data.

      IMPLICIT NONE

      INTEGER L, NLAYR
      REAL  WTDEP             !Depth to Water table (cm)
      REAL SATFRAC(NL), FACTOR
      REAL DLAYR(NL), DS(NL), DUL(NL), SAT(NL), SW(NL)
      REAL, PARAMETER :: TOL = 0.95

!-----------------------------------------------------------------------
      DO L = NLAYR, 1, -1
        SATFRAC(L) = (SW(L) - DUL(L)) / (SAT(L) - DUL(L))
        SATFRAC(L) = MIN(MAX(0.0, SATFRAC(L)), 1.0)
        IF (SATFRAC(L) > TOL) THEN
!         Layer is saturated, continue up to next layer
          CYCLE
        ELSEIF (L == NLAYR) THEN
!         Bottom layer is unsaturated
          WTDEP = DS(NLAYR) - DLAYR(NLAYR) * SATFRAC(NLAYR)
          EXIT
        ELSE
!         Layer is unsaturated.  Interpolate water table depth.
!         FACTOR prevents a step function when transitioning between
!           layers.
          FACTOR = MIN(MAX(0.0, (SATFRAC(L+1) - TOL) / (1.0 - TOL)),1.0)
          WTDEP = DS(L) - DLAYR(L) * SATFRAC(L) * FACTOR
          EXIT
        ENDIF
      ENDDO

!      IF (L > 0 .AND. L <= NLAYR) THEN
!        WRITE(6500,'(I4,9F8.3,3F8.2)') L, SW(L), SW(L+1), SAT(L),
!     &          SAT(L+1), DUL(L), DUL(L+1), SATFRAC(L), SATFRAC(L+1),
!     &          FACTOR, DLAYR(L), DS(L), WTDEP
!      ELSE
!        WRITE(6500,'(I4,9F8.3,3F8.2)') L, SW(NLAYR), 0.0, SAT(NLAYR),
!     &          0.0, DUL(NLAYR), 0.0, 0.0, 0.0,
!     &          0.0, DLAYR(NLAYR), DS(NLAYR), WTDEP
!      ENDIF

      RETURN
      END SUBROUTINE WTDEPT

!-----------------------------------------------------------------------
!     WTDEPT VARIABLE DEFINITIONS:
!-----------------------------------------------------------------------
! DLAYR(L) Soil thickness in layer L (cm)
! DS(L)    Cumulative depth in soil layer L (cm)
! NL       Maximum number of soil layers = 20
! NLAYR    Actual number of soil layers
! SAT(L)   Volumetric soil water content in layer L at saturation
!            (cm3 [water] / cm3 [soil])
! SW(L)    Volumetric soil water content in layer L (cm3[water]/cm3[soil])
! WTDEP    Water table depth  (cm)
! SATFRAC   Fraction of layer L which is saturated
!-----------------------------------------------------------------------
!     END SUBROUTINE WTDEPT
C=======================================================================
