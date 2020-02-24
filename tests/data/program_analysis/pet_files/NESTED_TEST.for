      SUBROUTINE PETPNO(TMAX, TMIN)

      IMPLICIT NONE

      REAL TMAX, TMIN, ESAT, EAIR

      REAL VPSLOP, VPSAT

C      ESAT = VPSAT(VPSLOP(VPSAT(TMIN))) + VPSLOP(VPSLOP(VPSAT(TMAX)))
C      ESAT = VPSAT(VPSLOP(VPSAT(TMIN)))
C      ESAT = VPSAT(VPSLOP(TMIN))

C      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0
C      EAIR = VPSAT(TDEW)

C      EAIR = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0

      RETURN
      END SUBROUTINE PETPNO

C=======================================================================

      REAL FUNCTION VPSAT(T)

      IMPLICIT NONE
      REAL T

      VPSAT = 610.78
C      VPSAT = 610.78 * EXP(17.269*T/(T+237.30))

      RETURN
      END FUNCTION VPSAT

C=======================================================================

      REAL FUNCTION VPSLOP(T)

      IMPLICIT NONE

      REAL T,VPSAT

      VPSLOP = 18.0
C      VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)

      RETURN
      END FUNCTION VPSLOP

C=======================================================================