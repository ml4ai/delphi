      SUBROUTINE PETPT(MSALB, SRAD, TMAX, TMIN, XHLAI,EO)                                             !Output
      IMPLICIT NONE
      REAL MSALB, SRAD, TMAX, TMIN, XHLAI
      REAL EO
      REAL ALBEDO, EEQ, SLANG, TD
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
      EO = MAX(EO,0.0001)
      RETURN
      END SUBROUTINE PETPT
