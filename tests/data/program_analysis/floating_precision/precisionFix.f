        SUBROUTINE LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,
     &         dN,p1, sla, dLAI)

!-----------------------------------------------------------------------
        IMPLICIT NONE
        REAL PD,EMP1,EMP2,N,nb,dLAI, SWFAC,a, dN, p1,sla
        REAL SWFAC1, SWFAC2, PT, di, FL
!-----------------------------------------------------------------------

        SWFAC = MIN(SWFAC1, SWFAC2)
        IF (FL .EQ. 1.0) THEN
            a = exp(EMP2 * exp(N-nb))
            DO WHILE (1.0+a.GT.1.0)
                a=a/2.0
            END DO
            a=a*2
            dLAI = SWFAC * PD * EMP1 * PT * (a/(1+a)) * dN
        ELSEIF (FL .EQ. 2.0) THEN

            dLAI = - PD * di * p1 * sla

        ENDIF
!-----------------------------------------------------------------------
        RETURN
        END SUBROUTINE LAIS


        PROGRAM MAIN
        IMPLICIT NONE

        REAL nb,N,PT,di,p1,sla
        REAL PD,EMP1,EMP2
        REAL dn,dLAI, FL

        REAL SWFAC1, SWFAC2

        FL = 1.0
        nb = 2.5
        di = 1.0
        PD = 2.0
        EMP1 = 2.0
        EMP2 = 2.0
        N = 2.5
        SWFAC1 = 5.5
        SWFAC2 = 4.5
        PT = 1.0
        dN = 2.5
        p1 = 1.0
        sla = 1.0


        CALL LAIS(FL,di,PD,EMP1,EMP2,N,nb,SWFAC1,SWFAC2,PT,
     &             dN,p1, sla, dLAI)

        PRINT *, dLAI

        END PROGRAM MAIN
