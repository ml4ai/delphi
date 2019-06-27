        PROGRAM MAIN

        IMPLICIT NONE
        REAL EPS
        EPS=1.0
        DO WHILE (1.0+EPS.GT.1.0)
            EPS=EPS/2.0
        END DO
        EPS=EPS*2

        PRINT *, EPS

        END PROGRAM MAIN