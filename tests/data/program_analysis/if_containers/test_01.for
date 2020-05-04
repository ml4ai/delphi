      PROGRAM TEST
      IMPLICIT NONE
      INTEGER :: X,Y,Z,W,COND1,COND2

      X = 0
      Y = 1
      Z = 2
      W = 3
      COND1 = 4
      COND2 = 9

      IF (COND1 .LE. 5) THEN
        X = X+1
        Y = X+2
        IF (COND1 .EQ. 6) THEN
          X = 4
          IF (Y .GT. 5) THEN
            Y = 2
           END IF
        END IF
        X = Y+2
        Z = 5
      ELSE IF (COND2 .GE. 9) THEN
        Y = Y+1
        Z = X+5
      ELSE IF (COND1 .GE. 6) THEN
        X = X+Y
        Z = X+W
      ELSE
        Z = Y+4
        W = W+2
        Z = X+Y
      END IF

      IF (X .GE. 4) THEN
        W = 2*Y
      ELSE
        Z = 5*X
      END IF

      WRITE(*,10)  X
      WRITE(*,20)  Y
      WRITE(*,30)  Z
      WRITE(*,40)  W

 10   FORMAT('X = ', I3)
 20   FORMAT('Y = ', I3)
 30   FORMAT('Z = ', I3)
 40   FORMAT('W = ', I3)

      END PROGRAM TEST