      PROGRAM MAIN

      IMPLICIT NONE

      REAL, DIMENSION(5) :: VEC
      INTEGER :: X,Y

      DATA X,VEC,Y /3*9.0, 0.1, 2*0.5,6/

      INTEGER :: I

      WRITE(*,30) 'X: ', X

      WRITE (*,11)
      WRITE (*,12) 'VEC: '
      DO I = 1, 5
          WRITE (*,10) VEC(I)
      END DO

      WRITE (*,11)
      WRITE(*,30) 'Y: ', Y

 10   FORMAT(F3.1)
 11   FORMAT('')
 12   FORMAT(A)
 20   FORMAT(4(F3.1,2X))
 30   FORMAT(A, I1)

      END PROGRAM MAIN