      PROGRAM MAIN
C     GAUSSIAN ELIMINATION
C     From: http://users.metu.edu.tr/azulfu/courses/es361/programs/fortran/GAUEL.FOR

      IMPLICIT NONE
      REAL, DIMENSION (20,21) :: A
      INTEGER :: I, J, N

 10   FORMAT(/,' GAUSS ELIMINATION')
      WRITE (*,10)      

C INITIALIZATION
C      DATA  N/4/
C      DATA  (A(1,J), J=1,5) /-40.0,28.5,0.,0.,1.81859/
C      DATA  (A(2,J), J=1,5) /21.5,-40.0,28.5,0.,-1.5136/
C      DATA  (A(3,J), J=1,5) /0.,21.5,-40.0,28.5,-0.55883/
C      DATA (A(4,J), J=1,5) /0.,0.,21.5,-40.0,1.69372/

      N = 4
      
 11   FORMAT(5(X,F8.2))
      OPEN(10, FILE="INFILE-GAUSSIAN")

      DO I = 1, N
         READ (10,11) A(I,1), A(I,2), A(I,3), A(I,4), A(I,5)
      END DO
      CLOSE(10)
C END INITIALIZATION

 12   FORMAT(/,' AUGMENTED MATRIX',/)
      WRITE(*,12)

61    FORMAT(5(1X,f8.4))
      DO I=1,N
      WRITE(*, 61) A(I,1), A(I,2), A(I,3), A(I,4), A(I,5)
      END DO

 13   FORMAT('')
 14   FORMAT(' SOLUTION')
 15   FORMAT(' ...........................................')
 16   FORMAT('         I       X(I)')
      
      WRITE(*,13)
      
      CALL GAUSS(N,A)

      WRITE(*,13)
      WRITE(*,14)
      WRITE(*,15)
      WRITE(*,16)      
      WRITE(*,15)

72    FORMAT(5X,I5, F12.6)

      DO I=1,N
          WRITE (*,72) I, A(I,N+1)
      END DO

      WRITE(*,15)
      WRITE(*,13)
      
      STOP
      END PROGRAM MAIN
C*************************************
      SUBROUTINE GAUSS(N,A)

      REAL, DIMENSION(20,21) :: A
      INTEGER PV, I, J, K, N, R, JC, JR, KC, NV
      REAL :: EPS, EPS2, DET, TM, TEMP, VA

      EPS=1.0
      DO WHILE (1.0+EPS.GT.1.0) 
          EPS=EPS/2.0
      END DO
      EPS=EPS*2
      
 11   FORMAT ('      MACHINE EPSILON=',E16.8)
      WRITE(*,11) EPS
      
      EPS2=EPS*2

1005  DET=1.0
      DO 1010 I=1,N-1
         PV=I
         DO J=I+1,N
            IF (ABS(A(PV,I)) .LT. ABS(A(J,I))) PV=J
         END DO
         IF (PV.NE.I) THEN
             DO JC=1,N+1
              TM=A(I,JC)
              A(I,JC)=A(PV,JC)
              A(PV,JC)=TM
            END DO
1045        DET=-1*DET
        END IF

1050    IF (A(I,I).EQ.0.0) THEN
            STOP 'MATRIX IS SINGULAR'
        END IF

        DO JR=I+1,N
           IF (A(JR,I).NE.0.0) THEN
              R=A(JR,I)/A(I,I)
              DO KC=I+1,N+1
              TEMP=A(JR,KC)
              A(JR,KC)=A(JR,KC)-R*A(I,KC)
              IF (ABS(A(JR,KC)).LT.EPS2*TEMP) A(JR,KC)=0.0
            END DO
          END IF
1060     END DO
1010  CONTINUE
      DO I=1,N
         DET=DET*A(I,I)
      END DO

 12   FORMAT(/,'  DETERMINANT= ',F16.5,/)
      WRITE(*,12) DET

      IF (A(N,N).EQ.0.0) THEN
          STOP 'MATRIX IS SINGULAR'
      END IF

      A(N,N+1)=A(N,N+1)/A(N,N)
      DO NV=N-1,1,-1
         VA=A(NV,N+1)
         DO K=NV+1,N
            VA=VA-A(NV,K)*A(K,N+1)
         END DO
         A(NV,N+1)=VA/A(NV,NV)
      END DO
      RETURN
1100  CONTINUE
      RETURN

      END
