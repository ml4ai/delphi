      PROGRAM MAIN
C     GAUSSIAN ELIMINATION
C     From: http://users.metu.edu.tr/azulfu/courses/es361/programs/fortran/GAUEL.FOR

      DIMENSION A(20,21)
      PRINT *
      PRINT *, 'GAUSS ELIMINATION'
      DATA  N/4/
      DATA  (A(1,J), J=1,5) /-40.0,28.5,0.,0.,1.81859/
      DATA  (A(2,J), J=1,5) /21.5,-40.0,28.5,0.,-1.5136/
      DATA  (A(3,J), J=1,5) /0.,21.5,-40.0,28.5,-0.55883/
      DATA (A(4,J), J=1,5) /0.,0.,21.5,-40.0,1.69372/
      PRINT *
      PRINT *, 'AUGMENTED MATRIX'
      PRINT *
      DO I=1,N
      PRINT 61, (A(I,J),J=1,5)
61        FORMAT(1X,5f8.4)
      END DO
      PRINT *
      CALL GAUSS(N,A)
65    PRINT *
68    PRINT *, 'SOLUTION'
      PRINT *, '...........................................'
69    PRINT *, '        I       X(I)'
70    PRINT *, '............................................'
        DO I=1,N
72         FORMAT(5X,I5, 1PE16.6)
           PRINT 72, I, A(I,N+1)
         END DO
75    PRINT *,'..............................................'
80    PRINT *
      STOP
      END PROGRAM MAIN
C*************************************
      SUBROUTINE GAUSS(N,A)
      INTEGER PV
      DIMENSION A(20,21)
      EPS=1.0
10    IF (1.0+EPS.GT.1.0) THEN
          EPS=EPS/2.0
          GOTO 10
      END IF
      EPS=EPS*2
      PRINT *,'      MACHINE EPSILON=',EPS
      EPS2=EPS*2
1005  DET=1.
      DO 1010 I=1,N-1
         PV=I
         DO J=I+1,N
            IF (ABS(A(PV,I)) .LT. ABS(A(J,I))) PV=J
         END DO
         IF (PV.EQ.I) GOTO 1050
         DO JC=1,N+1
          TM=A(I,JC)
          A(I,JC)=A(PV,JC)
          A(PV,JC)=TM
        END DO
1045    DET=-1*DET
1050    IF (A(I,I).EQ.0.0) GOTO 1200
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
      PRINT *
      PRINT *,' DETERMINANT= ',DET
      PRINT*
      IF (A(N,N).EQ.0.0) GOTO 1200
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
1200  PRINT *,'MATRIX IS SINGULAR'
      END
