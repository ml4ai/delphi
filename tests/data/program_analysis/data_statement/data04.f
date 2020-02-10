* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays using implied DO loops

**********************************************************************
* Expected Output: A:
*                  1.0  3.0
*				   2.0  4.0
*
*				   R:
*			   	   1.0  1.0
*			       1.0  1.0
*			       1.0  1.0
*
*			       S:
*			       1.0  0.0  0.0  0.0
*			       0.0  1.0  0.0  0.0
*			       0.0  0.0  1.0  0.0
*			       0.0  0.0  0.0  1.0
*
*			       B:
*			       1.1  2.1
*			       3.1  4.1
*
*			       C:
*			       2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0
*
*			       MATRIX:
*			       1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
*			       1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      REAL A(2,2), B(2,2), C(10)
      REAL R(3,2), S(4,4)
      INTEGER :: I, J
      REAL :: MATRIX(10,10)

      DATA A /1.0, 2.0, 3.0, 4.0/
      DATA B(1,1) /1.1/, B(1,2) /2.1/, B(2,1),B(2,2) /3.1, 4.1/
      DATA (C(I), I=1,10) /10*2.0/
      DATA ((R(I,J),J=1,2),I=1,3) /6*1.0/
      DATA (S(I,I), I=1,4) /4*1.0/

      DATA MATRIX(1,    1:10) / 10*1.0  / ! TOP ROW
      DATA MATRIX(10,  1:10) / 10*1.0  / ! BOT ROW
      DATA MATRIX(2:9, 1)     / 8*1.0   / ! LEFT COL
      DATA MATRIX(2:9, 10)   / 8*1.0   / ! RIGHT COL
      DATA MATRIX(2:9, 2:9)  / 64*0.0 / ! INTERIOR

      WRITE (*,12) 'A: '
      DO I = 1, 2
          WRITE (*,20) A(I,1), A(I,2)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'R: '
      DO I = 1, 3
          WRITE (*,20) R(I,1), R(I,2)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'S: '
      DO I = 1, 4
          WRITE (*,30) S(I,1), S(I,2), S(I,3), S(I,4)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'B: '
      DO I = 1, 2
          WRITE (*,20) B(I,1), B(I,2)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'C: '
      WRITE (*,40) C(1), C(2), C(3), C(4), C(5), C(6), C(7), C(8), C(9),
     &             C(10)

      WRITE (*,11)
      WRITE (*,12) 'MATRIX: '
      DO I = 1, 10
          WRITE (*,40) MATRIX(I,1), MATRIX(I,2), MATRIX(I,3),
     &          MATRIX(I,4), MATRIX(I,5), MATRIX(I,6), MATRIX(I,7),
     &          MATRIX(I,8), MATRIX(I,9), MATRIX(I,10)
      END DO

 11   FORMAT('')
 12   FORMAT(A)
 20   FORMAT(2(F3.1,2X))
 30   FORMAT(4(F3.1,2X))
 40   FORMAT(10(F3.1,2X))

      END PROGRAM MAIN
