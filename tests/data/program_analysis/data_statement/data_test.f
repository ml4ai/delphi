* FORTRAN test file to implement the DATA statement
* This file uses the DATA statements for passing the tests

**********************************************************************
* Expected Output: A:
*                  1.0  3.0
*				   2.0  4.0
*
*			       B:
*			       1.1  2.1
*			       3.1  4.1
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: A,B,C
      CHARACTER Z*4, M*1, N*1
      REAL, DIMENSION(5) :: VEC
      REAL :: X,Y
      REAL, DIMENSION(2) :: PAIR1
      REAL, DIMENSION(2) :: PAIR2
      REAL, DIMENSION(2) :: PAIR3
      REAL, DIMENSION(5,4) :: MULTI
      REAL, DIMENSION(2,2) :: MULT_A
      REAL, DIMENSION(2,2) :: MULT_B

      INTEGER :: I

      DATA A /1/, B,C /2,3/
      DATA Z,M,N /'Overwrite',2*'A'/
      DATA X,VEC,Y /3*9.0, 0.1, 2*0.5,6.0/
      DATA PAIR1 /4.0, 2.0/, PAIR2(1) /4.0/
      DATA PAIR3(2) /2.0/
      DATA MULTI /20*2.5/
      DATA MULT_A /1.0, 2.0, 3.0, 4.0/
      DATA MULT_B(1,1) /1.1/, MULT_B(1,2) /2.1/
      DATA MULT_B(2,1), MULT_B(2,2) /3.1, 4.1/

      WRITE(*,10) 'A: ', A, 'B: ', B, 'C: ', C
      WRITE (*,11)

      WRITE(*,20) 'Z: ', Z
      WRITE(*,20) 'M: ', M
      WRITE(*,20) 'N: ', N
      WRITE (*,11)

      WRITE(*,30) 'X: ', X
      WRITE (*,11)

      WRITE (*,25) 'VEC: '
      DO I = 1, 5
          WRITE (*,35) VEC(I)
      END DO
      WRITE (*,11)

      WRITE(*,30) 'Y: ', Y
      WRITE (*,11)

      WRITE (*,25) 'PAIR1: '
      DO I = 1, 2
          WRITE(*,35) PAIR1(I)
      END DO
      WRITE (*,11)

      WRITE (*,25) 'PAIR2: '
      DO I = 1, 2
          WRITE(*,35) PAIR2(I)
      END DO
      WRITE (*,11)

      WRITE (*,25) 'PAIR3: '
      DO I = 1, 2
          WRITE(*,35) PAIR3(I)
      END DO
      WRITE (*,11)

      WRITE (*,25) 'MULTI: '
      DO I = 1, 5
          WRITE (*,40) MULTI(I,1), MULTI(I,2), MULTI(I,3), MULTI(I,4)
      END DO
      WRITE (*,11)

      WRITE (*,25) 'MULT_A: '
      DO I = 1, 2
          WRITE (*,45) MULT_A(I,1), MULT_A(I,2)
      END DO
      WRITE (*,11)

      WRITE (*,11)
      WRITE (*,25) 'MULT_B: '
      DO I = 1, 2
          WRITE (*,45) MULT_B(I,1), MULT_B(I,2)
      END DO

 10   FORMAT(A, I1, 2X, A, I1, 2X, A, I1)
 11   FORMAT('')
 20   FORMAT(A, A)
 25   FORMAT(A)
 30   FORMAT(A, F3.1)
 35   FORMAT(F3.1)
 40   FORMAT(4(F3.1,2X))
 45   FORMAT(2(F3.1,2X))
      END PROGRAM MAIN