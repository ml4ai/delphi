* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign simple real and integer variables

**********************************************************************
* Expected Output:  A: 1  B: 2  C: 3  D: 60
*                   M: 5.50  N: 5.50  O: 5.50
*                   X: 3.20  Y: 3.20  Z: 2.20  W: 2.20
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: A,B,C,D
      REAL :: M,N,O,X,Y,Z,W

      DATA A /1/, B,C /2,3/
      DATA M,N,O /3*5.5/
      DATA X,Y,Z,W /2*3.2,2*2.2/

      D = 60

      WRITE(*,10) 'A: ', A, 'B: ', B, 'C: ', C, 'D: ', D
      WRITE(*,20) 'M: ', M, 'N: ', N, 'O: ', O
      WRITE(*,30) 'X: ', X, 'Y: ', Y, 'Z: ', Z, 'W: ', W

 10   FORMAT(A, I1, 2X, A, I1, 2X, A, I1, 2X, A, I2)
 20   FORMAT(A, F4.2, 2X, A, F4.2, 2X, A, F4.2)
 30   FORMAT(A, F4.2, 2X, A, F4.2, 2X, A, F4.2, 2X, A, F4.2)

      END PROGRAM MAIN
