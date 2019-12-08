* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign simple real and integer variables

**********************************************************************
* Expected Output:  A: 1  B: 2  C: 3
*                   X: 2.20  Y: 2.20  Z: 2.20
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      INTEGER :: A,B,C
      REAL :: X,Y,Z

      DATA A /1/, B,C /2,3/
      DATA X,Y,Z /3*2.2/

      WRITE(*,10) 'A: ', A, 'B: ', B, 'C: ', C
      WRITE(*,20) 'X: ', X, 'Y: ', Y, 'Z: ', Z

 10   FORMAT(A, I1, 2X, A, I1, 2X, A, I1)
 20   FORMAT(A, F4.2, 2X, A, F4.2, 2X, A, F4.2)

      END PROGRAM MAIN
