* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays

**********************************************************************
* Expected Output:  X: 2

*                   VEC:
*                   3.0
*                   4.0
*                   5.0
*
*                   Y: 6  Z: 6
*
*                   PAIR1:
*                   7.0
*                   8.0
*
*                   HELLO!
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      REAL, DIMENSION(3) :: VEC
      REAL, DIMENSION(2) :: PAIR1
      INTEGER :: X,Y,Z
      CHARACTER A*6

      INTEGER :: I

      DATA X,VEC,Y,Z,PAIR1,A /2,3,4,5,2*6,7,8,'HELLO!'/

      WRITE(*,50) 'X: ', X

      WRITE (*,11)
      WRITE (*,12) 'VEC: '
      DO I = 1, 3
          WRITE (*,10) VEC(I)
      END DO

      WRITE (*,11)
      WRITE(*,60) 'Y: ', Y, 'Z: ', Z

      WRITE (*,11)
      WRITE (*,12) 'PAIR1: '
      DO I = 1, 2
          WRITE(*,10) PAIR1(I)
      END DO

      WRITE (*,11)
      WRITE(*,12) A

 10   FORMAT(F3.1)
 11   FORMAT('')
 12   FORMAT(A)
 20   FORMAT(4(F3.1,2X))
 30   FORMAT(3(I1,2X))
 50   FORMAT(A, I1)
 60   FORMAT(A, I1, 2X, A, I1)

      END PROGRAM MAIN
