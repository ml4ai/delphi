* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays

**********************************************************************
* Expected Output:  X: 9.0
*
*                   VEC:
*                   9.0
*                   9.0
*                   0.1
*                   0.5
*                   0.5
*
*                   Y: 6.0
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      REAL, DIMENSION(5) :: VEC
      REAL :: X,Y

      DATA X,VEC,Y /3*9.0, 0.1, 2*0.5,6.0/
*      DATA VEC,X,Y /7*2/

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
 30   FORMAT(A, F3.1)

      END PROGRAM MAIN