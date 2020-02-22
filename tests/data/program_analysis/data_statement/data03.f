* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays

**********************************************************************
* Expected Output:  VEC:
*                   9.0
*                   9.0
*                   9.0
*                   0.1
*                   0.5
*
*                   PAIR1:
*                   4.0
*                   2.0
*
*                   PAIR2:
*                   4.0
*                   None
*
*                   PAIR3:
*                   None
*                   2.0
*
*                   MULTI:
*                   2.5  2.5  2.5  2.5
*                   2.5  2.5  2.5  2.5
*                   2.5  2.5  2.5  2.5
*                   2.5  2.5  2.5  2.5
*                   2.5  2.5  2.5  2.5
*
*                   XYZ:
*                   1  3  5
*                   2  4  6
*
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      REAL, DIMENSION(5) :: VEC
      REAL, DIMENSION(2) :: PAIR1
      REAL, DIMENSION(2) :: PAIR2
      REAL, DIMENSION(2) :: PAIR3
      REAL, DIMENSION(5,4) :: MULTI
      INTEGER, DIMENSION(2,3) :: XYZ

      INTEGER :: I, J

      DATA VEC /3*9.0, 0.1, 0.5/
      DATA PAIR1 /4.0, 2.0/, PAIR2(1) /4.0/
      DATA PAIR3(2) /2.0/
      DATA MULTI /20*2.5/
      DATA XYZ /1,2,3,4,5,6/

      WRITE (*,12) 'VEC: '
      DO I = 1, 5
          WRITE (*,10) VEC(I)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'PAIR1: '
      DO I = 1, 2
          WRITE(*,10) PAIR1(I)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'PAIR2: '
      DO I = 1, 2
          WRITE(*,10) PAIR2(I)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'PAIR3: '
      DO I = 1, 2
          WRITE(*,10) PAIR3(I)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'MULTI: '
      DO I = 1, 5
          WRITE (*,20) MULTI(I,1), MULTI(I,2), MULTI(I,3), MULTI(I,4)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'XYZ: '
      DO I = 1, 2
          WRITE (*,30) XYZ(I,1), XYZ(I,2), XYZ(I,3)
      END DO

 10   FORMAT(F3.1)
 11   FORMAT('')
 12   FORMAT(A)
 20   FORMAT(4(F3.1,2X))
 30   FORMAT(3(I1,2X))

      END PROGRAM MAIN
