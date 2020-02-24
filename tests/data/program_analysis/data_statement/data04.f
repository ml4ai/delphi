* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays using implied DO loops

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

      REAL, DIMENSION(2,2) :: A
      REAL, DIMENSION(2,2) :: B

      INTEGER :: I, J

      DATA A /1.0, 2.0, 3.0, 4.0/
      DATA B(1,1) /1.1/, B(1,2) /2.1/, B(2,1), B(2,2) /3.1, 4.1/

      WRITE (*,12) 'A: '
      DO I = 1, 2
          WRITE (*,20) A(I,1), A(I,2)
      END DO

      WRITE (*,11)
      WRITE (*,12) 'B: '
      DO I = 1, 2
          WRITE (*,20) B(I,1), B(I,2)
      END DO

 11   FORMAT('')
 12   FORMAT(A)
 20   FORMAT(2(F3.1,2X))
 30   FORMAT(4(F3.1,2X))
 40   FORMAT(10(F3.1,2X))

      END PROGRAM MAIN
