* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign arrays

**********************************************************************
* Expected Output:  MonthTxt:
*                   JAN
*                   FEB
*                   MAR
*                   APR
*                   MAY
*                   JUN
*                   JUL
*                   AUG
*                   SEP
*                   OCT
*                   NOV
*                   DEC
**********************************************************************

	  REAL FUNCTION PRINTMONTHS(MONTHARR, S)

      INTEGER I
      REAL S
      CHARACTER (LEN=*) MONTHARR(12)
      WRITE (*,12) 'MONTHARR: '
      DO I = 1, 12
          WRITE (*,12) MONTHARR(I)
      END DO
      PRINTMONTHS = 5.0
 12   FORMAT(A)
	  RETURN
	  END FUNCTION PRINTMONTHS


      PROGRAM MAIN
      IMPLICIT NONE
      CHARACTER*3 MonthTxt(12)
      DATA MonthTxt /'JAN','FEB','MAR','APR','MAY','JUN',
     &               'JUL','AUG','SEP','OCT','NOV','DEC'/
      REAL R, X, PRINTMONTHS
      R = 5.0
      X = PRINTMONTHS(MonthTxt, R)
      END PROGRAM MAIN