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

      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER*3 MonthTxt(12)
!      CHARACTER*3, DIMENSION(12) :: MonthTxt
      DATA MonthTxt /'JAN','FEB','MAR','APR','MAY','JUN',
     &               'JUL','AUG','SEP','OCT','NOV','DEC'/

      INTEGER I

      WRITE (*,12) 'MonthTxt: '
      DO I = 1, 12
          WRITE (*,12) MonthTxt(I)
      END DO

 12   FORMAT(A)

      END PROGRAM MAIN