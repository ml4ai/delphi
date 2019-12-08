* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign characters and strings

**********************************************************************
* Expected Output:  Hello!
*                   Paded     A
*                   Over
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER X*6, Y*10, Z*4, M*1

      DATA X /'Hello!'/, Y /'Paded'/
      DATA Z /'Overwrite'/, M /'A'/

      WRITE(*,10) X
      WRITE(*,20) Y, M
      WRITE(*,10) Z

 10   FORMAT(A)
 20   FORMAT(A, A)
      END PROGRAM MAIN

