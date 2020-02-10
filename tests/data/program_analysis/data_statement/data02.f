* FORTRAN test file to implement the DATA statement
* This file uses the DATA statement to assign characters and strings

**********************************************************************
* Expected Output:  Hello!
*                   Paded     A
*                   Over
*                   CC
*                   A
**********************************************************************

      PROGRAM MAIN

      IMPLICIT NONE

      CHARACTER X*6, Y*10, Z*4, M*1, W*5, N*1

      DATA X /'Hello!'/, Y /'Paded'/
      DATA Z,M,N /'Overwrite',2*'A'/

      W = "CC"

      WRITE(*,10) X
      WRITE(*,20) Y, M
      WRITE(*,10) Z
      WRITE(*,10) W
      WRITE(*,10) N

 10   FORMAT(A)
 20   FORMAT(A, A)
      END PROGRAM MAIN

