C     Source "http://www.chem.ox.ac.uk/fortran/subprograms.html"

	  SUBROUTINE CALC(X,A,B,C)
	  REAL X
	  CHARACTER*(*) A, B, C
	  CHARACTER*15 SUM
	  SUM = A // B // C
	  WRITE (*, 10) "THE SUM OF THE STRINGS: ", SUM
 10   format(A, A)
	  RETURN
	  END SUBROUTINE CALC

      PROGRAM SUBDEM
      CHARACTER*5 STR3
      REAL X
      X = 5
      STR3 = "LINE!"
      CALL CALC(X, "ADD", "THIS", STR3)
      END PROGRAM SUBDEM