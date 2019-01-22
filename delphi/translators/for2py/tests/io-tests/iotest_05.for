C Read two ints and two floats from a file using one FORMAT, 
C then write them out to another file using a different FORMAT.
C The expected output written to file 'outfile3' are the 
C following four lines:
C
C F =  23.5; I =  456
C
C F =  67.9; I =  123

      PROGRAM MAIN
      
      OPEN (10, FILE="infile3")
      OPEN (20, FILE="outfile3", STATUS="REPLACE")
 30   FORMAT(/,'F = ', F5.1, '; I = ', I4)
      READ (10,10) I,X,J,Y
      WRITE (20,30) X,J
      WRITE (20,30) Y,I
 10   FORMAT (2(I3,X,F5.2,X))
      STOP
      END PROGRAM MAIN
