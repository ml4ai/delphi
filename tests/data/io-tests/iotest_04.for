C Read two floats from a file, then write them out to another file.
C The expected output written to file 'outfile2' is the line:
C 56.7812.34

      PROGRAM MAIN
      
      OPEN (10, FILE="infile2")
      OPEN (20, FILE="outfile2", STATUS="REPLACE")
      READ (10,10) X,Y
      WRITE (20,10) Y,X
 10   FORMAT (F5.2,X,F5.2)
      STOP
      END PROGRAM MAIN
