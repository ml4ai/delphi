C Read two integers from a file, then write them out to another file.
C The expected output written to file 'outfile1' is the line:
C 6789012345

      PROGRAM MAIN
      
      OPEN (10, FILE="infile1")
      OPEN (20, FILE="outfile1", STATUS="REPLACE")
      READ (10,10) I,J
      WRITE (20,10) J,I
 10   FORMAT (I5,I5)
      STOP
      END PROGRAM MAIN
