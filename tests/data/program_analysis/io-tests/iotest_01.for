C Read a single integer from a file, then write it out to another file.  
C The expected output written to file 'outfile1' is the line:
C 12345

      PROGRAM MAIN

      OPEN (10, FILE="infile1")
      OPEN (20, FILE="outfile1", STATUS="REPLACE")
      READ (10,10) I
      WRITE (20,10) I
 10   FORMAT (I5)
      STOP
      END PROGRAM MAIN
