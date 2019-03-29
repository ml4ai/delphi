C Read a single float from a file, then write it out to another file.
C The expected output written to file 'outfile2' is the line:
C 2.345

      PROGRAM MAIN
      
      OPEN (10, FILE="infile2")
      OPEN (20, FILE="outfile2", STATUS="REPLACE")
      READ (10,10) X
      WRITE (20,10) X
 10   FORMAT (F5.3)
      STOP
      END PROGRAM MAIN
