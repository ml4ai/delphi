C Read a single integer from a file, then write it out to another file.
C The expected output written to file 'outfile1' is the line:
C 12345
      SUBROUTINE ERROR (ERRKEY)

      INTEGER ERRKEY
      WRITE(*,20) 'ERRKEY: ', ERRKEY

 20   FORMAT(A, I1)
      END SUBROUTINE ERROR

      PROGRAM MAIN

      INTEGER LUNIO, ERR
      CHARACTER*7 FILEIO

      LUNIO = 10
      FILEIO = "infile1c"

      OPEN (LUNIO, FILE=FILEIO, STATUS = 'OLD',IOSTAT=ERR)
      IF (ERR .NE. 0) CALL ERROR(5)

      OPEN (20, FILE="outfile1", STATUS="REPLACE")

      READ (LUNIO,10) I
      WRITE (20,10) I

 10   FORMAT (I5)
      CLOSE (LUNIO)
      STOP
      END PROGRAM MAIN
