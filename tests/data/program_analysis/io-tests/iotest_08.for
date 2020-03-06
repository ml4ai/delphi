C Read a single integer from a file, then write it out to another file.
C The expected output written to file 'outfile1' is the line:
C 12345

C      SUBROUTINE ERROR (ERRKEY)
C
C      INTEGER ERRKEY
C      WRITE(*,20) 'ERRKEY: ', ERRKEY
C
C 20   FORMAT(A, I1)
C      END SUBROUTINE ERROR

      PROGRAM MAIN

      INTEGER LUNIO, ERR
      CHARACTER*8 FILEIO

      LUNIO = 10
      FILEIO = "testfile"

      OPEN (LUNIO, FILE=FILEIO, STATUS = 'OLD',IOSTAT=ERR)
C      IF (ERR .NE. 0) CALL ERROR(5)

      OPEN (20, FILE="testout", STATUS="REPLACE")

      READ (LUNIO,10) I
      WRITE (20,10) I

 10   FORMAT (I5)
C      CLOSE (LUNIO)
      STOP
      END PROGRAM MAIN
