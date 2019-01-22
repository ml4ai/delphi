        PROGRAM MAIN
        IMPLICIT NONE 
        REAL X, Y

        OPEN (2,FILE='INFILE')
        READ(2,10) X, Y
   10   FORMAT(F5.2,X,F5.2)
        CLOSE(2)

        OPEN (1,FILE='OUTFILE',STATUS='REPLACE')
   11   FORMAT('The values of X and Y are: ')
        WRITE(1,11)
        WRITE(1,12) X, Y
   12   FORMAT(F6.3, 3X, F4.2)
        CLOSE(1)

        STOP
        END PROGRAM MAIN
