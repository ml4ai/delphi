      PROGRAM MAIN

      CHARACTER*10 TEST
      CHARACTER*4 NEW
      INTEGER A,B,C,D,E

       TEST = 'ABCDEFGHIJ'
       NEW = TEST(5:8)


C      WRITE(TEST, '(5(I1,X))')
C     &  A,B,C,D,E

      WRITE (*,10) NEW

 10   FORMAT(A)
      STOP
      END PROGRAM MAIN
