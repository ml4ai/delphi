      PROGRAM MAIN

      CHARACTER*10 TEST
      INTEGER A,B,C,D,E

      A = 1
      B = 2
      C = 3
      D = 4
      E = 5

      TEST(1:1) = A
      TEST(2:2) = ' '
      TEST(3:3) = 'B'
      TEST(4:4) = ' '
      TEST(5:5) = 'C'
      TEST(6:6) = ' '
      TEST(7:7) = 'D'
      TEST(8:8) = ' '
      TEST(9:9) = 'E'
      TEST(10:10) = ' '

C      WRITE(TEST, '(5(I1,X))')
C     &  A,B,C,D,E

      WRITE (*,10) TEST

 10   FORMAT(A)

      STOP
      END PROGRAM MAIN
