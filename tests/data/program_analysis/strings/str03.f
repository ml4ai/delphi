C     File: str03.f
C     This program tests declaration and assignment of strings.

      program main
      character(len = 10) str1, str2*5, str3*15

      str1 = "abcdefgh"
      str2 = str1
      str3 = str1

      write (*, 10) "str1", len(str1), str1
      write (*, 10) "str2", len(str2), str2
      write (*, 10) "str3", len(str3), str3

 10   format(A, ': len = ', I2, '; value = "', A, '"')

      stop
      end program main
      
