C     File: str01.f
C     This program tests declaration and assignment of strings.

      program main
      character(len = 10) str1

      str1 = "abcdef"

      write (*, 10) "str1", len(str1), str1

 10   format(A, ': len = ', I2, '; value = "', A, '"')

      stop
      end program main
