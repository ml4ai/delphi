C     Strings and string concatenation.  This program looks at what happens
C     when the string assigned has length different from the declared length
C     of the string variable.
      

      program str03
      character(len = 5), parameter :: str1 = "abcdefghijklm"
      character(len = 5) :: str2 = "abcdefghijklm"
      character(len = 5) :: str3 = "ab"

      write (*, 10) "str1", len(str1), str1
      write (*, 10) "str2", len(str2), str2
      write (*, 10) "str3", len(str3), str3

 10   format(A, ': len = ', I2, '; value = "', A, '"')

      stop
      end program str03
      
