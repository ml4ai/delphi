C     Strings and string concatenation

      program str01
      character(len = 10) str1, str2, str3*15

      str1 = "abcdef"
      str2 = "ijklmnop"

      str3 = str1 // str2       ! concatenation

      write (*, 10) "str1", len(str1), str1
      write (*, 10) "str2", len(str2), str2
      write (*, 10) "str1//str2", len(str1//str2), str1//str2
      write (*, 10) "str3", len(str3), str3
 10   format(A, ': len = ', I2, '; value = "', A, '"')

      stop
      end program str01
      
