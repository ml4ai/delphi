C     File: str06.f
C     This program tests getting and setting of substrings

      program main
      character(len = 10) str1, str2*5

      str1 = "abcdefgh"
      str2 = str1(3:8)
      str1(2:4) = str2

      write (*, 10) str1, str2

 10   format(A, "; ", A)

      stop
      end program main
      
