C     File: str07.f
C     This program tests ADJUSTL and ADJUSTR

      program main
      character(len = 10) str1, str2*5

      str1 = "   abc  def"
      str2 = str1(3:7)

      write (*, 10) str1
      write (*, 10) str2
      write (*, 10) adjustl(str2)
      write (*, 10) adjustr(str2)

 10   format(">>>", A, "<<<")

      stop
      end program main
      
