C     File: str05.f
C     This program tests the INDEX() function on strings.
C     The output generated is:
C
C       2    0   6   3   0

      program main
      character(len = 10) str1
      integer n1, n2, n3, n4, n5

      str1 = "abcdef"

      n1 = index(str1, "bc")
      n2 = index(str1, "xyz")
      n3 = index(str1, "f ", back=.true.)
      n4 = index(str1, "cde", back=.true.)
      n5 = index(str1, "xyz", back=.true.)

      write (*, 10) n1, n2, n3, n4, n5

 10   format(5(I3,X))

      stop
      end program main
      
