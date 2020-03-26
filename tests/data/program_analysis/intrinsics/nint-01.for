        program test_nint
          real :: k = 0.5
          real :: x = 1.45
          real :: y = 12.76
          real :: z = -0.5

          integer a, b, c, d
          a = nint(k)
          b = nint(x)
          c = nint(y)
          d = nint(z)

          print *, a, b, c, d
        end program test_nint
