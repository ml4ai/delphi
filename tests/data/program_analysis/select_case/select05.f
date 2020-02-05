      PROGRAM MAIN

      CHARACTER(LEN=1) :: c = 'd'

      SELECT CASE (c)
        CASE ('a' : 'j')
          WRITE(*,10)  'One of the first ten letters'
        CASE ('l' : 'p', 'u' : 'y')
          WRITE(*,10)  'One of l, m, n, o, p, u, v, w, x, y'
        CASE ('z', 'q' : 't')
          WRITE(*,10)  'One of z, q, r, s, t'
        CASE DEFAULT
          WRITE(*,10)  'Other characters, which may not be letters'
      END SELECT

 10   FORMAT(A)
      END PROGRAM MAIN
