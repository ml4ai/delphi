C File: test_module_07.f
C Purpose: Illustrates the use of private variables in Fortran modules.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_07.f     # << this will create a file "mymod7.mod"
C    gfortran test_module_07.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C        6912

      MODULE MYMOD7
          IMPLICIT NONE
          INTEGER, PRIVATE :: X = 1234
      contains
          subroutine myadd(y, sum)
          integer y, sum

          sum = x+y
          end subroutine myadd
      END MODULE MYMOD7

      PROGRAM PGM
      USE mymod7
      integer x, v

      x = 5678    ! << This assignment does NOT overwrite mymod7's private variable x

      call myadd(x,v)

      write (*,*) v

      stop
      end program PGM
