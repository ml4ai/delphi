C File: test_module_05.f
C Purpose: Illustrates a Fortran module that defines a subroutine which is
C    called from the main program.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_05.f     # << this will create a file "mymod6.mod"
C    gfortran test_module_05.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C        3579

      MODULE MYMOD6
          IMPLICIT NONE
          INTEGER :: X = 1234
      contains
          subroutine myadd(y, sum)
          integer y, sum

          sum = x+y
          end subroutine myadd
      END MODULE MYMOD6

      PROGRAM PGM
      USE mymod6
      integer v
      call myadd(2345,v)    ! << myadd is defined in module mymod6

      write (*,*) v

      stop
      end program PGM
