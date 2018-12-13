C File: test_module_06.f
C Purpose: Illustrates a Fortran module that defines a subroutine which is
C    called from the main program.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_06.f     # << this will create a file "mymod6.mod"
C    gfortran test_module_06.f        # << this will create a file "a.out"
C
C When executed this will generate the output
C
C       11356

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

      x = 5678        ! << x is defined  in module mymod6.  
                      ! << This assignment to x overwrites its initialization value

      call myadd(x,v) ! << myadd is defined in module mymod6

      write (*,*) v

      stop
      end program PGM
