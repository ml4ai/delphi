C File: test_module_08.f
C Purpose: Illustrates the use of selective imports from Fortran modules.  In 
C    this example, the program PGM uses module MYMOD8 but only imports the
C    name myadd from MYMOD8.  Since MYMOD's variable X is not imported by PGM,
C    its value is not overwritten by the assignment to X in the body of PGM.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_08.f     # << this will create a file "mymod8.mod"
C    gfortran test_module_08.f        # << this will create a file "a.out"
C
C The output generated by this program is (see test_module_08-OUT.txt):
C
C    5678      6912

!-------------------------------------------------------------------------------
      MODULE MYMOD8
          IMPLICIT NONE
          INTEGER :: X = 1234
      contains
          integer function myadd(y)
          integer y

          myadd = x+y
          end function myadd
      END MODULE MYMOD8

!-------------------------------------------------------------------------------
      PROGRAM PGM
      USE mymod8, only : myadd  ! << Note: only myadd() is imported; x is not.
      integer x, v

      x = 5678    ! << This assignment does NOT overwrite mymod8's variable x

      v = myadd(x)

 10   FORMAT (I8,2X,I8)
      write (*,10) x, v

      stop
      end program PGM
!-------------------------------------------------------------------------------
