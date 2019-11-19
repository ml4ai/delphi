C     File: multifile_multimod_01/main.f
C     Purpose: test for multiple files using multiple modules

      program main

      use ModuleDefs

      implicit none

      write (*,10) version%major, version%minor, VBranch
 10   format (I3,X,I3,X,A)

      end program main

