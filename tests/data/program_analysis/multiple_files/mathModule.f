C File: mathModule.c
C This file holds an external module that 
C will be included in other files.
C Command line to compile and link module in this file to other files
C that uses it:
C
C   $ gfortran -c mathModule.c __program_file.f__
C   $ gfortran mathModule.o __program_file.o__
C
C Source: Fortran Wiki - Compiling and linking modules
C http://fortranwiki.org/fortran/show/Compiling+and+linking+modules

      module mathModule

      implicit none
      private
      real, public, parameter :: pi = 3.1415, e = 2.7183, gamma = 0.57722

      end module mathModule

      
      module mathModule_2

      implicit none
      private
      real, public, parameter :: pi2 = 3.1415, e2 = 2.7183, gamma2 = 0.57722

      end module mathModule_2


      module mathModule_3

      implicit none
      private
      real, public, parameter :: pi3 = 3.1415, e3 = 2.7183, gamma3 = 0.57722

      end module mathModule_3
