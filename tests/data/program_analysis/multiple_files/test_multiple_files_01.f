C File: test_multiple_files_01.f
C This file uses (include) mathModule, which resides in
C a separate file mathModule.f.
C Source: Fortran Wiki - Compiling and linking modules
C http://fortranwiki.org/fortran/show/Compiling+and+linking+modules

      program testing
        use mathModule
        implicit none

        print *, "pi:", pi, "e:", e, "gamma:", gamma  

      end program testing
