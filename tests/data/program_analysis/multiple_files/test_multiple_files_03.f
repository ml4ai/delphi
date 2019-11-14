C   This program uses 1 module that a file resides in another directory.
C   mathModule_4: ./math_files/mathModule_02.f

      program testing
        use mathModule_4
        implicit none

        print *, "Module in another file"

      end program testing
