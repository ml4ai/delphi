C   This program uses 3 modules, which are residing in 3 different
C   files. module_abc: ./module_01.f, module_def:
C   ./module_files/module_02.f, and module_ghi: ./module_03.f.

      program testing
        use module_abc
        use module_def
        use module_ghi
        implicit none

        print *, "Test 2"
      end program testing
