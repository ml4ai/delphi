C   This program uses 2 modules that one module (module_mno) uses
C   another module (module_jkl), which uses another module that
C   is in another file (module_abc).

      program testing
        use module_def
        use module_mno
        implicit none

        print *, "test_multiple_files_03.f"

      end program testing
