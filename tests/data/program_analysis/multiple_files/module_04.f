        module module_jkl
            use module_abc
            real :: symbol_9 = 123.456, symbol_10 = 654.321
        end module module_jkl

        module module_mno
            use module_jkl
            real :: symbol_11 = 888.456, symbol_12 = 654.999
        end module module_mno
