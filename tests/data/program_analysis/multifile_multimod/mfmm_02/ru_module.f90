      module ru_module
    
      implicit none 

      integer :: iru                               !none            |counter
      integer :: mru_db                            !                |
      real, dimension (:), allocatable :: ru_tc    !                |    
      real, dimension (:), allocatable :: ru_n     !                | 
      real, dimension (:), allocatable :: hyd_flo  !                |
      integer, dimension (:), allocatable :: itsb  !none            |end of loop 
      real, dimension (:,:), allocatable :: uhs    !                |
   
      type ru_databases_char
        character(len=16) :: elem_def = ""
        character(len=16) :: elem_dr = ""
        character(len=16) :: toposub_db = ""
        character(len=16) :: field_db = ""
      end type ru_databases_char
      
      type ru_databases
        integer :: elem_def = 1
        integer :: elem_dr = 1
        integer :: toposub_db = 1
        integer :: field_db = 1
      end type ru_databases
    
      type ru_parameters
        character(len=16) :: name = ""
        real :: da_km2 = 0.      !! km2      drainage area
        type (ru_databases_char) :: dbsc
        type (ru_databases) :: dbs
      end type ru_parameters
      type (ru_parameters), dimension(:), allocatable :: ru

      end module ru_module