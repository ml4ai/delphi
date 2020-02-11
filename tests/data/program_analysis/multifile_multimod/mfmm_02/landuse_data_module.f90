      module landuse_data_module
    
      implicit none
     
      type land_use_management
        character (len=16) :: name = " "
        character (len=16) :: cal_group = " "
        character (len=35) :: plant_cov = ""
        character (len=35) :: mgt_ops = ""
        character (len=16) :: cn_lu      !! none     | land use for curve number table (cntable.lum)
        character (len=16) :: cons_prac  !! none     | conservation practice from table (cons_practice.lum)
        character (len=16) :: urb_lu     !! none     | type of urban land use- ie. residential, industrial, etc (urban.urb)
        character (len=16) :: urb_ro     !! none     | urban runoff model
                                         !!          | "usgs_reg", simulate using USGS regression eqs
                                         !!          | "buildup_washoff", simulate using build up/wash off alg       
        character (len=16) :: ovn        !! none     | Manning"s "n" land use type for overland flow (ovn_table.lum)
        !integer :: urb_lu = 0           !! none     | urban land type identification number
        !integer :: iurban = 0           !! none     | urban simulation code 
        !                                !!          | 0  no urban sections in HRU
        !                                !!          | 1  urban sections in HRU, simulate using USGS regression eqs
        !                                !!          | 2  urban sections in HRU, simulate using build up/wash off alg
        !real :: ovn = 0.1               !! none     | Manning"s "n" value for overland flow
        character (len=25) :: tiledrain
        character (len=25) :: septic
        character (len=25) :: fstrip
        character (len=25) :: grassww
        character (len=25) :: bmpuser
      end type land_use_management
      type (land_use_management), dimension (:), allocatable :: lum
      
      type land_use_structures
        integer :: plant_cov = 0
        integer :: mgt_ops = 0
        integer :: cn_lu = 0
        integer :: cons_prac = 0
        integer :: tiledrain = 0
        integer :: septic = 0
        integer :: fstrip = 0
        integer :: grassww = 0
        integer :: bmpuser = 0
      end type land_use_structures
      type (land_use_structures), dimension (:), allocatable :: lum_str
           
      type curvenumber_table
        character(len=16) :: name                      !name includes abbrev for lu/treatment/condition 
        real, dimension(4) :: cn = (/30.,55.,70.,77./) !curve number
      end type curvenumber_table
      type (curvenumber_table), dimension (:), allocatable :: cn
      
      type land_use_mgt_groups
        integer :: num
        character(len=16), dimension(:), allocatable :: name    !land use groups
      end type
      type (land_use_mgt_groups) :: lum_grp
      
      type conservation_practice_table
        character(len=16) :: name                   !name of conservation practice
        real :: pfac = 1.0                          !usle p factor
        real :: sl_len_mx = 1.0             !m      !maximum slope length
      end type conservation_practice_table
      type (conservation_practice_table), dimension (:), allocatable :: cons_prac
                       
      type overlandflow_n_table
        character(len=16) :: name                   !name of conservation practice
        real :: ovn = 0.5                           !overland flow mannings n - mean
        real :: ovn_min = 0.5                       !overland flow mannings n - min
        real :: ovn_max = 0.5                       !overland flow mannings n - max
      end type overlandflow_n_table
      type (overlandflow_n_table), dimension (:), allocatable :: overland_n
    
      end module landuse_data_module 