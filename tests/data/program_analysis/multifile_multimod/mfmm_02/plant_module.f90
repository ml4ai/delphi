     module plant_module
    
      implicit none
     
      integer :: basin_plants           !! number of different plants in the basin
      
      type plant_growth
         character(len=4) :: cpnm       !! N/A              |plant name 
         real :: cht = 0.               !! m                |canopy height 
         real :: lai = 0.               !! m**2/m**2        |leaf area index
         real :: plet = 0.              !! mm H2O           |actual ET simulated during life of plant
         real :: plpet = 0.             !! mm H2O           |potential ET simulated during life of plant
         real :: laimxfr = 0.           !! 
         real :: hi_adj = 0.            !! (kg/ha)/(kg/ha)  |temperature adjusted harvest index for current time during growing season
         real :: hi_prev = 0.           !! (kg/ha)/(kg/ha)  |optimal harvest index for current time during growing season
         real :: olai = 0.              !!
         real :: leaf_frac = 0.         !! none             |fraction of above ground tree biomass that is leaf
         real :: root_dep = 0.          !! mm               |root depth
         real :: root_frac = 0.         !! kg/ha            |root fraction of total plant mass
      end type plant_growth
      
      type plant_mass
        !character(len=4) :: cpnm      !! N/A              |4 letter char code represents organic name
        !real :: mass = 0.             !!kg/ha             |biomass
        !real :: cmass = 0.            !!kg/ha             |carbon mass
        !real :: nmass = 0.            !!kg/ha             |nitrogen mass
        !real :: pmass = 0.            !!kg/ha             |phosphorus mass
        real :: c_fr = 0.             !!none              |carbon fraction
        real :: n_fr = 0.             !!none              |nitrogen fraction
        real :: p_fr = 0.             !!none              |phosphorus fraction
      end type plant_mass
      !type (plant_mass) :: plt_mass_z
      type (plant_mass) :: yld_tbr
      type (plant_mass) :: yld_grn
      type (plant_mass) :: yld_veg
      type (plant_mass) :: yld_rsd
      !type (plant_mass), pointer :: pl_tot
      
      type plant_status
        integer :: idplt = 0            !! none         land cover code from plants.plt
        integer :: bsn_num = 0         !!none              |basin plant number
        character(len=1) :: gro = "y"   !               |land cover status
                                        !               |n = no land cover growing
                                        !               |y = land cover growing
        character(len=1) :: idorm = "n" !! none         |dormancy status code; 'n'=land cover growing 'y'=land cover dormant
        real :: phumat = 0.             !! C            |heat units to maturity
        real :: phuacc = 0.             !! fraction     |fraction of plant heatunit accumulated
        real :: laimx_pop = 0.          !!
        integer :: harv_num = 0         !!              |number of harvest operations
        integer :: curyr_mat = 1        !! 
        integer :: curyr_gro = 1        !!
        real :: pop_com = 0.            !! none 
        integer :: monsoon_init = 0     !! julian day   |monsoon initiation period
        integer :: days_senes = 0.      !! mm           |days since scenesence began (for moisture growth perennials)
        real :: leaf_tov = 0.           !! none         |leaf turnover rate - decline in lai and leaf biomass
      end type plant_status
      
      type plant_stress
        real :: reg                     !! none         |stress factor that most limits plant growth
                                        !!                on current day
        real :: strsw = 1.              !! none         |frac of potential plant growth achieved on the day where the
                                        !!                reduction is caused by water stress
        real :: strsa = 1.              !!              |frac of potential plant growth achieved on the day where the
                                        !!                reduction is caused by air stress
        real :: strsn = 1.              !! none         |frac of potential plant growth achieved on the day where the reduction
                                        !!                is caused by nit stress
        real :: strsp = 1.              !! none         |frac of potential plant growth achieved on the day where the reduction 
                                        !!                is caused by phos stress
        real :: strst = 1.              !! none         |frac of potential plant growth achieved on the day where the reduction
                                        !!                is caused by temp stress
        real :: sum_w = 0.              !! none         |sum of water stress
        real :: sum_tmp = 0.            !! none         |sum of temperature stress
        real :: sum_n = 0.              !! none         |sum of nitrogen stress
        real :: sum_p = 0.              !! none         |sum of phosphorus stress
        real :: sum_a = 0.              !! none         |sum of aeration stress 
      end type plant_stress
      
      type auto_operations
        integer, dimension(:), allocatable :: num_actions      !! current number of actions - reset on January 1 
      end type auto_operations
      
      type plant_community
       character(len=4) :: name
       integer :: npl                   !! number of plants in community
       integer :: pcomdb                !! current plant community database number
       integer :: rot_yr = 1            !! rotation year
       integer :: days_plant = 1        !!               |days since last harvest - for conditional scheduling planting of winter annuals
       integer :: days_harv = 1         !!               |days since last harvest - for conditional scheduling planting of winter annuals
       integer :: mseas = 0             !! none          |monsoon season to initiate tropical plant growth
                                        !!               |   0 = outside monsoon period and during monsoon after growth is triggered
                                        !!               |   1 = in monsoon period but new growth not triggered
       real :: cht_mx = 0.              !! m             |height of tallest plant in community for pet calculation
       real :: lai_sum = 0.             !! m/m           |sum of lai for each plant
       type (auto_operations), dimension(:), allocatable :: dtbl               !!d_tble action - to limit number of actions per year 
       type (plant_growth), dimension(:), allocatable :: plg    !!plant growth variables
       type (plant_stress), dimension(:), allocatable :: plstr  !!plant stress variables
       type (plant_status), dimension(:), allocatable :: plcur  !!plant status variables
       type (plant_mass), dimension(:), allocatable :: plm      !kg/ha            |total biomass for individual plant in community
      end type plant_community
      type (plant_community), dimension (:), allocatable :: pcom
      type (plant_community), dimension (:), allocatable :: pcom_init
      type (plant_growth) :: plgz
      type (plant_mass) :: plmz
      type (plant_mass) :: o_m1, o_m2, o_m3
      type (plant_stress) :: plstrz

      type basin_crop_yields
        real :: area_ha = 0.        !ha         |area of crop harvested
        real :: yield = 0.          !t          |yield mass removed in harvest
      end type basin_crop_yields
      type (basin_crop_yields), dimension(:), allocatable :: bsn_crop_yld
      type (basin_crop_yields), dimension(:), allocatable :: bsn_crop_yld_aa
      type (basin_crop_yields) :: bsn_crop_yld_z
        
      type plant_carbon
        real :: leaf = .41      !none   |carbon fraction in leaves
        real :: stem = .46      !none   |carbon fraction in stem
        real :: seed = .45      !none   |carbon fraction in seeds
        real :: root = .46      !none   |carbon fraction in roots
      end type plant_carbon
      type (plant_carbon) :: c_frac
            
     end module plant_module