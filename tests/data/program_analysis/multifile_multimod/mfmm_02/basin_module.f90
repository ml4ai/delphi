      module basin_module
    
      implicit none
      
      character(len=80) :: prog
      
      type basin_inputs
        character(len=25) :: name
        real :: area_ls_ha = 0.
        real :: area_tot_ha = 0. 
      end type basin_inputs
      type (basin_inputs) :: bsn
      
      type basin_control_codes
        !character(len=16) :: update     !! pointer to basin updates in schedule.upd
        character(len=16) :: petfile     !! potential et filename
        character(len=16) :: wwqfile     !! watershed stream water quality filename
        integer :: pet = 0       !! potential ET method code
                                 !!   0 = Priestley-Taylor 
                                 !!   1 = Penman-Monteith
                                 !!   2 = Hargreaves method
                                 !!   3 = read in daily pot ET values
        integer :: event = 0     !! event code
        integer :: crk = 0       !! crack flow code 
                                 !!   1 = compute flow in cracks
        integer :: subwq = 0     !! subbasin water quality code
                                 !!   0 = do not calc algae/CBOD
                                 !!   1 = calc algae/CBOD
        integer :: sed_det = 0   !! max half-hour rainfall frac calc
                                 !!   0 = gen from triangular dist
                                 !!   1 = use monthly mean frac
        integer :: rte = 0       !! water routing method
                                 !!   0 variable storage method
                                 !!   1 Muskingum method
        integer :: deg = 0       !! channel degradation code
                                 !!   0 = do not compute
                                 !!   1 = compute (downcutting and widening)
        integer :: wq = 0        !! stream water quality code
                                 !!   0 do not model
                                 !!   1 model (QUAL2E)
        integer :: nostress = 0  !! redefined to the sequence number  -- changed to no nutrient stress
                                 !!   0 = all stresses applied
                                 !!   1 = turn off all plant stress
                                 !!   2 = turn off nutrient plant stress only
        integer :: cn = 0        !! CN method flag
                                 !!   0 = use traditional SWAT method bases CN 
                                 !!   CN on soil moisture
                                 !!   1 = use alternative bases CN on plant ET
                                 !!   2 = use traditional SWAT mathod bases CN on 
                                 !!   soil moisture but retention is adjusted for 
                                 !!   mildly-sloped tiled-drained watersheds
        integer :: cfac = 0      !!  0 = C-factor calc using CMIN
                                 !!  1 = for new C-factor from RUSLE (no min needed)      
        integer :: cswat = 0     !! carbon code
                                 !!  = 0 Static soil carbon (old mineralization routines)
                                 !!  = 1 C-FARM one carbon pool model 
                                 !!  = 2 Century model
        integer :: bf_flg = 0    !! baseflow distribution factor during the day for subdaily runs
                                 !!   0 = baseflow evenly distributed to each time step during the day
                                 !!   0.5 = even weights between even distribution and rainfall pattern
                                 !!   1 = profile of baseflow in a day follows rainfall pattern
        integer :: uhyd = 1      !! Unit hydrograph method: 
                                 !!   1 = triangular UH
                                 !!   2 = gamma function UH
        integer :: sed_ch = 0    !! Instream sediment model
                                 !!   0 = Bagnold model
                                 !!   1 = Brownlie model
                                 !!   2 = Yang model
        integer :: tdrn = 0      !! tile drainage eq code
                                 !!   1 = sim tile flow using subsurface drains (wt_shall)
                                 !!   0 = sim tile flow using subsurface origtile (wt_shall,d)
        integer :: wtdn = 0      !! water table depth algorithms code
                                 !!   1 = sim wt_shall using subsurface new water table depth routine
                                 !!   0 = sim wt_shall using subsurface orig water table depth routine
        integer :: sol_p_model=0 !! 1 = new soil P model
        integer :: abstr = 0     !! Initial abstraction on impervious cover (mm) 
        character(len=1) :: atmo = "a"   !! atmospheric deposition interval
                                         !!   "m" = monthly
                                         !!   "y" = yearly
                                         !!   "a" = annual
        integer :: smax = 0      !! max depressional storage selection code
                                 !!   1 = dynamic stmaxd computed as a cunfction of random
                                 !!          roughness and rain intensity
                                 !!   0 = static stmaxd read from .bsn for the global value or .sdr
                                 !! for specific hrus 
        integer :: i_subhw = 0   !! ***not used - headwater code (0=do not route; 1=route) 
      end type basin_control_codes
      type (basin_control_codes) :: bsn_cc

      type basin_parms
        real :: evlai = 3.0         !! none          |leaf area index at which no evap occurs
        real :: ffcb = 0.           !! none          |initial soil water cont expressed as a fraction of fc 
        real :: surlag = 4.0        !! days          |surface runoff lag time (days)
        real :: adj_pkr = 1.0       !! none          |peak rate adjustment factor in the subbasin
        real :: prf = 1.0           !! peak rate adjustment factor for sediment routing in the channel
        real :: spcon = 0.0001      !! linear parm for calc sed reentrained in channel sed routing
        real :: spexp = 1.0         !! exponent parameter for calc sed reentrained in channel sed routing
        real :: cmn = 0.0003        !! rate factor for mineralization on active org N
        real :: n_updis = 20.0      !! nitrogen uptake dist parm
        real :: p_updis = 20.0      !! phosphorus uptake dist parm
        real :: nperco = 20.0       !! nitrate perc coeff (0-1)
                                    !!   0 = conc of nitrate in surface runoff is zero
                                    !!   1 = perc has same conc of nitrate as surf runoff
        real :: pperco = 10.0       !! phos perc coeff (0-1)
                                    !!  0 = conc of sol P in surf runoff is zero
                                    !!  1 = percolate has some conc of sol P as surf runoff      
        real :: phoskd = 175.0      !! phos soil partitioning coef
        real :: psp = 0.40          !! phos availability index
        real :: rsdco = 0.05        !! residue decomposition coeff
        real :: percop = 0.5        !! pestcide perc coeff (0-1)
        real :: msk_co1 = 0.75      !! calibration coeff to control impact of the storage
                                    !!  time constant for the reach at bankfull depth
        real :: msk_co2 = 0.25      !! calibration coefficient used to control impact of the 
                                    !!   storage time constant for low flow (where low flow is when
                                    !!   river is at 0.1 bankfull depth) upon the Km value calculated
                                    !!   for the reach
        real :: msk_x = 0.20        !! weighting factor control relative importance of inflow rate 
                                    !!  and outflow rate in determining storage on reach
        real :: trnsrch             !! fraction of transmission losses from main channel that enter
                                    !!  deep aquifer
        real :: evrch = 0.60        !! reach evaporation adjustment factor
        real :: scoef = 1.0         !! channel storage coefficient (0-1)
        real :: cdn = 1.40          !! denitrification expoential rate coefficient        
        real :: sdnco = 1.30        !! denitrification threshold frac of field cap
        real :: bact_swf = 0.15     !! frac of manure containing active colony forming units
        real :: tb_adj = 0.         !! adjustment factor for subdaily unit hydrograph basetime
        real :: cn_froz = 0.000862  !! parameter for frozen soil adjustment on infiltraion/runoff
        real :: dorm_hr = -1.       !! time threshold used to define dormant (hrs)
        real :: open_var2           !! variable not used
        real :: fixco = 0.50        !! nitrogen fixation coeff
        real :: nfixmx = 20.0       !! max daily n-fixation (kg/ha)
        real :: decr_min = 0.01     !! minimum daily residue decay
        real :: rsd_covco = 0.30    !! residue cover factor for computing frac of cover         
        real :: vcrit = 0.          !! critical velocity
        real :: res_stlr_co = 0.184 !! reservoir sediment settling coeff
        real :: uhalpha = 1.0       !! alpha coeff for est unit hydrograph using gamma func
        real :: eros_spl = 0.       !! coeff of splash erosion varing 0.9-3.1 
        real :: rill_mult = 0.      !! rill erosion coefficient
        real :: eros_expo = 0.      !! exponential coeffcient for overland flow
        real :: c_factor = 0.       !! scaling parameter for cover and management factor for 
                                    !!  overland flow erosion
        real :: ch_d50 = 0.         !! median particle diameter of main channel (mm)
        real :: sig_g = 0.          !! geometric std dev of part sizes for the main channel
        real :: open_var3 = 0.      !! curve number retention parameter adjustment for low gradient
                                    !!  non-draining soils
        integer :: igen = 0         !!  random generator code: 
                                    !!   0 = use default numbers
                                    !!   1 = generate new numbers in every simulation 
      end type basin_parms
      type (basin_parms) :: bsn_prm

      type print_interval
        character(len=1) :: d = "n"
        character(len=1) :: m = "n"
        character(len=1) :: y = "n"
        character(len=1) :: a = "n"
      end type print_interval
      
      type basin_print_codes
      !!    PRINT CODES: "avann" = average annual (always print....unless input is "null")
      !!                 "year"  = yearly
      !!                 "mon"   = monthly
      !!                 "day"   = daily 
      
        character (len=1)  :: day_print = "n"
        character (len=1)  :: day_print_over = "n"
        integer :: nyskip = 0                           !!  number of years to skip output summarization
      ! DAILY START/END AND INTERVAL
        integer :: day_start = 0                        !!  julian day to start printing output
        integer :: day_end = 0                          !!  julian day to end printing output
        integer :: yrc_start = 0                        !!  calendar year to start printing output
        integer :: yrc_end = 0                          !!  calendar year to end printing output
        integer :: int_day = 1                          !!  interval between daily printing
        integer :: int_day_cur = 1                      !!  current day since last print
      ! AVE ANNUAL END YEARS
        integer :: aa_numint                          !! number of print intervals for ave annual output
        integer, dimension(:), allocatable :: aa_yrs  !! end years for ave annual output
      ! SPECIAL OUTPUTS
        character(len=1) :: csvout = "    n"         !!  code to print .csv files n=no print; y=print;
        character(len=1) :: dbout  = "    n"         !!  code to print database (db) files n=no print; y=print;
        character(len=1) :: cdfout = "    n"         !!  code to print netcdf (cdf) files n=no print; y=print;
      ! OTHER OUTPUTS
        character(len=1) :: snutc  = "    a"         !!  soils nutrients carbon output (default ave annual-d,m,y,a input)
        character(len=1) :: mgtout = "    n"         !!  management output file (mgt.out) (default ave annual-d,m,y,a input)
        character(len=1) :: hydcon = "    n"         !!  hydrograph connect output file (hydcon.out)
        character(len=1) :: fdcout = "    n"         !!  flow duration curve output n=no print; avann=print;
      ! BASIN
        type(print_interval) :: wb_bsn          !!  water balance BASIN output
        type(print_interval) :: nb_bsn          !!  nutrient balance BASIN output
        type(print_interval) :: ls_bsn          !!  losses BASIN output
        type(print_interval) :: pw_bsn          !!  plant weather BASIN output
        type(print_interval) :: aqu_bsn         !!  
        type(print_interval) :: res_bsn         !!
        type(print_interval) :: chan_bsn        !!
        type(print_interval) :: sd_chan_bsn     !!
        type(print_interval) :: recall_bsn      !!
      ! REGION
        type(print_interval) :: wb_reg          !!  water balance REGION output
        type(print_interval) :: nb_reg          !!  nutrient balance REGION output
        type(print_interval) :: ls_reg          !!  losses REGION output
        type(print_interval) :: pw_reg          !!  plant weather REGION output
        type(print_interval) :: aqu_reg         !!  
        type(print_interval) :: res_reg         !!
        type(print_interval) :: chan_reg        !!
        type(print_interval) :: sd_chan_reg     !! 
        type(print_interval) :: recall_reg      !!
       ! LSU
        type(print_interval) :: wb_lsu          !!  water balance LSU output
        type(print_interval) :: nb_lsu          !!  nutrient balance LSU output
        type(print_interval) :: ls_lsu          !!  losses LSU output
        type(print_interval) :: pw_lsu          !!  plant weather LSU output
        ! HRU
        type(print_interval) :: wb_hru          !!  water balance HRU output
        type(print_interval) :: nb_hru          !!  nutrient balance HRU output
        type(print_interval) :: ls_hru          !!  losses HRU output
        type(print_interval) :: pw_hru          !!  plant weather HRU output
        ! HRU-LTE
        type(print_interval) :: wb_sd           !!  water balance SWAT-DEG output 
        type(print_interval) :: nb_sd           !!  nutrient balance SWAT-DEG output
        type(print_interval) :: ls_sd           !!  losses SWAT-DEG output
        type(print_interval) :: pw_sd           !!  plant weather SWAT-DEG output
        ! CHANNEL
        type(print_interval) :: chan            !!  channel output
        ! CHANNEL_LTE
        type(print_interval) :: sd_chan         !!  swat deg (lte) channel output
        ! AQUIFER
        type(print_interval) :: aqu             !!  aqufier output
        ! RESERVOIR
        type(print_interval) :: res             !!  reservoir output
        ! RECALL
        type(print_interval) :: recall          !!  recall output
        ! HYDIN AND HYDOUT
        type(print_interval) :: hyd             !!  hydin_output and hydout_output
        type(print_interval) :: ru
        type(print_interval) :: pest            !!  all constituents pesticide output files (hru, chan, res, basin_chan, basin_res,
                                                !!        basin_ls
      end type basin_print_codes
      type (basin_print_codes) :: pco
      type (basin_print_codes) :: pco_init
      
      type mgt_header         
          character (len=12) :: hru =       "        hru"
          character (len=12) :: year =      "       year"
          character (len=12) :: mon =       "        mon"
          character (len=11) :: day =       "        day"
          character (len=15) :: crop =      " crop/fert/pest"
          character (len=12) :: oper =      " operation"          
          character (len=12) :: phub =      "phubase"  
          character (len=11) :: phua =      "   phuplant"  
          character (len=12) :: sw =        "  soil_water" 
          character (len=17) :: bio =       "      plant_bioms"
          character (len=11) :: rsd =       "   surf_rsd"
          character (len=15) :: solno3 =    "       soil_no3"
          character (len=15) :: solp =      "      soil_solp"
          character (len=15) :: op_var =    "         op_var"
          character (len=15) :: var1 =      "           var1"
          character (len=14) :: var2 =      "          var2"
          character (len=17) :: var3 =      "             var3"
          character (len=17) :: var4 =      "             var4"
          character (len=16) :: var5 =      "            var5"    
          character (len=16) :: var6 =      "            var6"    
          character (len=16) :: var7 =      "           var7"    
      end type mgt_header
      type (mgt_header) :: mgt_hdr

      type mgt_header_unit1         
          character (len=12) :: hru =       "        --- "
          character (len=12) :: year =      "        --- "
          character (len=12) :: mon =       "        --- "
          character (len=12) :: day =       "        --- "
          character (len=11) :: crop =      "      ---  "
          character (len=13) :: oper =      "       ---   "          
          character (len=9) :: phub =       "    deg_c"  
          character (len=16) :: phua =      "           deg_c"  
          character (len=12) :: sw =        "          mm" 
          character (len=17) :: bio =       "            kg/ha"
          character (len=11) :: rsd =       "      kg/ha"
          character (len=15) :: solno3 =    "          kg/ha"
          character (len=15) :: solp =      "          kg/ha"
          character (len=15) :: op_var =    "          --- "
          character (len=16) :: var1 =      "            --- "
          character (len=15) :: var2 =      "          --- "
          character (len=16) :: var3 =      "            ---"
          character (len=16) :: var4 =      "             ---"
          character (len=16) :: var5 =      "             ---"       
          character (len=16) :: var6 =      "             ---"  
          character (len=15) :: var7 =      "            ---"  
      end type mgt_header_unit1
      type(mgt_header_unit1) :: mgt_hdr_unt1
      
     
      type snutc_header                              
          character (len=12) :: day =           "         day"
          character (len=12) :: year =          "        year"
          character (len=12) :: hru =           "         hru"                                                       
          character (len=14) :: soil_mn_no3 =   " soil_mn_no3  "
          character (len=16) :: soil_mn_nh4 =   "    soil_mn_nh4 "
          character (len=14) :: soil_mp_wsol =  "  soil_mp_wsol"
          character (len=13) :: soil_mp_lab  =  "  soil_mp_lab"  
          character (len=13 ) :: soil_mp_act  = "  soil_mp_act"
          character (len=15) :: soil_mp_sta  =  "    soil_mp_sta"
          character (len=19) :: soil_tot_m =    "         soil_tot_m"
          character (len=14) :: soil_tot_c =    "    soil_tot_c  "
          character (len=14) :: soil_tot_n =    "    soil_tot_n  "
          character (len=15) :: soil_tot_p  =   "    soil_tot_p  " 
          character (len=18) :: soil_str_m =    "    soil_str_m  "
          character (len=14) :: soil_str_c =    "    soil_str_c  "
          character (len=14) :: soil_str_n =    "    soil_str_n  "
          character (len=14) :: soil_str_p  =   "    soil_str_p"           
          character (len=16) :: soil_lig_m =    "    soil_lig_m  "
          character (len=14) :: soil_lig_c =    "    soil_lig_c  "
          character (len=14) :: soil_lig_n =    "    soil_lig_n  "
          character (len=14) :: soil_lig_p  =   "    soil_lig_p  " 
          character (len=14) :: soil_meta_m =   "   soil_meta_m  "
          character (len=14) :: soil_meta_c =   "   soil_meta_c  "
          character (len=14) :: soil_meta_n =   "   soil_meta_n  "
          character (len=14) :: soil_meat_p  =  "   soil_meta_p  "
          character (len=14) :: soil_man_m =    "    soil_man_m  "
          character (len=14) :: soil_man_c =    "    soil_man_c  "
          character (len=14) :: soil_man_n =    "    soil_man_n  "
          character (len=14) :: soil_man_p  =   "    soil_man_p  " 
          character (len=14) :: soil_hs_m =     "    soil_hs_m   "
          character (len=14) :: soil_hs_c =     "    soil_hs_c   "
          character (len=16) :: soil_hs_n =     "    soil_hs_n   "
          character (len=16) :: soil_hs_p  =    "    soil_hs_p   "   
          character (len=16) :: soil_hp_m =     "    soil_hp_m   "
          character (len=16) :: soil_hp_c =     "    soil_hp_c   "
          character (len=16) :: soil_hp_n =     "    soil_hp_n   "
          character (len=16) :: soil_hp_p  =    "    soil_hp_p   "
          character (len=16) :: soil_microb_m = " soil_microb_m  "
          character (len=16) :: soil_microb_c = " soil_microb_c  "
          character (len=16) :: soil_microb_n = " soil_microb_n  "
          character (len=16) :: soil_microb_p  =" soil_microb_p  "  
          character (len=16) :: soil_water_m =  "   soil_water_m "
          character (len=16) :: soil_water_c =  "   soil_water_c "
          character (len=16) :: soil_water_n =  "   soil_water_n "
          character (len=16) :: soil_water_p  = "   soil_water_p "  
      end type snutc_header
      type(snutc_header) :: snutc_hdr
      
      type basin_yld_header                              
          character (len=10) :: year =       "      year "                                                     
          character (len=16) :: plant_no =   "     plant_no"
          character (len=16) :: plant_name = "plant_name "
          character (len=16) :: area_ha =    " harv_area(ha)   "  
          character (len=16) :: yield_t =    "  yld(t)         "
          character (len=16) :: yield_tha =  " yld(t/ha)      "
      end type basin_yld_header
      type (basin_yld_header) :: bsn_yld_hdr
      
      end module basin_module