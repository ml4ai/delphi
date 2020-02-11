      module output_landscape_module
    
      implicit none 
      
      type output_waterbal
        real :: precip = 0.           !mm H2O        |prec falling on they HRU during timestep
        real :: snofall = 0.          !mm H2O        |amt of prec falling as snow, sleet or freezing rain during timestep
        real :: snomlt = 0.           !mm H2O        |amt of snow or ice melting during timestep
        real :: surq_gen = 0.         !mm H2O        |amt of surf runoff to main channel
        real :: latq = 0.             !mm H2O        |amt of lat flow contrib to main channel during HRU during mon
        real :: wateryld = 0.         !mm H2O        |water yld (tot amt of water entering main channel) from HRU during mon
        real :: perc = 0.             !mm H2O        |amt of water perc out of the soil profile and into the vadose zone in HRU during mon
        real :: et = 0.               !mm H2O        |actual ET in HRU during mon
        real :: tloss = 0.            !mm H2O        |amt of trans losses from trib channels in HRU for mon
        real :: eplant = 0.           !mm H2O        |actual amt of transpiration that occurs during mon in HRU
        real :: esoil = 0.            !mm H2O        |actual amt of evap (from soil) that occurs during mon in HRU
        real :: surq_cont = 0.        !mm H2O        |amt of surf runoff gen during mon in HRU
        real :: cn = 0.               !none          |CN values during mon in HRU
        real :: sw = 0.               !mm H2O        |soil water content of entire profile
        real :: sw_300 = 0.           !mm H2O        |soil water content of upper 300 mm
        real :: snopack = 0.          !mm            |snow pack
        real :: pet = 0.              !mm H2O        |pot et on current day in HRU
        real :: qtile = 0.            !mm H2O        |drainage tile flow contrib to main channel from HRU in mon
        real :: irr = 0.              !mm H2O        |amount of water applied to HRU
        real :: surq_runon = 0.       !mm H2O        |surface runoff from upland landscape units
        real :: latq_runon = 0.       !mm H2O        |lateral soil flow from upland landscape units
        real :: overbank = 0.         !mm H2O        |overbank flooding from channels
        real :: surq_cha = 0.         !mm H2O        |surface runoff flowing into channels
        real :: surq_res = 0.         !mm H2O        |surface runoff flowing into reservoirs
        real :: surq_ls = 0.          !mm H2O        |surface runoff flowing into a landscape element
        real :: latq_cha = 0.         !mm H2O        |lateral soil flow into channels
        real :: latq_res = 0.         !mm H2O        |lateral soil flow into reservoirs
        real :: latq_ls = 0.          !mm H2O        |lateral soil flow into a landscape element
      end type output_waterbal
       
      type (output_waterbal), dimension (:), allocatable :: hwb_d
      type (output_waterbal), dimension (:), allocatable :: hwb_m
      type (output_waterbal), dimension (:), allocatable :: hwb_y
      type (output_waterbal), dimension (:), allocatable :: hwb_a
      type (output_waterbal) :: hwbz
      
      type (output_waterbal), dimension (:), allocatable :: hltwb_d
      type (output_waterbal), dimension (:), allocatable :: hltwb_m
      type (output_waterbal), dimension (:), allocatable :: hltwb_y
      type (output_waterbal), dimension (:), allocatable :: hltwb_a
      
      type (output_waterbal), dimension (:), allocatable :: ruwb_d
      type (output_waterbal), dimension (:), allocatable :: ruwb_m
      type (output_waterbal), dimension (:), allocatable :: ruwb_y
      type (output_waterbal), dimension (:), allocatable :: ruwb_a
      
      type (output_waterbal) :: bwb_d
      type (output_waterbal) :: bwb_m
      type (output_waterbal) :: bwb_y
      type (output_waterbal) :: bwb_a
      
      type regional_output_waterbal
        type (output_waterbal), dimension (:), allocatable :: lum
      end type regional_output_waterbal
      type (regional_output_waterbal), dimension (:), allocatable :: rwb_d
      type (regional_output_waterbal), dimension (:), allocatable :: rwb_m
      type (regional_output_waterbal), dimension (:), allocatable :: rwb_y
      type (regional_output_waterbal), dimension (:), allocatable :: rwb_a
      
            
      type output_nutbal
        real :: grazn = 0.              !kg N/ha        |amt of nit added to soil in grazing on the day in HRU
        real :: grazp = 0.              !kg P/ha        |amt of phos added to soil in grazing on the day in HRU
        real :: lab_min_p = 0.          !kg P/ha        |amt of phos moving from the labile min pool to the active min pool
        !                                                    in the soil profile on the current day in the HRU
        real :: act_sta_p = 0.          !kg P/ha        |amt of phos moving from the active min pool to the stable min pool
        !                                                    in the soil profile on the current day in the HRU
        real :: fertn = 0.              !kg N/ha        |tot amt of nit applied to soil in HRU on day
        real :: fertp = 0.              !kg P/ha        |tot amt of phos applied to soil in HRU on day
        real :: fixn = 0.               !kg N/ha        |amt of nit added to plant biomass via fixation on the day in HRU
        real :: denit = 0.              !kg N/ha        |amt of nit lost from nitrate pool by denit in soil profile
        !                                                    on current day in HRU
        real :: act_nit_n = 0.          !kg N/ha        |amt of nit moving from active org to nitrate pool in soil profile
        !                                                    on current day in HRU 
        real :: act_sta_n = 0.          !kg N/ha        |amt of nit moving from active org to stable org pool in soil
        !                                                    profile on current day in HRU
        real :: org_lab_p = 0.          !kg P/ha        |amt of phos moving from the org to labile pool in soil profile
        !                                                    on current day in HRU
        real :: rsd_nitorg_n = 0.       !kg N/ha        |amt of nit moving from the fresh org (residue) to the nitrate(80%)
        !                                                    and active org(20%) pools in soil profile on current day in HRU
        real :: rsd_laborg_p = 0.       !kg P/ha        |amt of phos moving from the fresh org (residue) to the labile(80%)
        !                                                    and org(20%) pools in soil profile on current day in HRU
        real :: no3atmo = 0.            !kg N/ha        |nitrate added to the soil from atmospheric deposition (rainfall+dry)
        real :: nh4atmo = 0.            !kg N/ha        |ammonia added to the soil from atmospheric deposition (rainfall+dry)
      end type output_nutbal

      type (output_nutbal), dimension (:), allocatable :: hnb_d
      type (output_nutbal), dimension (:), allocatable :: hnb_m
      type (output_nutbal), dimension (:), allocatable :: hnb_y
      type (output_nutbal), dimension (:), allocatable :: hnb_a
      type (output_nutbal) :: hnbz
      
      type (output_nutbal), dimension (:), allocatable :: hltnb_d
      type (output_nutbal), dimension (:), allocatable :: hltnb_m
      type (output_nutbal), dimension (:), allocatable :: hltnb_y
      type (output_nutbal), dimension (:), allocatable :: hltnb_a
      
      type (output_nutbal), dimension (:), allocatable :: runb_d
      type (output_nutbal), dimension (:), allocatable :: runb_m
      type (output_nutbal), dimension (:), allocatable :: runb_y
      type (output_nutbal), dimension (:), allocatable :: runb_a
      
      type (output_nutbal) :: bnb_d
      type (output_nutbal) :: bnb_m
      type (output_nutbal) :: bnb_y
      type (output_nutbal) :: bnb_a
      
      type regional_output_nutbal
        type (output_nutbal), dimension (:), allocatable :: lum
      end type regional_output_nutbal
      type (regional_output_nutbal), dimension (:), allocatable :: rnb_d
      type (regional_output_nutbal), dimension (:), allocatable :: rnb_m
      type (regional_output_nutbal), dimension (:), allocatable :: rnb_y
      type (regional_output_nutbal), dimension (:), allocatable :: rnb_a
      
      type output_losses
        real :: sedyld = 0.               !metric tons    | daily soil loss caused by water erosion
        real :: sedorgn = 0.              !kg N/ha        | amt of org nit in surf runoff in HRU for the day
        real :: sedorgp = 0.              !kg P/ha        | amt of org phos in surf runoff in HRU for the day
        real :: surqno3 = 0.              !kg N/ha        | amt of NO3-N in surf runoff in HRU for the day
        real :: latno3 = 0.               !kg N/ha        | amt of NO3-N in lat flow in HRU for the day
        real :: surqsolp = 0.             !kg P/ha        | amt of soluble phos in surf runoff in HRU for the day
        real :: usle = 0.                 !metric tons/ha | daily soil loss predicted with USLE equation
        real :: sedmin = 0.               !
        real :: tileno3 = 0.              !kg N/ha        | NO3 in tile flow
      end type output_losses
      
      type (output_losses), dimension (:), allocatable :: hls_d
      type (output_losses), dimension (:), allocatable :: hls_m
      type (output_losses), dimension (:), allocatable :: hls_y
      type (output_losses), dimension (:), allocatable :: hls_a
      type (output_losses) :: hlsz
      
      type (output_losses), dimension (:), allocatable :: hltls_d
      type (output_losses), dimension (:), allocatable :: hltls_m
      type (output_losses), dimension (:), allocatable :: hltls_y
      type (output_losses), dimension (:), allocatable :: hltls_a
      
      type (output_losses), dimension (:), allocatable :: ruls_d
      type (output_losses), dimension (:), allocatable :: ruls_m
      type (output_losses), dimension (:), allocatable :: ruls_y
      type (output_losses), dimension (:), allocatable :: ruls_a
      
      type (output_losses) :: bls_d
      type (output_losses) :: bls_m
      type (output_losses) :: bls_y
      type (output_losses) :: bls_a
            
      type regional_output_losses
        type (output_losses), dimension (:), allocatable :: lum
      end type regional_output_losses
      type (regional_output_losses), dimension (:), allocatable :: rls_d
      type (regional_output_losses), dimension (:), allocatable :: rls_m
      type (regional_output_losses), dimension (:), allocatable :: rls_y
      type (regional_output_losses), dimension (:), allocatable :: rls_a
         
      type output_plantweather
        real :: lai = 0.                   !m**2/m**2     |leaf area index
        real :: bioms = 0.                 !kg/ha         |land cover/crop biomass 
        real :: yield = 0.                 !kg/ha         |yield (dry weight) by crop type
        real :: residue = 0.               !kga/ha        |initial residue cover
        real :: sol_tmp = 0.               !deg C         |daily average temperature of soil layer
        real :: strsw = 0.                 !0-1           |water (drought) stress
        real :: strsa = 0.                 !0-1           |water (aeration) stress
        real :: strstmp = 0.               !0-1           |temperature stress      
        real :: strsn = 0.                 !0-1           |nitrogen stress
        real :: strsp = 0.                 !0-1           |phosphorus stress
        real :: nplnt = 0.                 !kg N/ha       |plant uptake of nit in HRU for the day
        real :: percn = 0.                 !kg N/ha       |NO3-N leached from soil profile
        real :: pplnt = 0.                 !kg P/ha       |plant uptake of phos in HRU for the day
        real :: tmx = 0.                   !deg C         |maximum temperature for the day in HRU
        real :: tmn = 0.                   !deg C         |minimum temperature for the day in HRU
        real :: tmpav = 0.                 !deg C         |average air temperature on current day in HRU
        real :: solrad = 0.                !MJ/m^2        |solar radiation for the day in HRU
        real :: wndspd = 0.                !              |windspeed
        real :: rhum = 0.                  !              |relative humidity
        real :: phubase0 = 0.              !              |base zero potential heat units
      end type output_plantweather
      
      type (output_plantweather), dimension (:), allocatable :: hpw_d
      type (output_plantweather), dimension (:), allocatable :: hpw_m
      type (output_plantweather), dimension (:), allocatable :: hpw_y
      type (output_plantweather), dimension (:), allocatable :: hpw_a
      type (output_plantweather) :: hpwz
      
      type(output_plantweather), dimension (:), allocatable :: hltpw_d
      type(output_plantweather), dimension (:), allocatable :: hltpw_m
      type(output_plantweather), dimension (:), allocatable :: hltpw_y
      type(output_plantweather), dimension (:), allocatable :: hltpw_a
      
      type (output_plantweather), dimension (:), allocatable :: rupw_d
      type (output_plantweather), dimension (:), allocatable :: rupw_m
      type (output_plantweather), dimension (:), allocatable :: rupw_y
      type (output_plantweather), dimension (:), allocatable :: rupw_a
      
      type (output_plantweather) :: bpw_d
      type (output_plantweather) :: bpw_m
      type (output_plantweather) :: bpw_y
      type (output_plantweather) :: bpw_a
                  
      type regional_output_plantweather
        type (output_plantweather), dimension (:), allocatable :: lum
      end type regional_output_plantweather
      type (regional_output_plantweather), dimension (:), allocatable :: rpw_d
      type (regional_output_plantweather), dimension (:), allocatable :: rpw_m
      type (regional_output_plantweather), dimension (:), allocatable :: rpw_y
      type (regional_output_plantweather), dimension (:), allocatable :: rpw_a
      
      type output_waterbal_header
        character (len=5) :: day         =  " jday"
        character (len=6) :: mo          =  "   mon"
        character (len=6) :: day_mo      =  "   day"
        character (len=6) :: yrc         =  "    yr"
        character (len=8) :: isd         =  "    unit"
        character (len=8) :: id          =  "  gis_id"        
        character (len=16) :: name       =  "  name          "        
        character (len=14) :: precip     =  "        precip"
        character (len=12) :: snofall    =  "     snofall"
        character (len=12) :: snomlt     =  "      snomlt"        
        character (len=12) :: surq_gen   =  "    surq_gen"      
        character (len=12) :: latq       =  "        latq" 
        character (len=12) :: wateryld   =  "    wateryld"
        character (len=12) :: perc       =  "        perc"   
        character (len=12) :: et         =  "          et"
        character (len=12) :: tloss      =  "       tloss"
        character (len=12) :: eplant     =  "      eplant"
        character (len=12) :: esoil      =  "       esoil"
        character (len=12) :: surq_cont  =  "   surq_cont"
        character (len=12) :: cn         =  "          cn"
        character (len=12) :: sw         =  "          sw"
        character (len=12) :: sw_300     =  "      sw_300"
        character (len=12) :: snopack    =  "     snopack"   
        character (len=12) :: pet        =  "         pet"
        character (len=12) :: qtile      =  "       qtile"
        character (len=12) :: irr        =  "         irr"
        character (len=12) :: surq_runon =  "  surq_runon"
        character (len=12) :: latq_runon =  "  latq_runon"
        character (len=12) :: overbank   =  "    overbank"
        character (len=12) :: surq_cha   =  "    surq_cha"
        character (len=12) :: surq_res   =  "    surq_res"
        character (len=12) :: surq_ls    =  "     surq_ls"
        character (len=12) :: latq_cha   =  "    latq_cha"
        character (len=12) :: latq_res   =  "    latq_res"
        character (len=12) :: latq_ls    =  "     latq_ls"
      end type output_waterbal_header      
      type (output_waterbal_header) :: wb_hdr
      
      type output_waterbal_header_units
        character (len=5) :: day         =  "     "
        character (len=6) :: mo          =  "      "
        character (len=6) :: day_mo      =  "      "
        character (len=6) :: yrc         =  "      "
        character (len=8) :: isd         =  "        "
        character (len=8) :: id          =  "        "        
        character (len=16) :: name       =  "                "        
        character (len=14) :: precip     =  "          mm"
        character (len=12) :: snofall    =  "          mm"
        character (len=12) :: snomlt     =  "          mm"        
        character (len=12) :: surq_gen   =  "          mm"      
        character (len=12) :: latq       =  "          mm" 
        character (len=12) :: wateryld   =  "          mm"
        character (len=12) :: perc       =  "          mm"   
        character (len=12) :: et         =  "          mm"
        character (len=12) :: tloss      =  "          mm"
        character (len=12) :: eplant     =  "          mm"
        character (len=12) :: esoil      =  "          mm"
        character (len=12) :: surq_cont  =  "          mm"
        character (len=12) :: cn         =  "         ---"
        character (len=12) :: sw         =  "          mm"
        character (len=12) :: sw_300     =  "          mm"
        character (len=12) :: snopack    =  "          mm"  
        character (len=12) :: pet        =  "          mm"
        character (len=12) :: qtile      =  "          mm"
        character (len=12) :: irr        =  "          mm"
        character (len=12) :: surq_runon =  "          mm"
        character (len=12) :: latq_runon =  "          mm"
        character (len=12) :: overbank   =  "          mm"
        character (len=12) :: surq_cha   =  "          mm"
        character (len=12) :: surq_res   =  "          mm"
        character (len=12) :: surq_ls    =  "          mm"
        character (len=12) :: latq_cha   =  "          mm"
        character (len=12) :: latq_res   =  "          mm"
        character (len=12) :: latq_ls    =  "          mm"
      end type output_waterbal_header_units      
      type (output_waterbal_header_units) :: wb_hdr_units
      
       
      type output_nutbal_header
         character (len=5) :: day           =    " jday"
         character (len=6) :: mo            =    "   mon"
         character (len=6) :: day_mo        =    "   day"
         character (len=6) :: yrc           =    "    yr"
         character (len=9) :: isd           =    "    unit " 
         character (len=8) :: id            =    " gis_id "        
         character (len=9) :: name          =    "    name "         
         character(len=12) :: grazn         =    "        grzn"
         character(len=12) :: grazp         =    "        grzp"          
         character(len=17) :: lab_min_p     =    "        lab_min_p"     
         character(len=17) :: act_sta_p     =    "        act_sta_p"
         character(len=17) :: fertn         =    "            fertn"       
         character(len=17) :: fertp         =    "            fertp"       
         character(len=17) :: fixn          =    "             fixn"       
         character(len=17) :: denit         =    "            denit"
         character(len=17) :: act_nit_n     =    "        act_nit_n"
         character(len=17) :: act_sta_n     =    "        act_sta_n"
         character(len=17) :: org_lab_p     =    "        org_lab_p"
         character(len=17) :: rsd_nitorg_n  =    "     rsd_nitorg_n"      
         character(len=17) :: rsd_laborg_p  =    "     rsd_laborg_p"      
         character(len=17) :: no3atmo =    "          no3atmo" 
         character(len=17) :: nh4atmo =    "          nh4atmo"
      end type output_nutbal_header         
      type (output_nutbal_header) :: nb_hdr
      
      type output_nutbal_header_units
         character (len=5) :: day           =    "     "
         character (len=6) :: mo            =    "      "
         character (len=6) :: day_mo        =    "      "
         character (len=6) :: yrc           =    "      "
         character (len=9) :: isd           =    "         " 
         character (len=8) :: id            =    "        "        
         character (len=9) :: name          =    "         "         
         character(len=12) :: grazn         =    "        kgha"
         character(len=12) :: grazp         =    "        kgha"         
         character(len=17) :: lab_min_p     =    "             kgha"     
         character(len=17) :: act_sta_p     =    "             kgha" 
         character(len=17) :: fertn         =    "             kgha"        
         character(len=17) :: fertp         =    "             kgha"        
         character(len=17) :: fixn          =    "             kgha"        
         character(len=17) :: denit         =    "             kgha" 
         character(len=17) :: act_nit_n     =    "             kgha" 
         character(len=17) :: act_sta_n     =    "             kgha" 
         character(len=17) :: org_lab_p     =    "             kgha" 
         character(len=17) :: rsd_nitorg_n  =    "             kgha"       
         character(len=17) :: rsd_laborg_p  =    "             kgha"       
         character(len=17) :: no3atmo       =    "             kgha"  
         character(len=17) :: nh4atmo       =    "             kgha" 
      end type output_nutbal_header_units         
      type (output_nutbal_header_units) :: nb_hdr_units
      
      type output_losses_header
        character (len=6) :: day        =  "  jday"
        character (len=6) :: mo         =  "   mon"
        character (len=6) :: day_mo     =  "   day"
        character (len=6) :: yrc        =  "    yr"
        character (len=8) :: isd        =  "   unit "
        character (len=8) :: id         =  " gis_id "        
        character (len=16) :: name      =  " name               "        
        character (len=12) :: sedyld    =  "      sedyld"
        character (len=12)  :: sedorgn  =  "     sedorgn"
        character (len=12)  :: sedorgp  =  "     sedorgp"
        character (len=12)  :: surqno3  =  "     surqno3"
        character (len=12)  :: latno3   =  "     lat3no3"            
        character (len=12)  :: surqsolp =  "    surqsolp"
        character (len=12)  :: usle     =  "        usle"     
        character (len=12)  :: sedmin   =  "      sedmin"
        character (len=12)  :: tileno3  =  "     tileno3"
      end type output_losses_header      
      type (output_losses_header) :: ls_hdr
      
       type output_losses_header_units
        character (len=6) :: day        =  "      "
        character (len=6) :: mo         =  "      "
        character (len=6) :: day_mo     =  "      "
        character (len=6) :: yrc        =  "      "
        character (len=8) :: isd        =  "        "
        character (len=8) :: id         =  "        "        
        character (len=16) :: name      =  "                    "        
        character (len=12) :: sedyld    =  "         tha"
        character (len=12)  :: sedorgn  =  "        kgha"
        character (len=12)  :: sedorgp  =  "        kgha"
        character (len=12)  :: surqno3  =  "        kgha"
        character (len=12)  :: latno3   =  "        kgha"            
        character (len=12)  :: surqsolp =  "        kgha"
        character (len=12)  :: usle     =  "        tons"     
        character (len=12)  :: sedmin   =  "        ----"
        character (len=12)  :: tileno3  =  "        ----"
      end type output_losses_header_units      
      type (output_losses_header_units) :: ls_hdr_units
     
      type output_plantweather_header
        character (len=6) :: day        =  "  jday"
        character (len=6) :: mo         =  "   mon"
        character (len=6) :: day_mo     =  "   day"
        character (len=6) :: yrc        =  "    yr"
        character (len=8) :: isd        =  "   unit "
        character (len=8) :: id         =  " gis_id "        
        character (len=16) :: name      =  " name              "        
        character (len=13) :: lai       =  "          lai"
        character (len=12) :: bioms     =  "       bioms"
        character (len=12) :: yield     =  "       yield"
        character (len=12) :: residue   =  "     residue"
        character (len=12) :: sol_tmp   =  "     sol_tmp"
        character (len=12) :: strsw     =  "       strsw"
        character (len=12) :: strsa     =  "       strsa"
        character (len=12) :: strstmp   =  "     strstmp"
        character (len=12) :: strsn     =  "       strsn"
        character (len=12) :: strsp     =  "       strsp"
        character (len=12) :: nplnt     =  "        nplt"
        character (len=12) :: percn     =  "       percn"
        character (len=12) :: pplnt     =  "       pplnt"
        character (len=12) :: tmx       =  "         tmx"
        character (len=12) :: tmn       =  "         tmn"
        character (len=12) :: tmpav     =  "       tmpav"
        character (len=12) :: solrad    =  "     solarad"
        character (len=12) :: wndspd    =  "      wndspd"
        character (len=12) :: rhum      =  "        rhum"
        character (len=12) :: phubase0  =  "     phubas0"
      end type output_plantweather_header
      
      type (output_plantweather_header) :: pw_hdr
      
      type output_plantweather_header_units
        character (len=6) :: day        =  "      "
        character (len=6) :: mo         =  "      "
        character (len=6) :: day_mo     =  "      "
        character (len=6) :: yrc        =  "      "
        character (len=8) :: isd        =  "        "
        character (len=8) :: id         =  "        "        
        character (len=16) :: name      =  "                   " 
        character (len=13) :: lai       =  "    m**2/m**2"
        character (len=12) :: bioms     =  "        kgha"
        character (len=12) :: yield     =  "        kgha"
        character (len=12) :: residue   =  "        kgha"
        character (len=12) :: sol_tmp   =  "        degc"
        character (len=12) :: strsw     =  "         ----"
        character (len=12) :: strsa     =  "         ----"
        character (len=12) :: strstmp   =  "         ----"
        character (len=12) :: strsn     =  "         ----"
        character (len=12) :: strsp     =  "         ----"
        character (len=12) :: nplnt     =  "        kgha"
        character (len=12) :: percn     =  "        kgha"
        character (len=12) :: pplnt     =  "        kgha"
        character (len=12) :: tmx       =  "        degc"
        character (len=12) :: tmn       =  "        degc"
        character (len=12) :: tmpav     =  "        degc"
        character (len=12) :: solrad    =  "      mj/m^2"
        character (len=12) :: wndspd    =  "         m/s"
        character (len=12) :: rhum      =  "        frac"
        character (len=12) :: phubase0  =  "        degc"
      end type output_plantweather_header_units
      
      type (output_plantweather_header_units) :: pw_hdr_units
      
      
      interface operator (+)
        module procedure hruout_waterbal_add
      end interface
             
      interface operator (+)
        module procedure hruout_nutbal_add
      end interface
           
      interface operator (+)
        module procedure hruout_losses_add
      end interface
            
      interface operator (+)
        module procedure hruout_plantweather_add
      end interface
        
      interface operator (/)
        module procedure hruout_waterbal_div
      end interface
              
      interface operator (*)
        module procedure hruout_waterbal_mult
      end interface
      
      interface operator (//)
        module procedure hruout_waterbal_ave
      end interface
        
      interface operator (/)
        module procedure hruout_nutbal_div
      end interface
                
      interface operator (*)
        module procedure hruout_nutbal_mult
      end interface
        
      interface operator (/)
        module procedure hruout_losses_div
      end interface
                
      interface operator (*)
        module procedure hruout_losses_mult
      end interface
        
      interface operator (/)
        module procedure hruout_plantweather_div
        end interface
          
      interface operator (//)
        module procedure hruout_plantweather_ave
        end interface
            
      interface operator (*)
        module procedure hruout_plantweather_mult
        end interface
  
      contains

      function hruout_waterbal_add (hru1, hru2) result (hru3)
        type (output_waterbal), intent (in) :: hru1
        type (output_waterbal), intent (in) :: hru2
        type (output_waterbal) :: hru3
        hru3%precip = hru1%precip + hru2%precip
        hru3%snofall = hru1%snofall + hru2%snofall
        hru3%snomlt = hru1%snomlt + hru2%snomlt
        hru3%surq_gen = hru1%surq_gen + hru2%surq_gen
        hru3%latq = hru1%latq + hru2%latq
        hru3%wateryld = hru1%wateryld + hru2%wateryld
        hru3%perc = hru1%perc + hru2%perc
        hru3%et = hru1%et + hru2%et
        hru3%tloss = hru1%tloss + hru2%tloss
        hru3%eplant = hru1%eplant + hru2%eplant
        hru3%esoil = hru1%esoil + hru2%esoil
        hru3%surq_cont = hru1%surq_cont + hru2%surq_cont
        hru3%cn = hru1%cn + hru2%cn
        hru3%sw = hru1%sw + hru2%sw
        hru3%sw_300 = hru1%sw_300 + hru2%sw_300
        hru3%snopack = hru1%snopack + hru2%snopack
        hru3%pet = hru1%pet + hru2%pet
        hru3%qtile = hru1%qtile + hru2%qtile
        hru3%irr = hru1%irr + hru2%irr
        hru3%surq_runon = hru1%surq_runon + hru2%surq_runon
        hru3%latq_runon = hru1%latq_runon + hru2%latq_runon
        hru3%overbank = hru1%overbank + hru2%overbank
        hru3%surq_cha = hru1%surq_cha + hru2%surq_cha
        hru3%surq_res = hru1%surq_res + hru2%surq_res
        hru3%surq_ls = hru1%surq_ls + hru2%surq_ls
        hru3%latq_cha = hru1%latq_cha + hru2%latq_cha
        hru3%latq_res = hru1%latq_res + hru2%latq_res
        hru3%latq_ls = hru1%latq_ls + hru2%latq_ls
      end function hruout_waterbal_add
      
      function hruout_nutbal_add (hru1, hru2) result (hru3)
        type (output_nutbal), intent (in) :: hru1
        type (output_nutbal), intent (in) :: hru2
        type (output_nutbal) :: hru3
        hru3%grazn = hru1%grazn + hru2%grazn
        hru3%grazp = hru1%grazp + hru2%grazp
        hru3%lab_min_p = hru1%lab_min_p + hru2%lab_min_p
        hru3%act_sta_p = hru1%act_sta_p + hru2%act_sta_p
        hru3%fertn = hru1%fertn + hru2%fertn
        hru3%fertp = hru1%fertp + hru2%fertp
        hru3%fixn = hru1%fixn + hru2%fixn
        hru3%denit = hru1%denit + hru2%denit
        hru3%act_nit_n = hru1%act_nit_n + hru2%act_nit_n
        hru3%act_sta_n = hru1%act_sta_n + hru2%act_sta_n
        hru3%org_lab_p = hru1%org_lab_p + hru2%org_lab_p
        hru3%rsd_nitorg_n = hru1%rsd_nitorg_n + hru2%rsd_nitorg_n
        hru3%rsd_laborg_p = hru1%rsd_laborg_p + hru2%rsd_laborg_p
        hru3%no3atmo = hru1%no3atmo + hru2%no3atmo
        hru3%nh4atmo = hru1%nh4atmo + hru2%nh4atmo
      end function hruout_nutbal_add

      function hruout_losses_add (hru1, hru2) result (hru3)
        type (output_losses), intent (in) :: hru1
        type (output_losses), intent (in) :: hru2
        type (output_losses) :: hru3
        hru3%sedyld = hru1%sedyld + hru2%sedyld
        hru3%sedorgn = hru1%sedorgn + hru2%sedorgn
        hru3%sedorgp = hru1%sedorgp + hru2%sedorgp
        hru3%surqno3 = hru1%surqno3 + hru2%surqno3
        hru3%latno3 = hru1%latno3 + hru2%latno3
        hru3%surqsolp = hru1%surqsolp + hru2%surqsolp
        hru3%usle = hru1%usle + hru2%usle
        hru3%sedmin = hru1%sedmin + hru2%sedmin
        hru3%tileno3 = hru1%tileno3 + hru2%tileno3
      end function hruout_losses_add
      
      function hruout_plantweather_add (hru1, hru2) result (hru3)
        type (output_plantweather), intent (in) :: hru1
        type (output_plantweather), intent (in) :: hru2
        type (output_plantweather) :: hru3
        hru3%lai = hru1%lai + hru2%lai
        hru3%bioms = hru1%bioms + hru2%bioms
        hru3%yield = hru1%yield + hru2%yield
        hru3%residue = hru1%residue + hru2%residue
        hru3%sol_tmp = hru1%sol_tmp + hru2%sol_tmp
        hru3%strsw = hru1%strsw + hru2%strsw
        hru3%strsa = hru1%strsa + hru2%strsa
        hru3%strstmp = hru1%strstmp + hru2%strstmp
        hru3%strsn = hru1%strsn + hru2%strsn
        hru3%strsp = hru1%strsp + hru2%strsp
        hru3%nplnt = hru1%nplnt + hru2%nplnt
        hru3%percn = hru1%percn + hru2%percn
        hru3%tmx = hru1%tmx + hru2%tmx
        hru3%tmn = hru1%tmn + hru2%tmn
        hru3%tmpav = hru1%tmpav + hru2%tmpav
        hru3%wndspd = hru1%wndspd + hru2%wndspd
        hru3%rhum = hru1%rhum + hru2%rhum
        hru3%solrad = hru1%solrad + hru2%solrad
        hru3%phubase0 = hru1%phubase0 + hru2%phubase0
      end function hruout_plantweather_add

      function hruout_waterbal_div (hru1,const) result (hru2)
        type (output_waterbal), intent (in) :: hru1
        real, intent (in) :: const
        type (output_waterbal) :: hru2
        hru2%precip = hru1%precip / const
        hru2%snofall = hru1%snofall / const
        hru2%snomlt= hru1%snomlt / const
        hru2%surq_gen = hru1%surq_gen / const
        hru2%latq = hru1%latq / const
        hru2%wateryld = hru1%wateryld / const
        hru2%perc = hru1%perc / const
        hru2%et = hru1%et / const
        hru2%tloss = hru1%tloss / const
        hru2%eplant = hru1%eplant / const
        hru2%esoil = hru1%esoil / const
        hru2%surq_cont = hru1%surq_cont / const 
        hru2%cn = hru1%cn
        hru2%sw = hru1%sw
        hru2%sw_300 = hru1%sw_300
        hru2%snopack = hru1%snopack
        hru2%pet = hru1%pet / const 
        hru2%qtile = hru1%qtile / const 
        hru2%irr = hru1%irr / const
        hru2%surq_runon = hru1%surq_runon / const 
        hru2%latq_runon = hru1%latq_runon / const 
        hru2%overbank = hru1%overbank / const
        hru2%surq_cha = hru1%surq_cha / const 
        hru2%surq_res = hru1%surq_res / const 
        hru2%surq_ls = hru1%surq_ls / const
        hru2%latq_cha = hru1%latq_cha / const 
        hru2%latq_res = hru1%latq_res / const 
        hru2%latq_ls = hru1%latq_ls / const
      end function hruout_waterbal_div
      
      function hruout_waterbal_ave (hru1,const) result (hru2)
        type (output_waterbal), intent (in) :: hru1
        real, intent (in) :: const
        type (output_waterbal) :: hru2   
        hru2%precip = hru1%precip
        hru2%snofall = hru1%snofall
        hru2%snomlt= hru1%snomlt
        hru2%surq_gen = hru1%surq_gen
        hru2%latq = hru1%latq
        hru2%wateryld = hru1%wateryld
        hru2%perc = hru1%perc
        hru2%et = hru1%et
        hru2%tloss = hru1%tloss
        hru2%eplant = hru1%eplant
        hru2%esoil = hru1%esoil
        hru2%surq_cont = hru1%surq_cont
        hru2%cn = hru1%cn / const
        hru2%sw = hru1%sw / const
        hru2%sw_300 = hru1%sw_300 / const
        hru2%snopack = hru1%snopack / const
        hru2%pet = hru1%pet
        hru2%qtile = hru1%qtile
        hru2%irr = hru1%irr
        hru2%surq_runon = hru1%surq_runon
        hru2%latq_runon = hru1%latq_runon
        hru2%overbank = hru1%overbank
        hru2%surq_cha = hru1%surq_cha
        hru2%surq_res = hru1%surq_res
        hru2%surq_ls = hru1%surq_ls
        hru2%latq_cha = hru1%latq_cha
        hru2%latq_res = hru1%latq_res
        hru2%latq_ls = hru1%latq_ls
      end function hruout_waterbal_ave

      function hruout_waterbal_mult (hru1,const) result (hru2)
        type (output_waterbal), intent (in) :: hru1
        real, intent (in) :: const
        type (output_waterbal) :: hru2
        hru2%precip = hru1%precip * const
        hru2%snofall = hru1%snofall * const
        hru2%snomlt= hru1%snomlt * const
        hru2%surq_gen = hru1%surq_gen * const
        hru2%latq = hru1%latq * const
        hru2%wateryld = hru1%wateryld * const
        hru2%perc = hru1%perc * const
        hru2%et = hru1%et * const
        hru2%tloss = hru1%tloss * const
        hru2%eplant = hru1%eplant * const
        hru2%esoil = hru1%esoil * const
        hru2%surq_cont = hru1%surq_cont * const 
        hru2%cn = hru1%cn * const 
        hru2%sw = hru1%sw * const
        hru2%sw_300 = hru1%sw_300 * const
        hru2%snopack = hru1%snopack * const
        hru2%pet = hru1%pet * const 
        hru2%qtile = hru1%qtile * const 
        hru2%irr = hru1%irr * const
        hru2%surq_runon = hru1%surq_runon * const 
        hru2%latq_runon = hru1%latq_runon * const 
        hru2%overbank = hru1%overbank * const
        hru2%surq_cha = hru1%surq_cha * const 
        hru2%surq_res = hru1%surq_res * const 
        hru2%surq_ls = hru1%surq_ls * const
        hru2%latq_cha = hru1%latq_cha * const 
        hru2%latq_res = hru1%latq_res * const 
        hru2%latq_ls = hru1%latq_ls * const
      end function hruout_waterbal_mult
      
      
      function hruout_nutbal_div (hru1,const) result (hru2)
        type (output_nutbal), intent (in) :: hru1
        real, intent (in) :: const
        type (output_nutbal) :: hru2
        hru2%grazn = hru1%grazn / const
        hru2%grazp = hru1%grazp / const
        hru2%lab_min_p = hru1%lab_min_p / const
        hru2%act_sta_p = hru1%act_sta_p / const
        hru2%fertn = hru1%fertn / const
        hru2%fertp = hru1%fertp / const
        hru2%fixn = hru1%fixn / const
        hru2%denit = hru1%denit / const
        hru2%act_nit_n = hru1%act_nit_n / const
        hru2%act_sta_n = hru1%act_sta_n / const
        hru2%org_lab_p = hru1%org_lab_p / const
        hru2%rsd_nitorg_n = hru1%rsd_nitorg_n / const
        hru2%rsd_laborg_p = hru1%rsd_laborg_p / const
        hru2%no3atmo = hru1%no3atmo / const
        hru2%nh4atmo = hru1%nh4atmo / const
      end function hruout_nutbal_div
            
      function hruout_nutbal_mult (hru1,const) result (hru2)
        type (output_nutbal), intent (in) :: hru1
        real, intent (in) :: const
        type (output_nutbal) :: hru2
        hru2%grazn = hru1%grazn * const
        hru2%grazp = hru1%grazp * const
        hru2%lab_min_p = hru1%lab_min_p * const
        hru2%act_sta_p = hru1%act_sta_p * const
        hru2%fertn = hru1%fertn * const
        hru2%fertp = hru1%fertp * const
        hru2%fixn = hru1%fixn * const
        hru2%denit = hru1%denit * const
        hru2%act_nit_n = hru1%act_nit_n * const
        hru2%act_sta_n = hru1%act_sta_n * const
        hru2%org_lab_p = hru1%org_lab_p * const
        hru2%rsd_nitorg_n = hru1%rsd_nitorg_n * const
        hru2%rsd_laborg_p = hru1%rsd_laborg_p * const
        hru2%no3atmo = hru1%no3atmo * const
        hru2%nh4atmo = hru1%nh4atmo * const
      end function hruout_nutbal_mult
      
      function hruout_losses_div (hru1,const) result (hru2)
        type (output_losses), intent (in) :: hru1
        real, intent (in) :: const
        type (output_losses) :: hru2
        hru2%sedyld = hru1%sedyld / const
        hru2%sedorgn = hru1%sedorgn/ const
        hru2%sedorgp = hru1%sedorgp / const
        hru2%surqno3 = hru1%surqno3 / const
        hru2%latno3 = hru1%latno3 / const
        hru2%surqsolp = hru1%surqsolp / const
        hru2%usle = hru1%usle / const        
        hru2%sedmin = hru1%sedmin / const
        hru2%tileno3 = hru1%tileno3 / const
      end function hruout_losses_div
            
      function hruout_losses_mult (hru1,const) result (hru2)
        type (output_losses), intent (in) :: hru1
        real, intent (in) :: const
        type (output_losses) :: hru2
        hru2%sedyld = hru1%sedyld * const
        hru2%sedorgn = hru1%sedorgn * const
        hru2%sedorgp = hru1%sedorgp * const
        hru2%surqno3 = hru1%surqno3 * const
        hru2%latno3 = hru1%latno3 * const
        hru2%surqsolp = hru1%surqsolp * const
        hru2%usle = hru1%usle * const        
        hru2%sedmin = hru1%sedmin * const
        hru2%tileno3 = hru1%tileno3 * const
      end function hruout_losses_mult
      
      function hruout_plantweather_div (hru1,const) result (hru2)
        type (output_plantweather), intent (in) :: hru1
        real, intent (in) :: const
        type (output_plantweather) :: hru2
        hru2%lai = hru1%lai
        hru2%bioms = hru1%bioms
        hru2%yield = hru1%yield / const
        hru2%residue = hru1%residue
        hru2%sol_tmp = hru1%sol_tmp
        hru2%strsw = hru1%strsw / const
        hru2%strsa = hru1%strsa / const
        hru2%strstmp = hru1%strstmp / const
        hru2%strsn = hru1%strsn / const
        hru2%strsp = hru1%strsp / const
        hru2%nplnt = hru1%nplnt
        hru2%percn = hru1%percn / const
        hru2%pplnt = hru1%pplnt
        hru2%tmx = hru1%tmx
        hru2%tmn = hru1%tmn
        hru2%tmpav = hru1%tmpav
        hru2%solrad = hru1%solrad
        hru2%wndspd = hru1%wndspd
        hru2%rhum = hru1%rhum
        hru2%phubase0 = hru1%phubase0
      end function hruout_plantweather_div
                  
      function hruout_plantweather_ave (hru1,const) result (hru2)
        type (output_plantweather), intent (in) :: hru1
        real, intent (in) :: const
        type (output_plantweather) :: hru2
        hru2%lai = hru1%lai / const
        hru2%bioms = hru1%bioms / const
        hru2%yield = hru1%yield
        hru2%residue = hru1%residue / const
        hru2%sol_tmp = hru1%sol_tmp / const
        hru2%strsw = hru1%strsw
        hru2%strsa = hru1%strsa
        hru2%strstmp = hru1%strstmp
        hru2%strsn = hru1%strsn
        hru2%strsp = hru1%strsp
        hru2%nplnt = hru1%nplnt / const
        hru2%percn = hru1%percn
        hru2%pplnt = hru1%pplnt / const
        hru2%tmx = hru1%tmx / const
        hru2%tmn = hru1%tmn / const
        hru2%tmpav = hru1%tmpav / const
        hru2%wndspd = hru1%wndspd / const
        hru2%rhum = hru1%rhum / const
        hru2%solrad = hru1%solrad / const
        hru2%phubase0 = hru1%phubase0 / const
      end function hruout_plantweather_ave
                          
      function hruout_plantweather_mult (hru1,const) result (hru2)
        type (output_plantweather), intent (in) :: hru1
        real, intent (in) :: const
        type (output_plantweather) :: hru2
        hru2%lai = hru1%lai * const
        hru2%bioms = hru1%bioms * const
        hru2%yield = hru1%yield * const
        hru2%residue = hru1%residue * const
        hru2%sol_tmp = hru1%sol_tmp * const
        hru2%strsw = hru1%strsw * const
        hru2%strsa = hru1%strsa * const
        hru2%strstmp = hru1%strstmp * const
        hru2%strsn = hru1%strsn * const
        hru2%strsp = hru1%strsp * const
        hru2%nplnt = hru1%nplnt * const
        hru2%percn = hru1%percn * const
        hru2%pplnt = hru1%pplnt * const
        hru2%tmx = hru1%tmx * const
        hru2%tmn = hru1%tmn * const
        hru2%tmpav = hru1%tmpav * const
        hru2%solrad = hru1%solrad * const
        hru2%wndspd = hru1%wndspd * const
        hru2%rhum = hru1%rhum * const
        hru2%phubase0 = hru1%phubase0 * const
      end function hruout_plantweather_mult
                          
                            
      end module output_landscape_module