      module hru_module
    
      implicit none
      
      integer :: isep                !          |
      integer :: ith                 !          |
      integer :: ilu                 !          | 
      integer :: ulu                 !          |
      integer :: iwgen               !          |
      character (len=1) :: timest    !          |
     
      type uptake_parameters
       real :: water_dis = 10.        !               |the uptake distribution for water is hardwired
       real :: water_norm             !none           |water uptake normalization parameter 
       real :: n_norm                 !none           |nitrogen uptake normalization parameter 
       real :: p_norm                 !none           |phosphorus uptake normalization parameter
      end type uptake_parameters
      type (uptake_parameters)  :: uptake

      type irrigation_sources
        integer :: flag = 0   !0= don't irrigate, 1=irrigate
        integer, dimension(:), allocatable :: chan
        integer, dimension(:), allocatable :: res
        integer, dimension(:), allocatable :: pond
        integer, dimension(:), allocatable :: shal
        integer, dimension(:), allocatable :: deep
      end type irrigation_sources
      
      type topography
           character(len=13) :: name
           real :: elev = 0.         !!               |m             |elevation of HRU
           real :: slope = 0.        !!	hru_slp(:)    |m/m           |average slope steepness in HRU
           real :: slope_len = 0.    !! slsubbsn(:)   |m             |average slope length for erosion
           real :: dr_den = 0.       !!               |km/km2        |drainage density
           real :: lat_len = 0.      !! slsoil(:)     |m             |slope length for lateral subsurface flow
           real :: dis_stream = 0.   !! dis_stream(:) | m            |average distance to stream
           real :: dep_co = 1.       !!               |              |deposition coefficient
           integer :: field_db = 0   !!               |              |pointer to field.fld
           integer :: channel_db=0   !!               |              |pointer to channel.dat
      end type topography
      
      type field
           character(len=13) :: name = "rep field"
           real :: length = 0.2  !!            |km            |field length for wind erosion
           real :: wid = 0.2  !!               |km            |field width for wind erosion
           real :: ang = 60.  !!               |deg           |field angle for wind erosion
      end type field
      
      type hydrology
           character(len=13) :: name
           real :: lat_ttime = 0.   !! lat_ttime(:)  |none          |Exponential of the lateral flow travel time
           real :: lat_sed = 0.     !! lat_sed(:)    |g/L           |sediment concentration in lateral flow
           real :: canmx = 0.       !! canmx(:)      |mm H2O        |maximum canopy storage
           real :: esco = 0.        !! esco(:)       |none          |soil evaporation compensation factor
           real :: epco = 0.        !! epco(:)       |none          |plant water uptake compensation factor (0-1)
           real :: erorgn = 0.      !! erorgn(:)     |none          |organic N enrichment ratio, if left blank
                                    !!                              |the model will calculate for every event
           real :: erorgp = 0.      !! erorgp(:)     |none          |organic P enrichment ratio, if left blank
                                    !!                              |the model will calculate for every event
           real :: cn3_swf = 0.     !!               |none          |curve number adjustment factor - sw at cn3
           real :: biomix = 0.      !! biomix(:)     |none          |biological mixing efficiency.
                                    !!                              |Mixing of soil due to activity of earthworms
                                    !!                              |and other soil biota. Mixing is performed at
                                    !!                              |the end of every calendar year.
           real :: perco = 0.       !!               |0-1           |percolation coefficient - linear adjustment to daily perc
           real :: lat_orgn = 0.
           real :: lat_orgp = 0.
           real :: harg_pet  = .0023  
           real :: latq_co = 0.3    !!               |              |plant ET curve number coefficient
           real :: perco_lim = 1.   !!               |              |percolation coefficient-limits perc from bottom layer
      end type hydrology
      
      type snow_parameters
         character (len=16) :: name
         real :: falltmp = 0.     !deg C         |snowfall temp
         real :: melttmp = 0.     !deg C         |snow melt base temp 
         real :: meltmx = 0.      !mm/deg C/day  |Max melt rate for snow during year (June 21)
         real :: meltmn = 0.      !mm/deg C/day  |Min melt rate for snow during year (Dec 21)
         real :: timp             !none          |snow pack temp lag factor (0-1)
         real :: covmx = 0.       !mm H20        |Min snow water content
         real :: cov50 = 0.       !none          |frac of COVMX
         real :: init_mm = 0.     !mm H20        |initial snow water content at start of simulation
      end type snow_parameters
      
      type subsurface_drainage_parameters
        character(len=13) :: name = "default"
        real :: depth = 0.    !! |mm            |depth of drain tube from the soil surface
        real :: time = 0.     !! |hrs           |time to drain soil to field capacity
        real :: lag = 0.      !! |hours         |drain tile lag time
        real :: radius = 0.   !! |mm            |effective radius of drains
        real :: dist = 0.     !! |mm            |distance between two drain tubes or tiles
        real :: drain_co = 0. !! |mm/day        |drainage coefficient
        real :: pumpcap = 0.  !! |mm/hr         |pump capacity 
        real :: latksat = 0.  !! !na            |multiplication factor to determine lat sat hyd conductivity for profile
      end type subsurface_drainage_parameters
              
      type landuse
          character(len=15) :: name
          integer :: cn_lu = 0
          integer :: cons_prac = 0
          real :: usle_p = 0.           !! none     | USLE equation support practice (P) factor daily
          character (len=16) :: urb_ro  !! none     | urban runoff model
                                        !!          | "usgs_reg", simulate using USGS regression eqs
                                        !!          | "buildup_washoff", simulate using build up/wash off alg 
          integer ::  urb_lu = 0        !! none     | urban land type identification number
          real :: ovn = 0.05            !! none     | Manning's "n" value for overland flow
      end type landuse
      type (landuse), dimension (:), allocatable :: luse
      
      type soil_plant_initialize
        character(len=16) :: name = ""
        real :: sw_frac
        character(len=16) :: nutc = ""
        character(len=16) :: pestc = ""
        character(len=16) :: pathc = ""
        character(len=16) :: saltc = ""
        character(len=16) :: hmetc = ""
        integer :: nut = 1
        integer :: pest = 1
        integer :: path = 1
        integer :: salt = 1
        integer :: hmet = 1
      end type soil_plant_initialize
      type (soil_plant_initialize), dimension (:), allocatable :: sol_plt_ini
        
      type hru_databases
        character(len=13) :: name = ""
        integer :: topo = 1
        integer :: hyd = 1
        integer :: soil = 1
        integer :: land_use_mgt = 1
        integer :: soil_plant_init = 1
        integer :: surf_stor = 0
        integer :: snow = 1
        integer :: field = 0
      end type hru_databases
      
      type hru_databases_char
        character(len=25) :: name = ""
        character(len=25) :: topo = ""
        character(len=25) :: hyd = ""
        character(len=25) :: soil = ""
        character(len=25) :: land_use_mgt = ""
        character(len=25) :: soil_plant_init = ""
        character(len=25) :: surf_stor = ""
        character(len=25) :: snow = ""
        character(len=25) :: field = ""
      end type hru_databases_char
        
      type hru_parms_db
        real :: co2 = 350.
      end type hru_parms_db
      
      type hydrologic_response_unit_db
        character(len=13) :: name = "default"
        type (hru_databases) :: dbs
        type (hru_databases_char) :: dbsc
        type (hru_parms_db) :: parms
      end type hydrologic_response_unit_db
      type (hydrologic_response_unit_db), dimension(:),allocatable :: hru_db
      
      type land_use_mgt_variables
        real :: usle_p = 0.                 !! |none          |USLE equation comservation practice (P) factor
        real :: usle_ls = 0.                !! |none          |USLE equation length slope (LS) factor
        real :: usle_mult = 0.              !! |none          |product of USLE K,P,LS,exp(rock)
        real :: sdr_dep = 0.                !! |
        integer :: ldrain= 0.               !! |none          |soil layer where drainage tile is located
        real :: tile_ttime = 0.             !! |none          |Exponential of the tile flow travel time
        real :: vfsi = 0.                   !! |none          |initial SCS curve number II value
        real :: vfsratio = 0.               !! |none          |contouring USLE P factor
        real :: vfscon = 0.                 !! |none          |fraction of the total runoff from the entire field
        real :: vfsch = 0;                  !! |none          |fraction of flow entering the most concentrated 10% of the VFS.
                                            !!                     which is fully channelized
        integer :: ngrwat = 0
        real :: grwat_i = 0.                !! |none          |On/off Flag for waterway simulation
        real :: grwat_n = 0.                !! |none          |Mannings's n for grassed waterway
        real :: grwat_spcon = 0.            !! |none          |sediment transport coefficant defined by user
        real :: grwat_d = 0.                !! |m             |depth of Grassed waterway
        real :: grwat_w = 0.                !! |none          |Width of grass waterway
        real :: grwat_l = 0.                !! |km            |length of Grass Waterway
        real :: grwat_s = 0.                !! |m/m           |slope of grass waterway
        real :: bmp_flag = 0.  
        real :: bmp_sed = 0.                !! |%             | Sediment removal by BMP 
        real :: bmp_pp = 0.                 !! |%             | Particulate P removal by BMP
        real :: bmp_sp = 0.                 !! |%             | Soluble P removal by BMP
        real :: bmp_pn = 0.                 !! |%             | Particulate N removal by BMP 
        real :: bmp_sn = 0.                 !! |%             | Soluble N removal by BMP  
        real :: bmp_bac = 0.                !! |%             | Bacteria removal by BMP
      end type land_use_mgt_variables
     
      type hydrologic_response_unit
        character(len=13) :: name = ""
        integer :: obj_no
        real :: area_ha
        real :: km
        integer :: surf_stor                    !points to res() for surface storage
        type (hru_databases) :: dbs             !database pointers
        type (hru_databases_char) :: dbsc       !database pointers
        type (hru_parms_db) :: parms            !calibration parameters
        integer :: land_use_mgt
        character(len=16) :: land_use_mgt_c
        integer :: lum_group
        character(len=16) :: lum_group_c        !land use group for soft cal and output
        character(len=16) :: region
        integer :: plant_cov
        integer :: mgt_ops
        integer :: tiledrain = 0
        integer :: septic = 0
        integer :: fstrip = 0
        integer :: grassww = 0
        integer :: bmpuser = 0
        integer :: crop_reg = 0

        !! other data
        type (topography) :: topo
        type (field) :: field
        type (hydrology) :: hyd
        type (landuse) :: luse
        type (land_use_mgt_variables) :: lumv
        type (subsurface_drainage_parameters) :: sdr
        type (snow_parameters) :: sno
        integer :: cur_op = 1
        real :: sno_mm                          !mm H2O        |amount of water in snow on current day
        real :: water_fr
        real :: water_seep
        real :: water_evap
        integer :: ich_flood
      end type hydrologic_response_unit
      type (hydrologic_response_unit), dimension(:), allocatable, target :: hru
      type (hydrologic_response_unit), dimension(:), allocatable, target :: hru_init

      
      real :: precipday         !! mm   |daily precip for the hru
      real :: precip_eff        !! mm   |daily effective precip for runoff calculations = precipday + ls_overq + snomlt - canstor
                                !!      |precip_eff = precipday + ls_overq - snofall + snomlt - canstor
      real :: qday              !! mm   |surface runoff that reaches main channel during day in HRU                               
                                
!!    change per JGA 8/31/2011 gsm for output.mgt 
      real :: yield
      
!!    new/modified arrays for plant competition
      integer :: ipl, isol

      real :: strsa_av,strsn_av,strsp_av,strstmp_av
      real :: rto_no3,rto_solp,uno3d_tot,uapd_tot,sum_no3
      real :: sum_solp
      real, dimension (:), allocatable :: epmax,cvm_com,blai_com
      real, dimension (:), allocatable :: rsdco_plcom, translt
      real, dimension (:), allocatable :: uno3d,uapd
      real, dimension (:), allocatable :: par,htfac,un2,up2
      integer, dimension (:), allocatable :: iseptic
     
!! septic variables for output.std
      real :: peakr, sw_excess, albday
      real :: wt_shall
      real :: sq_rto
      real :: tloss, snomlt, snofall, fixn, qtile
      real :: latlyr                 !!mm            |lateral flow in soil layer for the day
      real :: inflpcp                !!mm            |amount of precipitation that infiltrates
      real :: fertn, sepday, bioday
      real :: sepcrk, sepcrktot, fertno3, fertnh3, fertorgn, fertsolp
      real :: fertorgp
      real :: fertp, grazn, grazp, sdti
      real :: voltot                 !!mm            |total volumne of cracks expressed as depth per area unit
      real :: volcrmin               !!mm            |minimum crack volume allowed in any soil layer
      real :: canev, usle, rcn
      real :: enratio
      real :: vpd
      real :: pet_day, ep_day
      real :: snoev
      real :: es_day, ls_overq, latqrunon, tilerunon
      real :: ep_max
      real :: bsprev
      real :: usle_ei
      real :: snocov1, snocov2, lyrtile

      real :: etday
      integer :: mo
      integer :: ihru             !!none          |HRU number
      integer :: nd_30
      integer :: mpst, mlyr
!  routing 5/3/2010 gsm per jga    
! date
      character(len=8) :: date

!! septic change added iseptic 1/28/09 gsm
      integer :: isep_ly
      real, dimension (:), allocatable :: percp    
      real, dimension (:), allocatable :: qstemm
!! septic changes added 1/28/09 gsm
      real, dimension (:), allocatable :: bio_bod, biom,rbiom
      real, dimension (:), allocatable :: fcoli, bz_perc, plqm
!! Septic system by Jaehak Jeong
      integer, dimension (:), allocatable :: i_sep
      integer, dimension (:), allocatable :: sep_tsincefail
      
 !!   change per JGA 9/8/2011 gsm for output.mgt 
      real, dimension (:), allocatable :: sol_sumno3, sol_sumsolp

!     Sediment parameters added by Balaji for the new routines

      real, dimension (:), allocatable :: sanyld,silyld,clayld,sagyld
      real, dimension (:), allocatable :: lagyld,grayld
      integer, dimension (:), allocatable :: itb
      
!!!!!! drains
      real, dimension (:), allocatable :: wnan
      real, dimension (:,:), allocatable :: uh

      real, dimension (:), allocatable :: phusw
      integer, dimension (:), allocatable :: yr_skip, isweep
      real :: sweepeff

      real, dimension (:), allocatable :: ranrns_hru
      integer, dimension (:), allocatable :: itill

      real, dimension (:), allocatable :: tc_gwat
      real, dimension (:), allocatable :: wfsh
      real, dimension (:), allocatable :: sed_con, orgn_con, orgp_con
      real, dimension (:), allocatable :: soln_con, solp_con
      real, dimension (:), allocatable :: filterw
      real, dimension (:), allocatable :: cn2
      real, dimension (:), allocatable :: smx
      real, dimension (:), allocatable :: cnday
      real, dimension (:), allocatable :: tmpav
      real, dimension (:), allocatable :: hru_ra
      real, dimension (:), allocatable :: tmx,tmn
      real, dimension (:), allocatable :: tconc,hru_rmx
      real, dimension (:), allocatable :: usle_cfac,usle_eifac
      real, dimension (:), allocatable :: t_ov
      real, dimension (:), allocatable :: u10,rhd
      real, dimension (:), allocatable :: canstor,ovrlnd

!    Drainmod tile equations  08/2006 
	  real, dimension (:), allocatable :: cumei,cumeira
	  real, dimension (:), allocatable :: cumrt, cumrai
      real, dimension (:), allocatable :: sstmaxd
	  real, dimension (:), allocatable :: stmaxd
!    Drainmod tile equations  08/2006
      real, dimension (:), allocatable :: surqsolp
      real, dimension (:), allocatable :: cklsp
      real, dimension (:), allocatable :: pplnt,snotmp
      real, dimension (:), allocatable :: brt

      real, dimension (:), allocatable :: twash,doxq
      real, dimension (:), allocatable :: percn
      real, dimension (:), allocatable :: cbodu,chl_a,qdr
      real, dimension (:), allocatable :: latno3,latq,nplnt
      real, dimension (:), allocatable :: tileno3
      real, dimension (:), allocatable :: sedminpa,sedminps,sedorgn
      real, dimension (:), allocatable :: sedorgp,sedyld,sepbtm
      real, dimension (:), allocatable :: surfq,surqno3
      real, dimension (:), allocatable :: phubase
      real, dimension (:), allocatable :: lai_yrmx,dormhr
      real, dimension (:,:), allocatable :: wrt
      real, dimension (:,:), allocatable :: bss,surf_bs  
      integer, dimension (:), allocatable :: swtrg
      !! burn
      integer, dimension (:), allocatable :: grz_days
      integer, dimension (:), allocatable :: igrz,ndeat

!!     gsm added for sdr (drainage) 7/24/08
      integer, dimension (:,:), allocatable :: mgt_ops

      real, dimension (:,:), allocatable :: hhqday
! additional reach variables , added by Ann van Griensven
! Modifications to Pesticide and Water routing routines by Balaji Narasimhan
!Additional buffer and filter strip variables Mike White

	real, dimension (:), allocatable :: ubnrunoff,ubntss
	real, dimension (:,:), allocatable :: ovrlnd_dt,hhsurfq	
	real, dimension (:,:,:), allocatable :: hhsurf_bs

!! subdaily erosion modeling by Jaehak Jeong
	real, dimension(:,:), allocatable:: hhsedy
	real, dimension(:), allocatable:: init_abstrc

      integer, dimension(:), allocatable :: tillage_switch
      real, dimension(:), allocatable :: tillage_depth
      integer, dimension(:), allocatable :: tillage_days
      real, dimension(:), allocatable :: tillage_factor

      end module hru_module