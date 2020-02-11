      module hydrology_data_module
    
      implicit none
     
      type hydrology_db
         character(len=16) :: name
         real :: lat_ttime = 0.     !! lat_ttime(:)  |none          |Exponential of the lateral flow travel time
         real :: lat_sed = 0.       !! lat_sed(:)    |g/L           |sediment concentration in lateral flow
         real :: canmx = 0.         !! canmx(:)      |mm H2O        |maximum canopy storage
         real :: esco = 0.          !! esco(:)       |none          |soil evaporation compensation factor (0-1)
         real :: epco = 0.          !! epco(:)       |none          |plant water uptake compensation factor (0-1)
         real :: erorgn = 0.        !! erorgn(:)     |none          |organic N enrichment ratio, if left blank
                                    !!                              |the model will calculate for every event
         real :: erorgp = 0.        !! erorgp(:)     |none          |organic P enrichment ratio, if left blank
                                    !!                              |the model will calculate for every event
         real :: cn3_swf = 0.       !!               |none          |soil water at cn3 - 0=fc; .99=near saturation
         real :: biomix = 0.        !! biomix(:)     |none          |biological mixing efficiency.
                                    !!                              |Mixing of soil due to activity of earthworms
                                    !!                              |and other soil biota. Mixing is performed at
                                    !!                              |the end of every calendar year.
         real :: perco = 0.         !!               |0-1           |percolation coefficient - linear adjustment to daily perc
         real :: lat_orgn = 0.      !!               |ppm           |organic N concentration in lateral flow
         real :: lat_orgp = 0.      !!               |ppm           |organic P concentration in lateral flow
         real :: harg_pet  = .0023  !!            |              |coefficient related to radiation used in 
                                    !!                              | Hargreaves equation
         real :: latq_co = 0.3      !!               |              |plant ET curve number coefficient
       end type hydrology_db
        type (hydrology_db), dimension (:), allocatable :: hyd_db
        
     type snow_database
         character (len=16) :: name
         real :: falltmp = 0.     !deg C         |snowfall temp
         real :: melttmp = 0.     !deg C         |snow melt base temp 
         real :: meltmx = 0.      !mm/deg C/day  |Max melt rate for snow during year (June 21)
         real :: meltmn = 0.      !mm/deg C/day  |Min melt rate for snow during year (Dec 21)
         real :: timp             !none          |snow pack temp lag factor (0-1)
         real :: covmx = 0.       !mm H20        |Min snow water content
         real :: cov50 = 0.       !none          |frac of COVMX
         real :: init_mm = 0.     !mm H20        |initial snow water content at start of simulation
      end type snow_database
      type (snow_database), dimension (:), allocatable :: snodb
            
      end module hydrology_data_module 