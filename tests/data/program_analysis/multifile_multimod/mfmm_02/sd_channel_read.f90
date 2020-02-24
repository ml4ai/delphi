      subroutine sd_channel_read
      
      use basin_module
      use input_file_module
      use maximum_data_module
      use channel_data_module
      use channel_velocity_module
      use ch_pesticide_module
      use sd_channel_module
      use hydrograph_module
      use constituent_mass_module
      use pesticide_data_module
      use pathogen_data_module
      use water_body_module

      implicit none
      
      character (len=80) :: titldum     !          |title of file
      character (len=80) :: header      !          |header of file
      integer :: eof                    !          |end of file
      integer :: imax                   !units     |description
      logical :: i_exist                !          |check to determine if file exists
      integer :: ichi                   !none      |counter
      integer :: isp_ini                !          |counter
      integer :: ics                    !none      |counter
      integer :: ich_ini                !none      |counter
      integer :: ipest_ini              !none      |counter
      integer :: ipest_db               !none      |counter
      integer :: ipath_ini              !none      |counter
      integer :: inut                   !none      |counter 
      integer :: ihydsed                !none      |counter
      integer :: ipest                  !none      |counter
      integer :: ipath                  !none      |counter
      integer :: iom_ini                !none      |counter
      integer :: idb                    !none      |counter
      integer :: i                      !none      |counter
      integer :: k                      !none      |counter
      real :: bedvol                    !m^3       |volume of river bed sediment
      integer :: icon, iob, ichdat
      eof = 0
      imax = 0
            
      !! allocate sd channel variables
      allocate (sd_ch(0:sp_ob%chandeg))
      allocate (sd_ch_vel(0:sp_ob%chandeg))
      allocate (chsd_d(0:sp_ob%chandeg))
      allocate (chsd_m(0:sp_ob%chandeg))
      allocate (chsd_y(0:sp_ob%chandeg))
      allocate (chsd_a(0:sp_ob%chandeg))
      allocate (ch_water(0:sp_ob%chandeg))
      allocate (ch_benthic(0:sp_ob%chandeg))
      allocate (ch_stor(0:sp_ob%chandeg))
      allocate (ch_stor_m(0:sp_ob%chandeg))
      allocate (ch_stor_y(0:sp_ob%chandeg))
      allocate (ch_stor_a(0:sp_ob%chandeg))
      allocate (ch_wat_d(0:sp_ob%chandeg))
      allocate (ch_wat_m(0:sp_ob%chandeg))
      allocate (ch_wat_y(0:sp_ob%chandeg))
      allocate (ch_wat_a(0:sp_ob%chandeg))
      allocate (ch_in_d(0:sp_ob%chandeg))
      allocate (ch_in_m(0:sp_ob%chandeg))
      allocate (ch_in_y(0:sp_ob%chandeg))
      allocate (ch_in_a(0:sp_ob%chandeg))
      allocate (ch_out_d(0:sp_ob%chandeg))
      allocate (ch_out_m(0:sp_ob%chandeg))
      allocate (ch_out_y(0:sp_ob%chandeg))
      allocate (ch_out_a(0:sp_ob%chandeg))
      allocate (ch_om_water_init(0:sp_ob%chandeg))
      allocate (chpst_d(0:sp_ob%chandeg))
      allocate (chpst_m(0:sp_ob%chandeg))
      allocate (chpst_y(0:sp_ob%chandeg))
      allocate (chpst_a(0:sp_ob%chandeg))
      
      if (cs_db%num_pests > 0) then
        allocate (chpst%pest(cs_db%num_pests))
        allocate (chpstz%pest(cs_db%num_pests))
        allocate (bchpst_d%pest(cs_db%num_pests))
        allocate (bchpst_m%pest(cs_db%num_pests))
        allocate (bchpst_y%pest(cs_db%num_pests))
        allocate (bchpst_a%pest(cs_db%num_pests))
        do ich = 1, sp_ob%chandeg
          allocate (sd_ch(ich)%aq_mix(cs_db%num_pests))
          allocate (chpst_d(ich)%pest(cs_db%num_pests))
          allocate (chpst_m(ich)%pest(cs_db%num_pests))
          allocate (chpst_y(ich)%pest(cs_db%num_pests))   
          allocate (chpst_a(ich)%pest(cs_db%num_pests))
          allocate (ch_water(ich)%pest(cs_db%num_pests))
          allocate (ch_benthic(ich)%pest(cs_db%num_pests))
        end do
      end if
            
      if (cs_db%num_paths > 0) then
        do ich = 1, sp_ob%chandeg
          allocate (ch_water(ich)%path(cs_db%num_paths))
          allocate (ch_benthic(ich)%path(cs_db%num_paths))
        end do
      end if
                  
      if (cs_db%num_metals > 0) then
        do ich = 1, sp_ob%chandeg
          allocate (ch_water(ich)%hmet(cs_db%num_metals))
          allocate (ch_benthic(ich)%hmet(cs_db%num_metals))
        end do
      end if
                  
      if (cs_db%num_salts > 0) then
        do ich = 1, sp_ob%chandeg
          allocate (ch_water(ich)%salt(cs_db%num_salts))
          allocate (ch_benthic(ich)%salt(cs_db%num_salts))
        end do
      end if

      inquire (file=in_cha%chan_ez, exist=i_exist)
      if (.not. i_exist .or. in_cha%chan_ez == "null") then
        allocate (sd_dat(0:0))
      else   
      do
       open (105,file=in_cha%chan_ez)
       read (105,*,iostat=eof) titldum
       if (eof < 0) exit
       read (105,*,iostat=eof) header
       if (eof < 0) exit
        do while (eof == 0)
          read (105,*,iostat=eof) i
          if (eof < 0) exit
          imax = Max(imax,i)
        end do
        
      db_mx%sdc_dat = imax

      allocate (sd_dat(0:imax))
      
      rewind (105)
      read (105,*,iostat=eof) titldum
      if (eof < 0) exit
      read (105,*,iostat=eof) header
      if (eof < 0) exit
      
      do ichi = 1, db_mx%sdc_dat
         read (105,*,iostat=eof) i
         if (eof < 0) exit
         backspace (105)
         read (105,*,iostat=eof) k, sd_dat(i)%name, sd_dat(i)%initc, sd_dat(i)%hydc, sd_dat(i)%sedc, &
            sd_dat(i)%nutc
         if (eof < 0) exit
         
        !! initialize orgaincs and minerals in water
        do isp_ini = 1, db_mx%ch_init
          if (sd_dat(ichi)%initc == ch_init(isp_ini)%name) then
            sd_dat(ichi)%init = isp_ini
            !! initial organic mineral
            do ics = 1, db_mx%om_water_init
              if (ch_init(isp_ini)%org_min == om_init_name(ics)) then
                sd_init(isp_ini)%org_min = ics
                exit
              end if
            end do
            !! initial pesticides
            do ics = 1, db_mx%pestw_ini
              if (ch_init(isp_ini)%pest == pest_init_name(ics)) then
                sd_init(isp_ini)%pest = ics
                exit
              end if
            end do
            !! initial pathogens
            do ics = 1, db_mx%pathw_ini
              if (ch_init(isp_ini)%path == path_init_name(ics)) then
                sd_init(isp_ini)%path = ics
                exit
              end if
            end do
!            !! initial heavy metals
!            do ics = 1, db_mx%hmetw_ini
!              if (ch_init(isp_ini)%hmetc == ch_hmet_init(ics)%name) then
!                sd_init(isp_ini)%hmet = ics
!                exit
!              end if
!            end do
             !! initial salts
             do ics = 1, db_mx%saltw_ini
               if (ch_init(isp_ini)%salt == salt_init_name(ics)) then
                 sd_init(isp_ini)%salt = ics
                 exit
               end if
             end do
          end if
        end do
        
          !! set hydraulic and sediment input data (all in one file now)
          do ihydsed = 1, db_mx%ch_lte
            if (sd_chd(ihydsed)%name == sd_dat(ichi)%hydc) then
              sd_dat(ichi)%hyd = ihydsed
              exit
            end if
          end do
          
          do inut = 1, db_mx%ch_nut
            if (ch_nut(inut)%name == sd_dat(ichi)%nutc) then
              sd_dat(ichi)%nut = inut
              exit
            end if
          end do   

      end do
      close (105)
      exit
      end do
      
      end if
     
      ! initialize organics-minerals in channel water and benthic from input data
      do ich = 1, sp_ob%chandeg
        iob = sp_ob1%chandeg + ich - 1
        ichdat = ob(iob)%props
        ich_ini = sd_dat(ichdat)%init
        iom_ini = sd_init(ich_ini)%org_min
        ch_stor(ich) = om_init_water(iom_ini)
        ch_om_water_init(ich) = ch_stor(ich)
      end do
      
      ! initialize pesticides in channel water and benthic from input data
      do ich = 1, sp_ob%chandeg
        iob = sp_ob1%chandeg + ich - 1
        ichdat = ob(iob)%props
        ich_ini = sd_dat(ichdat)%init
        ipest_ini = sd_init(ich_ini)%pest
        do ipest = 1, cs_db%num_pests
          ipest_db = cs_db%pest_num(ipest)
          ! mg = mg/kg * m3*1000. (kg=m3*1000.)
          ch_water(ich)%pest(ipest) = pest_water_ini(ipest_ini)%water(ipest) * ch_stor(ich)%flo * 1000.
          !! calculate volume of active river bed sediment layer - m3
          bedvol = sd_ch(ich)%chw *sd_ch(ich)%chl * 1000.* pestdb(ipest_ini)%ben_act_dep
          ch_benthic(ich)%pest(ipest) = pest_water_ini(ipest_ini)%benthic(ipest) * bedvol * 1000.   ! mg = mg/kg * m3*1000.
          !! calculate mixing velocity using molecular weight and porosity
          sd_ch(ich)%aq_mix(ipest) = pestdb(ipest_db)%mol_wt ** (-.6666) * (1. - sd_chd(ich)%bd / 2.65) * (69.35 / 365)
        end do
      end do
                  
      ! initialize pathogens in channel water and benthic from input data
      do ich = 1, sp_ob%chandeg
        iob = sp_ob1%chandeg + ich - 1
        ichdat = ob(iob)%props
        ich_ini = sd_dat(ichdat)%init
        ipath_ini = sd_init(ich_ini)%path
        do ipath = 1, cs_db%num_paths
          ch_water(ich)%path(ipath) = path_water_ini(ipest_ini)%water(ipath)
          ch_benthic(ich)%path(ipath) = path_water_ini(ipest_ini)%benthic(ipath)
        end do
      end do
      
      return    
      end subroutine sd_channel_read