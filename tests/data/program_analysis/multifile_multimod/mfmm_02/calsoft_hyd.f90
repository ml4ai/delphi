      subroutine calsoft_hyd

      use hru_module, only : cn2, hru, hru_init
      use soil_module
      use plant_module
      use hydrograph_module
      use ru_module
      use aquifer_module
      use channel_module
      use hru_lte_module
      use sd_channel_module
      use basin_module
      use maximum_data_module
      use calibration_data_module
      use conditional_module
      use reservoir_module
      use organic_mineral_mass_module
      use time_module
      
      implicit none
      
      integer :: iter_all      !none      |counter
      integer :: iterall       !none      |counter
      integer :: isim          !          |
      integer :: ireg          !none      |counter
      integer :: ilum          !none      |counter
      integer :: iihru         !none      |counter
      integer :: icn           !none      |counter
      integer :: ihru_s        !none      |counter
      integer :: iter_ind      !          |end of loop
      integer :: ietco         !none      |counter
      integer :: ik            !none      |counter
      integer :: nly           !          |end of loop
      integer :: iperco        !none      |counter
      real :: rmeas            !          |
      real :: denom            !          |
      real :: soft             !          |
      real :: diff             !          |
      real :: rto              !          |
      real :: chg_val          !          | 
      real :: dep_below_soil   !          |  
      real :: perc_ln_func

      ! calibrate hydrology
        ical_hyd = 0
        iter_all = 1
        iter_ind = 1

      ! first calibrate potential et
   
      do ireg = 1, db_mx%lsu_reg
        do ilum = 1, region(ireg)%nlum
          lscal(ireg)%lum(ilum)%petco = 0.
          if (lscal(ireg)%lum(ilum)%meas%pet > 0.) then
            rto = lscal(ireg)%lum(ilum)%meas%pet / lscal(ireg)%lum(ilum)%pet_aa
            lscal(ireg)%lum(ilum)%petco = rto
            !check all hru"s for proper lum
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for pet adjustment and run
                hru(iihru)%hyd%harg_pet = rto * hru(iihru)%hyd%harg_pet
                hru_init(iihru)%hyd%harg_pet = hru(iihru)%hyd%harg_pet
              end if
            end do
                       
            !! re-initialize all objects
            call re_initialize
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z

            ! pet adjustment
            cal_sim =  " pet adj "
            call time_control
          end if
        end do
      end do
      
      do iterall = 1, iter_all
          
        ! 1st cn2 adjustment
        isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%srr) / soft)
            if (diff > .02 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%cn < 1.e-6) then
            isim = 1
            
                lscal(ireg)%lum(ilum)%prm_prev = lscal(ireg)%lum(ilum)%prm
                lscal(ireg)%lum(ilum)%prev = lscal(ireg)%lum(ilum)%aa

                diff = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa - lscal(ireg)%lum(ilum)%aa%srr
                chg_val = diff / 15.     !assume 10 mm runoff for 1 cn
                lscal(ireg)%lum(ilum)%prm_prev%cn = lscal(ireg)%lum(ilum)%prm%cn
                lscal(ireg)%lum(ilum)%prm%cn = lscal(ireg)%lum(ilum)%prm%cn + chg_val
                lscal(ireg)%lum(ilum)%prev%srr = lscal(ireg)%lum(ilum)%aa%srr
                
                if (lscal(ireg)%lum(ilum)%prm%cn >= ls_prms(1)%pos) then
                  chg_val = ls_prms(1)%pos - lscal(ireg)%lum(ilum)%prm_prev%cn
                  lscal(ireg)%lum(ilum)%prm%cn = ls_prms(1)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%cn = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%cn <= ls_prms(1)%neg) then
                  chg_val = ls_prms(1)%neg - lscal(ireg)%lum(ilum)%prm_prev%cn
                  lscal(ireg)%lum(ilum)%prm%cn = ls_prms(1)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%cn = 1.
                end if

            !! re-initialize all objects
            call re_initialize

            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st surface runoff calibration and rerun
                cn2(iihru) = cn2(iihru) + chg_val
                cn2(iihru) = amin1 (cn2(iihru), ls_prms(1)%up)
                cn2(iihru) = Max (cn2(iihru), ls_prms(1)%lo)
                call curno (cn2(iihru), iihru)
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        ! 1st cn2 adjustment 
        if (isim > 0) then
          cal_sim = " first cn2 adj "
          call time_control
        end if

          ! adjust surface runoff using cn2
          do icn = 1, iter_ind
          isim = 0
          do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%srr) / soft)
            if (diff > .02 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%cn < 1.e-6) then
            isim = 1
            
                rmeas = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
                denom = lscal(ireg)%lum(ilum)%prev%srr - lscal(ireg)%lum(ilum)%aa%srr
                if (abs(denom) > 1.e-6) then
                  chg_val = - (lscal(ireg)%lum(ilum)%prm_prev%cn - lscal(ireg)%lum(ilum)%prm%cn)                  &
                    * (lscal(ireg)%lum(ilum)%aa%srr - rmeas) / denom
                else
                  chg_val = diff / 200.
                end if
                lscal(ireg)%lum(ilum)%prm_prev%cn = lscal(ireg)%lum(ilum)%prm%cn
                lscal(ireg)%lum(ilum)%prm%cn = lscal(ireg)%lum(ilum)%prm%cn + chg_val
                lscal(ireg)%lum(ilum)%prev%srr = lscal(ireg)%lum(ilum)%aa%srr
                                
                if (lscal(ireg)%lum(ilum)%prm%cn >= ls_prms(1)%pos) then
                  chg_val = ls_prms(1)%pos - lscal(ireg)%lum(ilum)%prm_prev%cn
                  lscal(ireg)%lum(ilum)%prm%cn = ls_prms(1)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%cn = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%cn <= ls_prms(1)%neg) then
                  chg_val = ls_prms(1)%neg - lscal(ireg)%lum(ilum)%prm_prev%cn
                  lscal(ireg)%lum(ilum)%prm%cn = ls_prms(1)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%cn = 1.
                end if
            
            !! re-initialize all objects
            call re_initialize

            !check all hru"s for proper lum
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st surface runoff calibration and rerun
                cn2(iihru) = cn2(iihru) + chg_val
                cn2(iihru) = amin1 (cn2(iihru), ls_prms(1)%up)
                cn2(iihru) = Max (cn2(iihru), ls_prms(1)%lo)
                call curno (cn2(iihru), iihru)
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
            !else
            !lscal(ireg)%lum(ilum)%prm_lim%etco = 1.
            end if
          end do
        end do
          
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        ! cn2 adjustment
        if (isim > 0) then
          cal_sim =  " cn2 adj "
          call time_control
        end if
        end do      ! icn
          
        ! 1st esco adjustment
        isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%etr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%etr) / soft)
            if (diff > .01 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%etco < 1.e-6) then
            isim = 1
            
                lscal(ireg)%lum(ilum)%prm_prev = lscal(ireg)%lum(ilum)%prm
                lscal(ireg)%lum(ilum)%prev = lscal(ireg)%lum(ilum)%aa

                diff = lscal(ireg)%lum(ilum)%meas%etr * lscal(ireg)%lum(ilum)%precip_aa - lscal(ireg)%lum(ilum)%aa%etr
                chg_val = - diff / 250.     ! increment etco .4 for every 100 mm difference
                lscal(ireg)%lum(ilum)%prm_prev%etco = lscal(ireg)%lum(ilum)%prm%etco
                lscal(ireg)%lum(ilum)%prm%etco = lscal(ireg)%lum(ilum)%prm%etco + chg_val
                lscal(ireg)%lum(ilum)%prev%etr = lscal(ireg)%lum(ilum)%aa%etr
                
                if (lscal(ireg)%lum(ilum)%prm%etco >= ls_prms(2)%pos) then
                  chg_val = ls_prms(2)%pos - lscal(ireg)%lum(ilum)%prm_prev%etco
                  lscal(ireg)%lum(ilum)%prm%etco = ls_prms(2)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%etco = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%etco <= ls_prms(2)%neg) then
                  chg_val = lscal(ireg)%lum(ilum)%prm_prev%etco + ls_prms(2)%neg
                  lscal(ireg)%lum(ilum)%prm%etco = ls_prms(2)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%etco = 1.
                end if
                           
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st et calibration
                hru(iihru)%hyd%esco = hru(iihru)%hyd%esco - chg_val
                hru(iihru)%hyd%esco = amin1 (hru(iihru)%hyd%esco, ls_prms(2)%up)
                hru(iihru)%hyd%esco = Max (hru(iihru)%hyd%esco, ls_prms(2)%lo)
                hru_init(iihru)%hyd%esco = hru(iihru)%hyd%esco
                !hru(iihru)%hyd%epco = hru(iihru)%hyd%epco + chg_val
                !hru(iihru)%hyd%epco = amin1 (hru(iihru)%hyd%epco, ls_prms(2)%up)
                !hru(iihru)%hyd%epco = Max (hru(iihru)%hyd%epco, ls_prms(2)%lo)
                !hru_init(iihru)%hyd%epco = hru(iihru)%hyd%epco
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! 1st esco adjustment 
        if (isim > 0) then
          cal_sim =  " first esco adj "
          call time_control
        end if
        
        ! adjust et using esco
        do ietco = 1, iter_ind
          isim = 0
          do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            !check all hru"s for proper lum
            soft = lscal(ireg)%lum(ilum)%meas%etr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%etr) / soft)
            if (diff > .01 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%etco < 1.e-6) then
            isim = 1

                rmeas = lscal(ireg)%lum(ilum)%meas%etr * lscal(ireg)%lum(ilum)%precip_aa
                denom = lscal(ireg)%lum(ilum)%prev%etr - lscal(ireg)%lum(ilum)%aa%etr
                if (abs(denom) > 1.e-6) then
                  chg_val = - (lscal(ireg)%lum(ilum)%prm_prev%etco - lscal(ireg)%lum(ilum)%prm%etco)                  &
                    * (lscal(ireg)%lum(ilum)%aa%etr - rmeas) / denom
                else
                  chg_val = diff / 200.
                end if
                lscal(ireg)%lum(ilum)%prm_prev%etco = lscal(ireg)%lum(ilum)%prm%etco
                lscal(ireg)%lum(ilum)%prm%etco = lscal(ireg)%lum(ilum)%prm%etco + chg_val
                lscal(ireg)%lum(ilum)%prev%etr = lscal(ireg)%lum(ilum)%aa%etr
                      
                if (lscal(ireg)%lum(ilum)%prm%etco >= ls_prms(2)%pos) then
                  chg_val = ls_prms(2)%pos - lscal(ireg)%lum(ilum)%prm_prev%etco
                  lscal(ireg)%lum(ilum)%prm%etco = ls_prms(2)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%etco = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%etco <= ls_prms(2)%neg) then
                  chg_val = ls_prms(2)%neg - lscal(ireg)%lum(ilum)%prm_prev%etco
                  lscal(ireg)%lum(ilum)%prm%etco = ls_prms(2)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%etco = 1.
                end if
                
            do ihru_s = 1, region(ireg)%num_tot
                iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for et calibration
                hru(iihru)%hyd%esco = hru(iihru)%hyd%esco - chg_val
                hru(iihru)%hyd%esco = amin1 (hru(iihru)%hyd%esco, ls_prms(2)%up)
                hru(iihru)%hyd%esco = Max (hru(iihru)%hyd%esco, ls_prms(2)%lo)
                hru_init(iihru)%hyd%esco = hru(iihru)%hyd%esco
                !hru(iihru)%hyd%epco = hru(iihru)%hyd%epco + chg_val
                !hru(iihru)%hyd%epco = amin1 (hru(iihru)%hyd%epco, ls_prms(2)%up)
                !hru(iihru)%hyd%epco = Max (hru(iihru)%hyd%epco, ls_prms(2)%lo)
                !hru_init(iihru)%hyd%epco = hru(iihru)%hyd%epco
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
          end do

          !! re-initialize all objects
          call re_initialize

          !zero plant calibration data in case plants are calibrated
          do ireg = 1, db_mx%plcal_reg
            do ilum = 1, plcal(ireg)%lum_num
              plcal(ireg)%lum(ilum)%nbyr = 0
              plcal(ireg)%lum(ilum)%precip_aa = 0.
              plcal(ireg)%lum(ilum)%aa = plcal_z
            end do
          end do
          
          !! re-initialize all objects
          call re_initialize

          ! et adjustment 
          if (isim > 0) then
            cal_sim =  " esco adj "
            call time_control
          end if
        
        end do      ! iesco

        ! 1st perco for percolation
        isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            !check all hru"s for proper lum
            soft = lscal(ireg)%lum(ilum)%meas%pcr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%pcr) / soft)
            if (diff > .05 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%perco < 1.e-6) then
            isim = 1
            
                !set parms for 1st perco calibration
                lscal(ireg)%lum(ilum)%prm_prev = lscal(ireg)%lum(ilum)%prm
                lscal(ireg)%lum(ilum)%prev = lscal(ireg)%lum(ilum)%aa

                !chg_val = (soft - lscal(ireg)%lum(ilum)%aa%pcr) / lscal(ireg)%lum(ilum)%aa%pcr   ! assume perco is linear
                chg_val = 0.001 * (soft - lscal(ireg)%lum(ilum)%aa%pcr)
                lscal(ireg)%lum(ilum)%prm_prev%perco = lscal(ireg)%lum(ilum)%prm%perco 
                lscal(ireg)%lum(ilum)%prm%perco = lscal(ireg)%lum(ilum)%prm%perco + chg_val
                lscal(ireg)%lum(ilum)%prev%pcr = lscal(ireg)%lum(ilum)%aa%pcr

                if (lscal(ireg)%lum(ilum)%prm%perco >= ls_prms(8)%pos) then
                  lscal(ireg)%lum(ilum)%prm%perco = ls_prms(8)%pos
                  chg_val = ls_prms(8)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%perco <= ls_prms(8)%neg) then
                  lscal(ireg)%lum(ilum)%prm%perco = ls_prms(8)%neg
                  chg_val = ls_prms(8)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
                end if
                
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !! don't change for tile  *********************Mike
                if (hru(iihru)%tiledrain == 0) then
                hru(iihru)%hyd%perco = hru(iihru)%hyd%perco + chg_val
                hru(iihru)%hyd%perco = amin1 (hru(iihru)%hyd%perco, ls_prms(8)%up)
                hru(iihru)%hyd%perco = Max (hru(iihru)%hyd%perco, ls_prms(8)%lo)
                hru_init(iihru)%hyd%perco = hru(iihru)%hyd%perco
                if (hru(iihru)%hyd%perco > 1.e-9) then
                  perc_ln_func = 1.0052 * log(-log(hru(iihru)%hyd%perco - 1.e-6)) + 5.6862
                  hru(iihru)%hyd%perco_lim = exp(-perc_ln_func)
                  hru(iihru)%hyd%perco_lim = amin1 (1., hru(iihru)%hyd%perco_lim)
                else
                  hru(iihru)%hyd%perco_lim = 0.
                end if
                hru_init(iihru)%hyd%perco_lim = hru(iihru)%hyd%perco_lim
                end if
              end if
            end do
            
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
            else
            lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
            end if
            end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! 1st perco adjustment 
        if (isim > 0) then
          cal_sim =  " first perco adj "
          call time_control
        end if
  
          ! adjust percolation using perco
          do iperco = 1, 2  !iter_ind
          isim = 0
          do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%pcr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%pcr) / soft)
            if (diff > .05 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%perco < 1.e-6) then
            isim = 1

                rmeas = lscal(ireg)%lum(ilum)%meas%pcr * lscal(ireg)%lum(ilum)%precip_aa
                denom = lscal(ireg)%lum(ilum)%prev%pcr - lscal(ireg)%lum(ilum)%aa%pcr
                if (abs(denom) > 2.) then
                  chg_val = - (lscal(ireg)%lum(ilum)%prm_prev%perco - lscal(ireg)%lum(ilum)%prm%perco)                  &
                    * (lscal(ireg)%lum(ilum)%aa%pcr - rmeas) / denom
                else
                  chg_val = 0.
                end if
                lscal(ireg)%lum(ilum)%prm_prev%perco = lscal(ireg)%lum(ilum)%prm%perco 
                lscal(ireg)%lum(ilum)%prm%perco = lscal(ireg)%lum(ilum)%prm%perco + chg_val
                lscal(ireg)%lum(ilum)%prev%pcr = lscal(ireg)%lum(ilum)%aa%pcr

                if (lscal(ireg)%lum(ilum)%prm%perco >= ls_prms(8)%pos) then
                  lscal(ireg)%lum(ilum)%prm%perco = ls_prms(8)%pos
                  chg_val = ls_prms(8)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%perco <= ls_prms(8)%neg) then
                  lscal(ireg)%lum(ilum)%prm%perco = ls_prms(8)%neg
                  chg_val = ls_prms(8)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
                end if
                
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st perco calibration
                !! don't change for tile  *********************Mike
                if (hru(iihru)%tiledrain == 0) then
                hru(iihru)%hyd%perco = hru(iihru)%hyd%perco + chg_val
                hru(iihru)%hyd%perco = amin1 (hru(iihru)%hyd%perco, ls_prms(8)%up)
                hru(iihru)%hyd%perco = Max (hru(iihru)%hyd%perco, ls_prms(8)%lo)
                hru_init(iihru)%hyd%perco = hru(iihru)%hyd%perco
                if (hru(iihru)%hyd%perco > 1.e-9) then
                  perc_ln_func = 1.0052 * log(-log(hru(iihru)%hyd%perco - 1.e-6)) + 5.6862
                  hru(iihru)%hyd%perco_lim = exp(-perc_ln_func)
                  hru(iihru)%hyd%perco_lim = amin1 (1., hru(iihru)%hyd%perco_lim)
                else
                  hru(iihru)%hyd%perco_lim = 0.
                end if
                hru_init(iihru)%hyd%perco_lim = hru(iihru)%hyd%perco_lim
                end if
              end if
            end do
            
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
            else
            lscal(ireg)%lum(ilum)%prm_lim%perco = 1.
            end if
          end do
          end do
          
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! perco adjustment 
        if (isim > 0) then
          cal_sim =  " perco adj "
          call time_control
        end if
        
        end do      ! iperco  

        ! 1st lat_len adjustment for lateral soil flow
        isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            !check all hru"s for proper lum
            soft = lscal(ireg)%lum(ilum)%meas%lfr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%lfr) / soft)
            if (diff > .1 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%lat_len < 1.e-6) then
            isim = 1
            
                lscal(ireg)%lum(ilum)%prm_prev = lscal(ireg)%lum(ilum)%prm
                lscal(ireg)%lum(ilum)%prev = lscal(ireg)%lum(ilum)%aa

                diff = lscal(ireg)%lum(ilum)%meas%lfr * lscal(ireg)%lum(ilum)%precip_aa - lscal(ireg)%lum(ilum)%aa%lfr
                chg_val =  0.005 * diff        ! increment lst_len by 2 m for 1 mm difference
                lscal(ireg)%lum(ilum)%prm_prev%lat_len = lscal(ireg)%lum(ilum)%prm%lat_len
                lscal(ireg)%lum(ilum)%prm%lat_len = lscal(ireg)%lum(ilum)%prm%lat_len + chg_val
                lscal(ireg)%lum(ilum)%prev%lfr = lscal(ireg)%lum(ilum)%aa%lfr
                           
                if (lscal(ireg)%lum(ilum)%prm%lat_len >= ls_prms(3)%pos) then
                  lscal(ireg)%lum(ilum)%prm%lat_len = ls_prms(3)%pos
                  chg_val = ls_prms(3)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%lat_len = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%lat_len <= ls_prms(3)%neg) then
                  lscal(ireg)%lum(ilum)%prm%lat_len = ls_prms(3)%neg
                  chg_val = ls_prms(3)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%lat_len = 1.
                end if
                
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st perco calibration
                hru(iihru)%hyd%latq_co = hru(iihru)%hyd%latq_co + chg_val
                hru(iihru)%hyd%latq_co = amin1 (hru(iihru)%hyd%latq_co, ls_prms(3)%up)
                hru(iihru)%hyd%latq_co = Max (hru(iihru)%hyd%latq_co, ls_prms(3)%lo)
                hru_init(iihru)%hyd%latq_co = hru(iihru)%hyd%latq_co
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! 1st lat_len adjustment 
        if (isim > 0) then
          cal_sim =  " first lat_len adj "
          call time_control
        end if

        ! adjust lat_len adjustment for lateral soil flow
        do ik = 1, iter_ind
          isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%lfr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%lfr) / soft)
            if (diff > .05 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%lat_len < 1.e-6) then
            isim = 1
            
                rmeas = lscal(ireg)%lum(ilum)%meas%lfr * lscal(ireg)%lum(ilum)%precip_aa
                denom = lscal(ireg)%lum(ilum)%prev%lfr - lscal(ireg)%lum(ilum)%aa%lfr
                if (abs(denom) > 1.e-6) then
                  chg_val = - (lscal(ireg)%lum(ilum)%prm_prev%lat_len - lscal(ireg)%lum(ilum)%prm%lat_len)                  &
                    * (lscal(ireg)%lum(ilum)%aa%lfr - rmeas) / denom
                else
                  chg_val = - diff    ! 1 m for each mm
                end if
                lscal(ireg)%lum(ilum)%prm_prev%lat_len = lscal(ireg)%lum(ilum)%prm%lat_len
                lscal(ireg)%lum(ilum)%prm%lat_len = lscal(ireg)%lum(ilum)%prm%lat_len + chg_val
                lscal(ireg)%lum(ilum)%prev%lfr = lscal(ireg)%lum(ilum)%aa%lfr
                           
                if (lscal(ireg)%lum(ilum)%prm%lat_len >= ls_prms(3)%pos) then
                  lscal(ireg)%lum(ilum)%prm%lat_len = ls_prms(3)%pos
                  chg_val = ls_prms(3)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%lat_len = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%lat_len <= ls_prms(3)%neg) then
                  lscal(ireg)%lum(ilum)%prm%lat_len = ls_prms(3)%neg
                  chg_val = ls_prms(3)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%lat_len = 1.
                end if
                
            !check all hru"s for proper lum
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st perco calibration
                hru(iihru)%hyd%latq_co = hru(iihru)%hyd%latq_co + chg_val
                hru(iihru)%hyd%latq_co = amin1 (hru(iihru)%hyd%latq_co, ls_prms(3)%up)
                hru(iihru)%hyd%latq_co = Max (hru(iihru)%hyd%latq_co, ls_prms(3)%lo)
                hru_init(iihru)%hyd%latq_co = hru(iihru)%hyd%latq_co
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! lat_len adjustment for lateral soil flow
        if (isim > 0) then
          cal_sim =  " lat_len adj "
          call time_control
        end if
        end do  
                  
        ! 1st cn3_swf adjustment
        isim = 0
        do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%srr) / soft)
            if (diff > .02 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%cn3_swf < 1.e-6) then
            isim = 1
            
                lscal(ireg)%lum(ilum)%prm_prev = lscal(ireg)%lum(ilum)%prm
                lscal(ireg)%lum(ilum)%prev = lscal(ireg)%lum(ilum)%aa

                diff = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa - lscal(ireg)%lum(ilum)%aa%srr
                !chg_val = - diff / 300.     !assume 10 mm runoff for .3 cn3_swf
                chg_val = - diff / 700.     !assume 10 mm runoff for .7 cn3_swf
                lscal(ireg)%lum(ilum)%prm_prev%cn3_swf = lscal(ireg)%lum(ilum)%prm%cn3_swf
                lscal(ireg)%lum(ilum)%prm%cn3_swf = lscal(ireg)%lum(ilum)%prm%cn3_swf + chg_val
                lscal(ireg)%lum(ilum)%prev%srr = lscal(ireg)%lum(ilum)%aa%srr
                
                if (lscal(ireg)%lum(ilum)%prm%cn3_swf >= ls_prms(10)%pos) then
                  chg_val = ls_prms(10)%pos - lscal(ireg)%lum(ilum)%prm_prev%cn3_swf
                  lscal(ireg)%lum(ilum)%prm%cn3_swf = ls_prms(10)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%cn3_swf = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%cn3_swf <= ls_prms(10)%neg) then
                  chg_val = ls_prms(10)%neg - lscal(ireg)%lum(ilum)%prm_prev%cn3_swf
                  lscal(ireg)%lum(ilum)%prm%cn3_swf = ls_prms(10)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%cn3_swf = 1.
                end if

            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st perco calibration
                !! don't change for tile  *********************Mike
                if (hru(iihru)%tiledrain == 0) then
                hru(iihru)%hyd%cn3_swf = hru(iihru)%hyd%cn3_swf + chg_val
                hru(iihru)%hyd%cn3_swf = amin1 (hru(iihru)%hyd%cn3_swf, ls_prms(10)%up)
                hru(iihru)%hyd%cn3_swf = Max (hru(iihru)%hyd%cn3_swf, ls_prms(10)%lo)
                hru_init(iihru)%hyd%cn3_swf = hru(iihru)%hyd%cn3_swf
                call curno (cn2(iihru), iihru)
                end if
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
          end if
          end do
        end do
        
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! 1st cn3_swf adjustment 
        if (isim > 0) then
          cal_sim =  " first cn3_swf adj "
          call time_control
        end if

          ! adjust surface runoff using cn3_swf
          do icn = 1, 2 !iter_ind
          isim = 0
          do ireg = 1, db_mx%lsu_reg
          do ilum = 1, region(ireg)%nlum
            soft = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - lscal(ireg)%lum(ilum)%aa%srr) / soft)
            if (diff > .02 .and. lscal(ireg)%lum(ilum)%ha > 1.e-6 .and. lscal(ireg)%lum(ilum)%prm_lim%cn3_swf < 1.e-6) then
            isim = 1
            
                rmeas = lscal(ireg)%lum(ilum)%meas%srr * lscal(ireg)%lum(ilum)%precip_aa
                denom = lscal(ireg)%lum(ilum)%prev%srr - lscal(ireg)%lum(ilum)%aa%srr
                if (abs(denom) > 1.e-6) then
                  chg_val = - (lscal(ireg)%lum(ilum)%prm_prev%cn3_swf - lscal(ireg)%lum(ilum)%prm%cn3_swf)                  &
                    * (lscal(ireg)%lum(ilum)%aa%srr - rmeas) / denom
                else
                  chg_val = - diff / 300.
                end if
                lscal(ireg)%lum(ilum)%prm_prev%cn3_swf = lscal(ireg)%lum(ilum)%prm%cn3_swf
                lscal(ireg)%lum(ilum)%prm%cn3_swf = lscal(ireg)%lum(ilum)%prm%cn3_swf + chg_val
                lscal(ireg)%lum(ilum)%prev%srr = lscal(ireg)%lum(ilum)%aa%srr
                                
                if (lscal(ireg)%lum(ilum)%prm%cn3_swf >= ls_prms(10)%pos) then
                  chg_val = ls_prms(10)%pos - lscal(ireg)%lum(ilum)%prm_prev%cn3_swf
                  lscal(ireg)%lum(ilum)%prm%cn3_swf = ls_prms(10)%pos
                  lscal(ireg)%lum(ilum)%prm_lim%cn3_swf = 1.
                end if
                if (lscal(ireg)%lum(ilum)%prm%cn3_swf <= ls_prms(10)%neg) then
                  chg_val = ls_prms(10)%neg - lscal(ireg)%lum(ilum)%prm_prev%cn3_swf
                  lscal(ireg)%lum(ilum)%prm%cn3_swf = ls_prms(10)%neg
                  lscal(ireg)%lum(ilum)%prm_lim%cn3_swf = 1.
                end if
            
            !check all hru"s for proper lum
            do ihru_s = 1, region(ireg)%num_tot
              iihru = region(ireg)%num(ihru_s)
              if (lscal(ireg)%lum(ilum)%meas%name == hru(iihru)%lum_group_c .or. lscal(ireg)%lum(ilum)%meas%name == "basin") then
                !set parms for 1st perco calibration
                !! don't change for tile  *********************Mike
                if (hru(iihru)%tiledrain == 0) then
                hru(iihru)%hyd%cn3_swf = hru(iihru)%hyd%cn3_swf + chg_val
                hru(iihru)%hyd%cn3_swf = amin1 (hru(iihru)%hyd%cn3_swf, ls_prms(10)%up)
                hru(iihru)%hyd%cn3_swf = Max (hru(iihru)%hyd%cn3_swf, ls_prms(10)%lo)
                hru_init(iihru)%hyd%cn3_swf = hru(iihru)%hyd%cn3_swf
                call curno (cn2(iihru), iihru)
                end if
              end if
            end do
            lscal(ireg)%lum(ilum)%nbyr = 0
            lscal(ireg)%lum(ilum)%precip_aa = 0.
            lscal(ireg)%lum(ilum)%aa = lscal_z
            end if
          end do
        end do
          
        !zero plant calibration data in case plants are calibrated
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end do
        end do
        
        !! re-initialize all objects
        call re_initialize

        ! cn3_swf adjustment
        if (isim > 0) then
          cal_sim =  " cn3_swf adj "
          pco%wb_hru%a = "y"
          call time_control
        end if
        end do      ! icn
          
      end do    ! iter_all loop
        
      cal_codes%hyd_hru = "n"
      
	  return
      end subroutine calsoft_hyd