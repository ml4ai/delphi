      subroutine calsoft_plant

      use hru_module, only : ihru 
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
      
      implicit none
      
      integer :: iter_all      !          |end of loop
      integer :: iterall       !none      |counter
      integer :: isim          !          |
      integer :: ireg          !none      |counter
      integer :: ilum          !none      |counter
      integer :: iihru         !none      |counter
      integer :: ihru_s        !none      |counter
      integer :: iter_ind      !          !end of loop
      integer :: ist           !          |
      real :: rmeas            !          |
      real :: denom            !          |
      real :: soft             !          |
      real :: diff             !          |
      real :: chg_val          !          | 
      
      

      !calibrate hydrology
        ical_hyd = 0
        iter_all = 1
        iter_ind = 1
        
      do iterall = 1, iter_all
        ! 1st plant stress adjustment
        isim = 0
        do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            soft = plcal(ireg)%lum(ilum)%meas%yield
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - plcal(ireg)%lum(ilum)%aa%yield) / soft)
            if (diff > .02 .and. plcal(ireg)%lum(ilum)%ha > 1.e-6 .and. plcal(ireg)%lum(ilum)%prm_lim%stress < 1.e-6) then
            isim = 1
            
                plcal(ireg)%lum(ilum)%prm_prev = plcal(ireg)%lum(ilum)%prm
                plcal(ireg)%lum(ilum)%prev = plcal(ireg)%lum(ilum)%aa

                diff = plcal(ireg)%lum(ilum)%meas%yield - plcal(ireg)%lum(ilum)%aa%yield
                chg_val = diff / 10.     !assume 1 t/ha for .1 stress??
                plcal(ireg)%lum(ilum)%prm_prev%stress = plcal(ireg)%lum(ilum)%prm%stress
                plcal(ireg)%lum(ilum)%prm%stress = plcal(ireg)%lum(ilum)%prm%stress + chg_val
                plcal(ireg)%lum(ilum)%prev%yield = plcal(ireg)%lum(ilum)%aa%yield
                
                if (plcal(ireg)%lum(ilum)%prm%stress >= pl_prms(1)%pos) then
                  chg_val = pl_prms(1)%pos - plcal(ireg)%lum(ilum)%prm_prev%stress
                  plcal(ireg)%lum(ilum)%prm%stress = pl_prms(1)%pos
                  plcal(ireg)%lum(ilum)%prm_lim%stress = 1.
                end if
                if (plcal(ireg)%lum(ilum)%prm%stress <= pl_prms(1)%neg) then
                  chg_val = pl_prms(1)%neg - plcal(ireg)%lum(ilum)%prm_prev%stress
                  plcal(ireg)%lum(ilum)%prm%stress = pl_prms(1)%neg
                  plcal(ireg)%lum(ilum)%prm_lim%stress = 1.
                end if

                exit    !!  start debug
            do ihru_s = 1, plcal(ireg)%num_tot
              iihru = plcal(ireg)%num(ihru_s)
              if (plcal(ireg)%lum(ilum)%meas%name == hlt(ihru)%plant) then
                !set parms for 1st surface runoff calibration and rerun
                hlt(iihru)= hlt_init(iihru)
                hlt(iihru)%stress = hlt(iihru)%stress + chg_val
                hlt_init(iihru) = hlt(iihru)
              end if
            end do
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
          end if
          end do
        end do
        ! 1st plant stress adjustment 
        if (isim > 0) then
          write (4304,*) " first plant stress adj "
          call time_control
        end if

          ! adjust plant growth using stress parameter
          do ist = 1, iter_ind
          isim = 0
          do ireg = 1, db_mx%plcal_reg
          do ilum = 1, plcal(ireg)%lum_num
            soft = plcal(ireg)%lum(ilum)%meas%yield
            diff = 0.
            if (soft > 1.e-6) diff = abs((soft - plcal(ireg)%lum(ilum)%aa%yield) / soft)
            if (diff > .02 .and. plcal(ireg)%lum(ilum)%ha > 1.e-6 .and. plcal(ireg)%lum(ilum)%prm_lim%stress < 1.e-6) then
            isim = 1
            
                rmeas = plcal(ireg)%lum(ilum)%meas%yield
                denom = plcal(ireg)%lum(ilum)%prev%yield - plcal(ireg)%lum(ilum)%aa%yield
                if (denom > 1.e-6) then
                  chg_val = - (plcal(ireg)%lum(ilum)%prm_prev%stress - plcal(ireg)%lum(ilum)%prm%stress) *                &
                              (plcal(ireg)%lum(ilum)%aa%yield - rmeas) / denom
                else
                  chg_val = diff / 200.
                end if
                plcal(ireg)%lum(ilum)%prm_prev%stress = plcal(ireg)%lum(ilum)%prm%stress
                plcal(ireg)%lum(ilum)%prm%stress = plcal(ireg)%lum(ilum)%prm%stress + chg_val
                plcal(ireg)%lum(ilum)%prev%yield = plcal(ireg)%lum(ilum)%aa%yield
                                
                if (plcal(ireg)%lum(ilum)%prm%stress >= pl_prms(1)%pos) then
                  chg_val = pl_prms(1)%pos - plcal(ireg)%lum(ilum)%prm_prev%stress
                  plcal(ireg)%lum(ilum)%prm%stress = pl_prms(1)%pos
                  plcal(ireg)%lum(ilum)%prm_lim%stress = 1.
                end if
                if (plcal(ireg)%lum(ilum)%prm%stress <= pl_prms(1)%neg) then
                  chg_val = plcal(ireg)%lum(ilum)%prm_prev%stress - pl_prms(1)%neg
                  plcal(ireg)%lum(ilum)%prm%stress = pl_prms(1)%neg
                  plcal(ireg)%lum(ilum)%prm_lim%stress = 1.
                end if
            
            !check all hru"s for proper lum
            do ihru_s = 1, plcal(ireg)%num_tot
              iihru = plcal(ireg)%num(ihru_s)
              if (plcal(ireg)%lum(ilum)%meas%name == hlt(ihru)%plant) then
                !set parms for surface runoff calibration and rerun
                hlt(iihru)= hlt_init(iihru)
                hlt(iihru)%stress = hlt(iihru)%stress + chg_val
                hlt(iihru)%stress = amin1 (hlt(iihru)%stress, pl_prms(1)%up)
                hlt(iihru)%stress = Max (hlt(iihru)%stress, pl_prms(1)%lo)
                hlt_init(iihru) = hlt(iihru)
              end if
            end do
            plcal(ireg)%lum(ilum)%nbyr = 0
            plcal(ireg)%lum(ilum)%precip_aa = 0.
            plcal(ireg)%lum(ilum)%aa = plcal_z
            !else
            !plcal(ireg)%lum(ilum)%prm_lim%etco = 1.
            end if
          end do
        end do
        ! plant stress adjustment
        if (isim > 0) then
          write (4304,*) " plant stress adj "
          call time_control
        end if
        end do      ! ist
          
      end do    ! iter_all loop
        
      cal_codes%plt = "n"
      
	  return
      end subroutine calsoft_plant