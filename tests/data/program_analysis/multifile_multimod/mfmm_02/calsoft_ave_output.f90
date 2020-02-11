      subroutine calsoft_ave_output
    
      use sd_channel_module
      use hru_lte_module
      use maximum_data_module
      use calibration_data_module
      use time_module
      
      implicit none
      
      integer :: ireg       !none      |counter
      integer :: ilu        !none      |counter
      integer :: ich        !none      |counter 
      

        !! average output for soft data calibration
        if (cal_codes%hyd_hru == "y" .or. cal_codes%sed == "y") then
          !average annual for hru calibration
          do ireg = 1, db_mx%lsu_reg
            do ilu = 1, region(ireg)%nlum
              if (lscal(ireg)%lum(ilu)%nbyr > 0) then
                !! convert back to mm, t/ha, kg/ha
                lscal(ireg)%lum(ilu)%precip_aa = lscal(ireg)%lum(ilu)%precip_aa / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%precip_aa_sav = lscal(ireg)%lum(ilu)%precip_aa
                lscal(ireg)%lum(ilu)%pet_aa = lscal(ireg)%lum(ilu)%pet_aa / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%aa%srr = lscal(ireg)%lum(ilu)%aa%srr / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%aa%lfr = lscal(ireg)%lum(ilu)%aa%lfr / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%aa%pcr = lscal(ireg)%lum(ilu)%aa%pcr / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%aa%etr = lscal(ireg)%lum(ilu)%aa%etr / lscal(ireg)%lum(ilu)%nbyr
                lscal(ireg)%lum(ilu)%aa%tfr = lscal(ireg)%lum(ilu)%aa%tfr / lscal(ireg)%lum(ilu)%nbyr
                !lscal(ireg)%lum(ilu)%aa%sed = lscal(ireg)%lum(ilu)%aa%sed / lscal(ireg)%lum(ilu)%nbyr
                ! add nutrients
              end if
            end do
          end do
        end if
        
        if (cal_codes%hyd_hrul == "y") then
          !average annual for hru_lte calibration
          do ireg = 1, db_mx%lsu_reg
            do ilu = 1, lscalt(ireg)%lum_num
              if (lscalt(ireg)%lum(ilu)%nbyr > 0) then
                !! convert back to mm, t/ha, kg/ha
                lscalt(ireg)%lum(ilu)%precip_aa = lscalt(ireg)%lum(ilu)%precip_aa / lscalt(ireg)%lum(ilu)%nbyr
                lscalt(ireg)%lum(ilu)%precip_aa_sav = lscalt(ireg)%lum(ilu)%precip_aa
                lscalt(ireg)%lum(ilu)%aa%srr = lscalt(ireg)%lum(ilu)%aa%srr / lscalt(ireg)%lum(ilu)%nbyr
                lscalt(ireg)%lum(ilu)%aa%lfr = lscalt(ireg)%lum(ilu)%aa%lfr / lscalt(ireg)%lum(ilu)%nbyr
                lscalt(ireg)%lum(ilu)%aa%pcr = lscalt(ireg)%lum(ilu)%aa%pcr / lscalt(ireg)%lum(ilu)%nbyr
                lscalt(ireg)%lum(ilu)%aa%etr = lscalt(ireg)%lum(ilu)%aa%etr / lscalt(ireg)%lum(ilu)%nbyr
                lscalt(ireg)%lum(ilu)%aa%tfr = lscalt(ireg)%lum(ilu)%aa%tfr / lscalt(ireg)%lum(ilu)%nbyr
                !lscalt(ireg)%lum(ilu)%aa%sed = lscalt(ireg)%lum(ilu)%aa%sed / lscalt(ireg)%lum(ilu)%nbyr
                ! add nutrients
              end if
            end do
          end do
        end if
            
        !! average output for soft data calibration
        if (cal_codes%plt == "y") then
            
          !average annual for plant calibration
          do ireg = 1, db_mx%plcal_reg
            do ilu = 1, plcal(ireg)%lum_num
              if (plcal(ireg)%lum(ilu)%ha > 0) then
                plcal(ireg)%lum(ilu)%aa%yield = plcal(ireg)%lum(ilu)%sim%yield / plcal(ireg)%lum(ilu)%ha
              end if
              !if (plcal(ireg)%lum(ilu)%nbyr > 0) then
              !  !! convert back to mm, t/ha, kg/ha
              !  plcal(ireg)%lum(ilu)%precip_aa = plcal(ireg)%lum(ilu)%precip_aa / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%precip_aa_sav = plcal(ireg)%lum(ilu)%precip_aa
              !  plcal(ireg)%lum(ilu)%aa%yield = plcal(ireg)%lum(ilu)%aa%yield / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%aa%npp = plcal(ireg)%lum(ilu)%aa%npp / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%aa%lai_mx = plcal(ireg)%lum(ilu)%aa%lai_mx / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%aa%wstress = plcal(ireg)%lum(ilu)%aa%wstress / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%aa%astress = plcal(ireg)%lum(ilu)%aa%astress / plcal(ireg)%lum(ilu)%nbyr
              !  plcal(ireg)%lum(ilu)%aa%tstress = plcal(ireg)%lum(ilu)%aa%tstress / plcal(ireg)%lum(ilu)%nbyr
              !  ! add nutrients
              !end if
            end do
          end do
        end if
        
        !! average channel output for soft data calibration
        if (cal_codes%chsed == "y" .and. cal_codes%sed == "n" .and. cal_codes%plt == "n" .and. cal_codes%hyd_hru == "n" .and. cal_codes%hyd_hrul == "n") then
          do ireg = 1, db_mx%ch_reg
            do ich = 1, chcal(ireg)%ord_num
              if (chcal(ireg)%ord(ich)%nbyr > 0) then
                !! soft data for w and d in mm/year (convert to m) -- hc soft and model in m -- fpd soft and model in mm
                chcal(ireg)%ord(ich)%aa%chd = 1000. * chcal(ireg)%ord(ich)%aa%chd / chcal(ireg)%ord(ich)%nbyr
                chcal(ireg)%ord(ich)%aa%chw = 1000. * chcal(ireg)%ord(ich)%aa%chw / chcal(ireg)%ord(ich)%nbyr
                chcal(ireg)%ord(ich)%aa%hc = chcal(ireg)%ord(ich)%aa%hc / chcal(ireg)%ord(ich)%nbyr
                chcal(ireg)%ord(ich)%aa%fpd = chcal(ireg)%ord(ich)%aa%fpd / chcal(ireg)%ord(ich)%nbyr
              end if
            end do
          end do
        end if

      return
      end subroutine calsoft_ave_output