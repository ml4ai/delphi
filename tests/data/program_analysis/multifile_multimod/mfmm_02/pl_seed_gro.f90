      subroutine pl_seed_gro
      
      use plant_data_module
      use basin_module
      use hru_module, only : hru, uapd, uno3d, lai_yrmx, par, bioday, ep_day, es_day,              &
         ihru, ipl, pet_day, rto_no3, rto_solp, sum_no3, sum_solp, uapd_tot, uno3d_tot, vpd
      use plant_module
      use carbon_module
      use organic_mineral_mass_module
      use climate_module
      USE hydrograph_module
      
      implicit none 
      
      integer :: j              !none               |HRU number
      integer :: idp            !                   |
      real :: ajhi              !
      real :: dhi               !
      real :: temp_dif          !
      real :: temp_adj          !
    
      j = ihru
      idp = pcom(j)%plcur(ipl)%idplt
      iwst = ob(j)%wst
      
      !! calculate plant ET values
      if (pcom(j)%plcur(ipl)%phuacc > 0.5 .and. pcom(j)%plcur(ipl)%phuacc < pldb(idp)%dlai) then 
        pcom(j)%plg(ipl)%plet = pcom(j)%plg(ipl)%plet + ep_day + es_day
        pcom(j)%plg(ipl)%plpet = pcom(j)%plg(ipl)%plpet + pet_day
      end if
     
      ajhi = pldb(idp)%hvsti * 100. * pcom(j)%plcur(ipl)%phuacc /          &
                (100. * pcom(j)%plcur(ipl)%phuacc + Exp(11.1 - 10. * pcom(j)%plcur(ipl)%phuacc))
       
      !! adjust harvest index for temperature stress
      dhi = ajhi - pcom(j)%plg(ipl)%hi_prev
      temp_dif = pldb(idp)%t_opt - wst(iwst)%weat%tave
      if (temp_dif < 0. .and. pcom(j)%plcur(ipl)%phuacc > 0.7) then
        temp_adj = Exp (8. * temp_dif / pldb(idp)%t_opt)
        dhi = dhi * temp_adj
      end if
      pcom(j)%plg(ipl)%hi_adj = pcom(j)%plg(ipl)%hi_adj + dhi
      pcom(j)%plg(ipl)%hi_adj = amax1 (0., pcom(j)%plg(ipl)%hi_adj)
      pcom(j)%plg(ipl)%hi_prev = ajhi

      return
      end subroutine pl_seed_gro