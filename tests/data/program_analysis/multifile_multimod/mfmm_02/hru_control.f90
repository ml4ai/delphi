      subroutine hru_control
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine controls the simulation of the land phase of the 
!!    hydrologic cycle

      use hru_module, only : hru, ihru, tmx, tmn, tmpav, hru_ra, hru_rmx, rhd, u10, tillage_switch,      &
         tillage_days, ndeat, qdr, phubase, sedyld, surfq, grz_days, yield, &
         yr_skip, latq, tconc, smx, sepbtm, igrz, iseptic, i_sep, filterw, sed_con, soln_con, solp_con, & 
         orgn_con, orgp_con, cnday, nplnt, percn, tileno3, pplnt, sedorgn, sedorgp, surqno3, latno3,    &
         surqsolp, sedminpa, sedminps,       &
         fertn, fertp, fixn, grazn, grazp, ipl, peakr, qtile,      &
         snofall, snomlt, tloss, usle, canev,   &
         ep_day, es_day, etday, inflpcp, isep, iwgen, ls_overq, nd_30, pet_day,              &
         precipday, precip_eff, qday, latqrunon
      use soil_module 
      use plant_module
      use basin_module
      use organic_mineral_mass_module
      use hydrograph_module
      use climate_module, only : wst, weat, wgn_pms
      use septic_data_module
      use reservoir_data_module
      use plant_data_module
      use mgt_operations_module
      use reservoir_module
      use output_landscape_module
      use output_ls_pesticide_module
      use time_module
      use conditional_module
      use constituent_mass_module
      use water_body_module
      
      implicit none

      integer :: j                  !none          |same as ihru (hru number)
      integer :: sb                 !              |  
      integer :: idp                !              |
      real :: ulu                   !              | 
      integer :: iob                !              |
      integer :: ith                !              |
      integer :: iwgn               !              |
      integer :: ires               !none          |reservoir number
      integer :: isched             !              |
      integer :: idat               !              |
      integer :: ihyd               !              |
      integer :: ics                !              |
      integer :: iauto              !none          |counter
      integer :: id                 !              |
      integer :: jj                 !              |
      integer :: ly                 !none          |soil layer
      integer :: ipest              !none          |sequential pesticide number
      real :: strsa_av              !              |
      real :: runoff_m3             !              |
      real :: bf_m3                 !              |
      real :: peakrbf               !              |
      integer :: icn                !              |
      real :: xx                    !              |
      integer :: iob_out            !              |object type out 
      integer :: iout               !none          |counter
      integer :: iac
      integer :: npl_gro            !           |number of plants currently growing
      real :: over_flow             !              |
      real :: dep                   !              |
      real :: strsw_av
      real :: strsn_av
      real :: strsp_av
      real :: strstmp_av
      real :: wet_outflow           !mm             |outflow from wetland
      
      j = ihru
      !if (pcom(j)%npl > 0) idp = pcom(ihru)%plcur(1)%idplt
      ulu = hru(j)%luse%urb_lu
      iob = hru(j)%obj_no
      iwst = ob(iob)%wst
      iwgen = wst(iwst)%wco%wgn
      ith = hru(j)%dbs%topo
      iwgn = wst(iwst)%wco%wgn
      ires =  hru(j)%dbs%surf_stor
      isched = hru(j)%mgt_ops
      
      weat = wst(iwst)%weat
      precip_eff = precipday
      
      precipday = wst(iwst)%weat%precip
      tmx(j) = wst(iwst)%weat%tmax
      tmn(j) = wst(iwst)%weat%tmin
      tmpav(j) = (tmx(j) + tmn(j)) / 2.
      hru_ra(j) = wst(iwst)%weat%solrad
      hru_rmx(j) = wst(iwst)%weat%solradmx
      rhd(j) = wst(iwst)%weat%rhum
      u10(j) = wst(iwst)%weat%windsp
      
      hru(ihru)%water_seep = 0.
      !plt => hru(j)%pl(1)

      if (bsn_cc%cswat == 2) then
        if (tillage_switch(ihru) .eq. 1) then
          if (tillage_days(ihru) .ge. 30) then
            tillage_switch(ihru) = 0
            tillage_days(ihru) = 0
          else
            tillage_days(ihru) = tillage_days(ihru) + 1
          end if                
          !tillage_depth(ihru) = dtil
          !tillage_switch(ihru) = .TRUE. 
        end if
      end if

      !! zero pesticide balance variables
      if (cs_db%num_pests > 0) then
        do ipest = 1, cs_db%num_pests
          hpestb_d(j)%pest(ipest) = pestbz
        end do
      end if
        
      call varinit
      nd_30 = nd_30 + 1
      if (nd_30 > 30) nd_30 = 1

        !!ht1== deposition: write to deposition.out
        !!ht2== outflow from inflow: added to hru generated flows
        ht1 = hz
        ht2 = hz
                  
        !!route overland flow across hru
        if (ob(icmd)%hin_sur%flo > 1.e-6) then
          !!route incoming surface runoff
          if (ires > 0) then
            !! add surface runon to wetland
            wet(j) = wet(j) + ob(icmd)%hin_sur
          else
            !! route across hru - infiltrate and deposit sediment
            call rls_routesurf (icmd)
          end if
        end if
        
        !!add lateral flow soil water
        if (ob(icmd)%hin_lat%flo > 0) then
          !!Route incoming lateral soil flow
          call rls_routesoil (icmd)
        end if
          
        !!add tile flow to tile (subirrigation and saturated buffer)
        if (ob(icmd)%hin_til%flo > 1.e-6) then
          call rls_routetile (icmd)
        end if

        !! check auto operations
        if (sched(isched)%num_autos > 0) then
          do iauto = 1, sched(isched)%num_autos
            id = sched(isched)%num_db(iauto)
            jj = j
            d_tbl => dtbl_lum(id)
            call conditions (jj)
            call actions (jj, iob, iauto)
            
            !! if end of year, reset the one time fert application per year
            if (time%end_yr == 1) then
              do iac = 1, d_tbl%acts
                pcom(j)%dtbl(iauto)%num_actions(iac) = 1
              end do
            end if
          end do
          if (time%end_yr == 1) then
            pcom(j)%rot_yr = pcom(j)%rot_yr + 1
          end if
          !! increment days since last plant and harvest
          pcom(j)%days_plant = pcom(j)%days_plant + 1
          pcom(j)%days_harv = pcom(j)%days_harv + 1
        end if
        
        !! update base zero total heat units
        if (tmpav(j) > 0. .and. wgn_pms(iwgn)%phutot > 0.01) then
           phubase(j) = phubase(j) + tmpav(j) / wgn_pms(iwgn)%phutot
        end if
   
        !! zero stresses
        do ipl = 1, pcom(j)%npl
          pcom(j)%plstr(ipl)%strsw = 1.
          pcom(j)%plstr(ipl)%strst = 1.
          pcom(j)%plstr(ipl)%strsn = 1.
          pcom(j)%plstr(ipl)%strsp = 1.
          pcom(j)%plstr(ipl)%strsa = 1.
        end do

        !! calculate albedo for day
        call albedo

        !! calculate soil temperature for soil layers
        call stmp_solt
        
        !! compute surface runoff processes
        call surface
                  
        !! compute evapotranspiration
        call et_pot
        call et_act

        !! ht2%sed==sediment routed across hru from surface runon
        sedyld(j) = sedyld(j) + ht2%sed

        !! compute effective rainfall (amount that percs into soil)
        !! add infiltration from surface runon
        inflpcp = (precip_eff - surfq(j)) * (1.-hru(j)%water_fr) + hru(j)%water_seep /(10.* hru(j)%area_ha)
        inflpcp = Max(0., inflpcp)
         
        !! perform management operations
        if (yr_skip(j) == 0) call mgt_operatn   
        
        !! perform soil water routing
        call swr_percmain

        !! wetland processes
        hru(j)%water_fr = 0.
        if (ires > 0) then
          call wetland_control
        else
          ht2%flo = wet(j)%flo * hru(j)%area_ha * 10.
          wet(j)%flo = 0.
        end if
 
        !! compute peak rate similar to swat-deg using SCS triangular unit hydrograph
        runoff_m3 = 10. * surfq(j) * hru(j)%area_ha
        bf_m3 = 10. * latq(j) * hru(j)%area_ha
        peakr = 2. * runoff_m3 / (1.5 * tconc(j) * 3600.)
        peakrbf = bf_m3 / 86400.
        peakr = (peakr + peakrbf)     !* prf     

        !! graze only if adequate biomass in HRU
        if (igrz(j) == 1) then
          ndeat(j) = ndeat(j) + 1
          !! if total above ground biomass is available - graze
          call pl_graze
          !! check to set if grazing period is over
          if (ndeat(j) == grz_days(j)) then
            igrz(j) = 0
            ndeat(j) = 0
          end if
        end if

        !! compute plant community partitions
        call pl_community

        !! compute plant biomass, leaf, root and seed growth
        call pl_grow
      
        !! moisture growth perennials - start growth
        if (pcom(j)%mseas == 1) then
          call pl_moisture_gro_init
        end if
        !! moisture growth perennials - start senescence
        if (pcom(j)%mseas == 0) then
          call pl_moisture_senes_init
        end if
        
        !! compute total parms for all plants in the community
        strsw_av = 0.; strsa_av = 0.; strsn_av = 0.; strsp_av = 0.; strstmp_av = 0.
        npl_gro = 0
        do ipl = 1, pcom(j)%npl
          if (pcom(j)%plcur(ipl)%gro == 'y' .and. pcom(j)%plcur(ipl)%idorm == 'n') then
            npl_gro = npl_gro + 1
            strsw_av = strsw_av + (1. - pcom(j)%plstr(ipl)%strsw)
            strsa_av = strsa_av + (1. - pcom(j)%plstr(ipl)%strsa)
            strsn_av = strsn_av + (1. - pcom(j)%plstr(ipl)%strsn)
            strsp_av = strsp_av + (1. - pcom(j)%plstr(ipl)%strsp)
            strstmp_av = strstmp_av + (1. - pcom(j)%plstr(ipl)%strst)
          end if
        end do
        if (npl_gro > 0) then
          strsw_av = strsw_av / npl_gro
          strsa_av = strsa_av / npl_gro
          strsn_av = strsn_av / npl_gro
          strsp_av = strsp_av / npl_gro
          strstmp_av = strstmp_av / npl_gro
        end if

        !! compute aoil water content to 300 mm depth
        soil(j)%sw_300 = 0.
        do ly = 1, soil(j)%nly
          if (ly == 1) then
            dep = 0.
          else
            dep = soil(j)%phys(ly-1)%d
          end if
          if (soil(j)%phys(ly)%d >= 300.) then
            soil(j)%sw_300 = soil(j)%sw_300 + soil(j)%phys(ly)%st *         &
                                   (300. - dep) / soil(j)%phys(ly)%thick
            exit
          else
            soil(j)%sw_300 = soil(j)%sw_300 + soil(j)%phys(ly)%st
          end if
        end do
        
        !! compute total surface residue
        do ipl = 1, pcom(j)%npl
          rsd1(j)%tot_com = rsd1(j)%tot_com + rsd1(j)%tot(ipl)
        end do
        
        !! compute actual ET for day in HRU
        etday = ep_day + es_day + canev

        !! compute nitrogen and phosphorus mineralization 
        if (bsn_cc%cswat == 0) then
          call nut_nminrl
        end if

	    if (bsn_cc%cswat == 2) then
	      call cbn_zhang2
	    end if

        call nut_nitvol
        if (bsn_cc%sol_P_model == 0) then
            call nut_pminrl
        else
            call nut_pminrl2
        end if

        !! compute biozone processes in septic HRUs
        !! if 1) current is septic hru and 2) soil temperature is above zero
        isep = iseptic(j)
	    if (sep(isep)%opt /= 0. .and. time%yrc >= sep(isep)%yr) then
	      if (soil(j)%phys(i_sep(j))%tmp > 0.) call sep_biozone     
        endif

        !! compute pesticide washoff   
        if (precipday >= 2.54) call pest_washp

        !! compute pesticide degradation
        call pest_decay

        !! compute pesticide movement in soil
        call pest_lch
      
        !! sum total pesticide in soil
        call pest_soil_tot
        
        if (surfq(j) > 0. .and. peakr > 1.e-6) then
          if (precip_eff > 0.) then
            call pest_enrsb
            if (sedyld(j) > 0.) call pest_pesty

		  if (bsn_cc%cswat == 0) then
			call nut_orgn
	      end if
	      if (bsn_cc%cswat == 1) then	    
		    call nut_orgnc
		  end if
		  
		  !! Add by zhang
		  !! ====================
		  if (bsn_cc%cswat == 2) then
		    call nut_orgnc2
		  end if
		  !! Add by zhang
		  !! ====================

            call nut_psed
          end if
        end if

        !! add nitrate in rainfall to soil profile
        call nut_nrain

        !! compute nitrate movement leaching
        call nut_nlch

        !! compute phosphorus movement
        call nut_solp

        !! compute chl-a, CBOD and dissolved oxygen loadings
        call swr_subwq

        !! compute pathogen transport
        if (cs_db%num_paths > 0.) then
          call path_ls_swrouting
          call path_ls_runoff
          call path_ls_process
        end if

        !! compute loadings from urban areas
        if (hru(j)%luse%urb_lu > 0) then
	     if(time%step == 0) then
	        call hru_urban ! daily simulation
	     else
		     call hru_urbanhr ! subdaily simulation J.Jeong 4/20/2009
	     endif
	  endif	  

        !! compute sediment loading in lateral flow and add to sedyld
        call swr_latsed

        !! lag nutrients and sediment in surface runoff
        call stor_surfstor

        !! lag subsurface flow and nitrate in subsurface flow
        call swr_substor

        !! compute reduction in pollutants due to edge-of-field filter strip
        if (hru(j)%lumv%vfsi > 0.)then
          call smp_filter
          if (filterw(j) > 0.) call smp_buffer
        end if

	 !! compute reduction in pollutants due to in field grass waterway
         if (hru(j)%lumv%grwat_i == 1) then
          call smp_grass_wway
        end if

	   !! compute reduction in pollutants due to in fixed BMP eff
	   if (hru(j)%lumv%bmp_flag == 1) then
          call smp_bmpfixed
        end if

        !! compute water yield for HRU
        qdr(j) = qday + latq(j) + qtile
        
        !! ht2%flo is outflow from wetland or total saturation excess if no wetland

        if(ht2%flo > 0.) then
          wet_outflow = ht2%flo / hru(j)%area_ha / 10.   !! mm = m3/ha *ha/10000m2 *1000mm/m
          qday = qday + wet_outflow
          qdr(j) = qdr(j) + wet_outflow
          surfq(j) = surfq(j) + wet_outflow
          ht2%flo = 0.
        end if

        if (qdr(j) < 0.) qdr(j) = 0.

        xx = sed_con(j) + soln_con(j) + solp_con(j) + orgn_con(j) + orgp_con(j)
        if (xx > 1.e-6) then
          call hru_urb_bmp
        end if
      
      ! update total residue on surface
      rsd1(j)%tot_com = orgz
      do ipl = 1, pcom(j)%npl
        rsd1(j)%tot_com = rsd1(j)%tot_com + rsd1(j)%tot(ipl)
      end do

      ! compute outflow objects (flow to channels, reservoirs, or landscape)
      ! if flow from hru is directly routed
      iob_out = iob
      ! if the hru is part of a ru and it is routed
      if (ob(iob)%ru_tot > 0) then
        iob_out = sp_ob1%ru + ob(iob)%ru(1) - 1
      end if
      
      hwb_d(j)%surq_cha = 0.
      hwb_d(j)%latq_cha = 0.
      hwb_d(j)%surq_res = 0.
      hwb_d(j)%latq_res = 0.
      hwb_d(j)%surq_ls = 0.
      hwb_d(j)%latq_ls = 0.
      
      do iout = 1, ob(iob_out)%src_tot
        select case (ob(iob_out)%obtyp_out(iout))
        case ("cha")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_cha = hwb_d(j)%surq_cha + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_cha = hwb_d(j)%latq_cha + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        case ("sdc")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_cha = hwb_d(j)%surq_cha + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_cha = hwb_d(j)%latq_cha + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        case ("res")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_res = hwb_d(j)%surq_res + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_res = hwb_d(j)%latq_res + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        case ("hru")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_ls = hwb_d(j)%surq_ls + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_ls = hwb_d(j)%latq_ls + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        case ("ru")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_ls = hwb_d(j)%surq_ls + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_ls = hwb_d(j)%latq_ls + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        case ("hlt")
          if (ob(iob_out)%htyp_out(iout) == "sur" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%surq_ls = hwb_d(j)%surq_ls + surfq(j) * ob(iob_out)%frac_out(iout)
          end if
          if (ob(iob_out)%htyp_out(iout) == "lat" .or. ob(iob_out)%htyp_out(iout) == "tot") then
            hwb_d(j)%latq_ls = hwb_d(j)%latq_ls + latq(j) * ob(iob_out)%frac_out(iout)
          end if
        end select
      end do

      ! output_waterbal
        hwb_d(j)%precip = wst(iwst)%weat%precip
        hwb_d(j)%snofall = snofall
        hwb_d(j)%snomlt = snomlt
        hwb_d(j)%surq_gen = qday
        hwb_d(j)%latq = latq(j)
        hwb_d(j)%wateryld = qdr(j)
        hwb_d(j)%perc = sepbtm(j)
        !! add evap from impounded water (wetland) to et and esoil
        hwb_d(j)%et = etday + hru(j)%water_evap
        hwb_d(j)%tloss = tloss
        hwb_d(j)%eplant = ep_day
        hwb_d(j)%esoil = es_day + hru(j)%water_evap 
        hwb_d(j)%surq_cont = surfq(j)
        hwb_d(j)%cn = cnday(j)
        hwb_d(j)%sw = soil(j)%sw
        hwb_d(j)%sw_300 = soil(j)%sw_300
        hwb_d(j)%snopack = hru(j)%sno_mm
        hwb_d(j)%pet = pet_day
        hwb_d(j)%qtile = qtile
        hwb_d(j)%irr = irrig(j)%applied
        irrig(j)%applied = 0.
        hwb_d(j)%surq_runon = ls_overq
        hwb_d(j)%latq_runon = latqrunon !/ (10. * hru(j)%area_ha) 
        ! hwb_d(j)%overbank = over_flow     !overbank is not added yet

      ! output_nutbal
        hnb_d(j)%grazn = grazn
        hnb_d(j)%grazp = grazp
        hnb_d(j)%fertn = fertn
        hnb_d(j)%fertp = fertp
        hnb_d(j)%fixn = fixn

      ! output_plantweather
        hpw_d(j)%lai = pcom(j)%lai_sum
        hpw_d(j)%bioms = pl_mass(j)%tot_com%m
        hpw_d(j)%residue = rsd1(j)%tot_com%m
        hpw_d(j)%yield = pl_yield%m
        pl_yield = plt_mass_z
        hpw_d(j)%sol_tmp =  soil(j)%phys(2)%tmp
        hpw_d(j)%strsw = strsw_av
        hpw_d(j)%strsa = strsa_av
        hpw_d(j)%strstmp = strstmp_av
        hpw_d(j)%strsn = strsn_av      
        hpw_d(j)%strsp = strsp_av
        hpw_d(j)%nplnt = nplnt(j)
        hpw_d(j)%percn = percn(j)
        hpw_d(j)%pplnt = pplnt(j)
        hpw_d(j)%tmx = tmx(j)
        hpw_d(j)%tmn = tmn(j)
        hpw_d(j)%tmpav = tmpav(j)
        hpw_d(j)%solrad = hru_ra(j)
        hpw_d(j)%phubase0 = phubase(j)

      ! output_losses
        hls_d(j)%sedyld = sedyld(j) / hru(j)%area_ha
        hls_d(j)%sedorgn = sedorgn(j)
        hls_d(j)%sedorgp = sedorgp(j)
        hls_d(j)%surqno3 = surqno3(j)
        hls_d(j)%latno3 = latno3(j)
        hls_d(j)%surqsolp = surqsolp(j)
        hls_d(j)%usle = usle
        hls_d(j)%sedmin = sedminpa(j) + sedminps(j)
        hls_d(j)%tileno3 = tileno3(j)

      !! set hydrographs for direct routing or landscape unit
      call hru_hyds
      
      !! check decision table for flow control - water allocation
      if (ob(iob)%ruleset /= "null" .and. ob(iob)%ruleset /= "0") then
        id = ob(iob)%flo_dtbl
        jj = j
        d_tbl => dtbl_flo(id)
        call conditions (jj)
        call actions (jj, iob, id)
      end if

      return
      end subroutine hru_control