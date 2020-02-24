      subroutine sd_channel_control

      use sd_channel_module
      use channel_velocity_module
      use basin_module
      use hydrograph_module
      use constituent_mass_module
      use channel_data_module
      use channel_module
      use ch_pesticide_module
      use climate_module
      use water_body_module
      use time_module
    
      implicit none     
    
      !real :: rcharea                !m^2           |cross-sectional area of flow
      !real :: sdti                   !m^3/s         |flow rate in reach for day
      integer :: isd_db               !              |
      integer :: iob                  !              |
      integer :: idb                  !none          |channel data pointer
      integer :: ihyd                 !              |
      integer :: ipest                !              |
      real :: erode_btm               !              |
      real :: erode_bank              !              |
      real :: deg_btm                 !tons          |bottom erosion
      real :: deg_bank                !tons          |bank erosion
      real :: sedout                  !mg		     |sediment out of waterway channel
      real :: washld                  !tons          |wash load  
      real :: bedld                   !tons          |bed load
      real :: dep                     !tons          |deposition
      real :: hc_sed                  !tons          |headcut erosion
      real :: chside                  !none          |change in horizontal distance per unit
                                      !              |change in vertical distance on channel side
                                      !              |slopes; always set to 2 (slope=1/2)
      real :: a                       !m^2           |cross-sectional area of channel
      real :: b                       !m             |bottom width of channel
      real :: c                       !none          |inverse of channel side slope
      real :: p                       !m             |wetting perimeter

      real :: rh                      !m             |hydraulic radius
      real :: qman                    !m^3/s or m/s  |flow rate or flow velocity
      real :: frac                    !0-1           |fraction of hydrograph 
      real :: valint                  !              | 
      integer :: ivalint              !              |
      real :: tbase                   !none          |flow duration (fraction of 24 hr)
      real :: tb_pr                   !              |
      real :: tb                      !              |
      real :: vol_ovb                 !              |
      real :: const                   !              |
      integer :: ics                  !none          |counter
      real :: ob_const                !              |
      integer :: ii                   !none          |counter
      real :: sum_vol                 !              |
      real :: xx                      !              | 
      integer :: ic                   !              |
      real :: vol_overmx              !              |
      real :: flood_dep               !              | 
      real :: dep_e                   !              |
      real :: rto                     !none          |cloud cover factor 
      real :: e_btm                   !              |
      real :: sumtime                 !              |
      real :: vc                      !m/s           |flow velocity in reach
      real :: pr_ratio                !              |
      real  :: tw                     !              |
      real :: tave                    !              |
      real :: shear_btm_cr            !              |
      real :: shear_btm_adj           !none          |take out bedld_cap adjustment
      real :: shear_btm               !              |
      real :: shear_bank_cr           !              | 
      real :: qmm                     !              | 
      real :: qh                      !              | 
      real :: hc                      !m/yr          |head cut advance
      integer :: max                  !              |
      real :: chns                    !              |
      integer :: ihval                !none          |counter 
      real :: bedld_cap               !              |
      real :: perim_bed               !              |
      real :: vol
      real :: perim_bank              !              |
      real :: s_bank                  !              |
      real :: shear_bank              !              |
      real :: shear_bank_adj          !              | 
      real :: e_bank                  !              |
      real :: perc                    !              |
      integer :: iaq
      integer :: iaq_ch
      real :: det                     !hr            |time step
      real :: scoef                   !none          |Storage coefficient
      real :: rchvol
      
      ich = isdch
      isd_db = ob(icmd)%props
      iwst = ob(icmd)%wst
      erode_btm = 0.
      erode_bank = 0.
      deg_btm = 0.
      deg_bank = 0.
      sedout = 0.
      washld = 0.
      bedld = 0.
      dep = 0.
      hc = 0.
      hc_sed = 0.
      
      !! set ht1 to incoming hydrograph
      ht1 = ob(icmd)%hin
      chsd_d(ich)%flo_in = ht1%flo / 86400.     !flow for morphology output
      ch_in_d(ich) = ht1                        !set inflow om hydrograph
      ch_in_d(ich)%flo = ht1%flo / 86400.       !flow for om output
      
      !! if connected to aquifer - add flow
      if (sd_ch(ich)%aqu_link > 0) then
        iaq = sd_ch(ich)%aqu_link
        iaq_ch = sd_ch(ich)%aqu_link_ch
        if (aq_ch(iaq)%ch(iaq_ch)%flo_fr > 0.) then
          chsd_d(ich)%aqu_in = (aq_ch(iaq)%ch(iaq_ch)%flo_fr * aq_ch(iaq)%hd%flo) / 86400.
          ht1 = ht1 + aq_ch(iaq)%ch(iaq_ch)%flo_fr * aq_ch(iaq)%hd
          aq_ch(iaq)%ch(iaq_ch)%flo_fr = 0.
        end if
      end if
      hcs1 = obcs(icmd)%hin
      
      !! set outgoing flow and sediment - ht2
      ht2 = hz
      peakrate = 0.
      hyd_rad = 0.
      timeint = 0.

      !assume triangular hydrograph
      peakrate = 2. * ht1%flo / (1.5 * sd_chd(isd_db)%tc)
      peakrate = peakrate / 60.   !convert min to sec
      if (peakrate > 1.e-9) then
         !! compute changes in channel dimensions
          chside = sd_chd(isd_db)%chss
          b = sd_ch(ich)%chw
          sd_ch_vel(ich)%wid_btm = b
          sd_ch_vel(ich)%dep_bf = sd_ch(ich)%chd

          !! compute flow and travel time at bankfull depth
          p = b + 2. * sd_ch(ich)%chd * Sqrt(chside * chside + 1.)
          a = b * sd_ch(ich)%chd + chside * sd_ch(ich)%chd * sd_ch(ich)%chd
          rh = a / p
          sd_ch_vel(ich)%area = a
          sd_ch_vel(ich)%vel_bf = Qman (a, rh, sd_chd(isd_db)%chn, sd_ch(ich)%chs)
  
          IF (peakrate > sd_ch_vel(ich)%vel_bf) THEN
          !! OVERBANK FLOOD
            sd_ch(ich)%overbank = "ob"
            rcharea = sd_ch_vel(ich)%area
            rchdep = sd_ch(ich)%chd
            !calculate hydraulic radius at hydrograph time increments for degredation
            sdti = 0.
            rchdep = 0.
            valint = 1. / float(maxint)
            ivalint = 1
            tbase = 1.5 * sd_chd(isd_db)%tc * 60.   !hydrograph base time in s
            tb_pr = tbase
            DO WHILE (sdti < peakrate)
              rchdep = rchdep + 0.01
              rcharea = (sd_ch_vel(ich)%wid_btm + chside * rchdep) * rchdep
              p = sd_ch_vel(ich)%wid_btm + 2. * rchdep * Sqrt (1. + chside *chside)
              rh = rcharea / p
              sdti = Qman(rcharea, rh, sd_chd(isd_db)%chn, sd_ch(ich)%chs)
              !need to save hydraulic radius and time for each flow interval for downcutting and widening
              if (sdti > valint * peakrate) then
                hyd_rad(ivalint) = rh
                tb = (peakrate - sdti) * tbase / peakrate
                timeint(ivalint) = (tb_pr - tb) / 3600.   !sec to hr
                tb_pr = tb
                ivalint = ivalint + 1
                valint = float (ivalint) / float (maxint)
              end if
            END DO
            
            !! estimate overbank flow - assume a triangular hyd
            tbase = 1.5 * sd_chd(isd_db)%tc * 60.  !seconds
            vol_ovb = 0.5 * (peakrate - sd_ch_vel(ich)%vel_bf) * sd_ch_vel(ich)%vel_bf / peakrate * tbase
            vol_ovb = amin1(vol_ovb, ht1%flo)
            vol_ovb = peakrate - sd_ch_vel(ich)%vel_bf
            const = vol_ovb / peakrate
            ob(icmd)%hd(3) = const * ob(icmd)%hin
            
            !find current total flood volume (ht1)
            ics = ob(icmd)%props2
            if (ics > 0) then   ! flood elements are specified - link to surface elements
            ht1 = hz
            ob_const = const
            do ii = 1, ch_sur(ics)%num
              ht1 = ht1 + ch_sur(ics)%hd(ii)
            end do
            
            !add current and new flood volumes
            ht1 = ht1 + ob(icmd)%hd(3)
           if (ht1%flo > ch_sur(ics)%flood_volmx(0)) then
            !calc flood depth above channel bottom (flood_dep)
            sum_vol = 0.
            do ii = 1, ch_sur(ics)%num
              if (ht1%flo < ch_sur(ics)%flood_volmx(ii)) then
                !solve quadrative for depth above base of current element
                a = sd_ch(ich)%chl * 1000. / sd_ch(ich)%chs
                b = ch_sur(ics)%wid(ii-1) * sd_ch(ich)%chl * 1000.
                c =  ch_sur(ics)%flood_volmx(ii-1) - ht1%flo
                xx = b ** 2 - 4. * a * c
                dep = (-b + sqrt(xx)) / (2. * a)
                dep = Max(0., dep)
                ic = ii
                exit
              end if
              if (ii == ch_sur(ics)%num) then
                !flood is over the  max storage - assume linear upward wall
                ic = ch_sur(ics)%num
                vol_overmx = ht1%flo - ch_sur(ics)%flood_volmx(ii)
                dep = ch_sur(ics)%dep(ic) + (vol_overmx / (sd_ch(ich)%chl *  &
                               (sd_ch(ich)%chw + 2. * ch_sur(ics)%wid(ic))))
              end if
            end do
            !calc flood depth above channel bottom
            flood_dep = dep + ch_sur(ics)%dep(ic-1)
            !calc new flood volume for each element
            do ii = 1, ch_sur(ics)%num
              !calc depth in the element
              
              if (flood_dep < ch_sur(ics)%dep(ii-1)) then
                !no flooding on element
                ch_sur(ics)%hd(ii)%flo = 0.
              else if (flood_dep < ch_sur(ics)%dep(ii)) then
                !flood level within the element
                dep_e = flood_dep - ch_sur(ics)%dep(ii-1)
                ch_sur(ics)%hd(ii)%flo = dep_e ** 2 / sd_ch(ich)%chs * sd_ch(ich)%chl
              else
                !flood level over max element depth
                ch_sur(ics)%hd(ii)%flo = 2. * sd_ch(ich)%chw *             & 
                  (flood_dep - ch_sur(ics)%dep(ii)) + sd_ch(ich)%chw *     &
                  (ch_sur(ics)%dep(ii) - ch_sur(ics)%dep(ii-1))
              end if
              ch_sur(ics)%hd(ii)%flo = amin1 (ch_sur(ics)%hd(ii)%flo, ch_sur(ics)%flood_volmx(ii))
              sum_vol = sum_vol + ch_sur(ics)%hd(ii)%flo
            end do
            !determine fraction of total flood volume and set volume for each element
            rto = sum_vol / ht1%flo  !ensure water balance is maintained
            do ii = 1, ch_sur(ics)%num
              const = rto * ch_sur(ics)%hd(ii)%flo / ht1%flo
              ch_sur(ics)%hd(ii) = const * ht1
            end do
           end if
           end if
          ELSE
            !! CHANNELIZED FLOW
            !! find the crossectional area and depth for volrt
            !! by iteration method at 1cm interval depth
            !! find the depth until the discharge rate is equal to volrt
            !zero overbank flow
            sd_ch(ich)%overbank = "ib"
            ob(icmd)%hd(3) = hz
            ob_const = 1.
            sdti = 0.
            rchdep = 0.
            sumtime = 0.
            valint = 1. / float(maxint)
            ivalint = 1
            tbase = 1.5 * sd_chd(isd_db)%tc * 60.   !hydrograph base time in s
            tb_pr = tbase
            DO WHILE (sdti < peakrate)
              rchdep = rchdep + 0.01
              rcharea = (sd_ch_vel(ich)%wid_btm + chside * rchdep) * rchdep
              p = sd_ch_vel(ich)%wid_btm + 2. * rchdep*Sqrt(1. + chside * chside)
              rh = rcharea / p
              sdti = Qman(rcharea, rh, sd_chd(isd_db)%chn, sd_ch(ich)%chs)
              !need to save hydraulic radius and time for each flow interval for downcutting and widening
              if (sdti > valint * peakrate) then
                hyd_rad(ivalint) = rh
                tb = (peakrate - sdti) * tbase / peakrate
                timeint(ivalint) = (tb_pr - tb) / 3600.   !sec to hr
                sumtime = sumtime + timeint(ivalint)
                tb_pr = tb
                ivalint = ivalint + 1
                valint = float (ivalint) / float(maxint)
              end if
            END DO
            timeint = timeint / sumtime
          END IF

        !! adjust peak rate for headcut advance -also adjusts CEAP gully from
        !! edge-of-field to trib (assuming rectangular shape and constant tc)
        pr_ratio = (sd_ch(ich)%chl - sd_ch(ich)%hc_len / 1000.) / sd_ch(ich)%chl
        pr_ratio = Max(pr_ratio, 0.)
        
        !! new q*qp (m3 * m3/s) equation for entire runoff event
        qmm = ht1%flo / (10. * ob(icmd)%area_ha)
        if (qmm > 3.) then
          qh = (ht1%flo / 86400.) ** .5 * sd_chd(isd_db)%hc_hgt ** .225
          hc = sd_ch(ich)%hc_co * qh            !m per event
          hc = Max(hc, 0.)
          sd_ch(ich)%hc_len = sd_ch(ich)%hc_len + hc
          if (sd_ch(ich)%hc_len > sd_ch(ich)%chl * 1000.) then
            hc = hc - (sd_ch(ich)%hc_len - sd_ch(ich)%chl * 1000.)
            sd_ch(ich)%hc_len = sd_ch(ich)%chl * 1000.
          end if
            
          !! compute sediment yield from headcut- assume bd = 1.2 t/m3
          !! assume channel dimensions are same as data file
          hc_sed = hc * sd_chd(isd_db)%chw * sd_chd(isd_db)%chd * 1.2
        end if
        
        if (sd_ch(ich)%chs > sd_chd(isd_db)%chseq) then
          !break hydrograph into maxint segments and compute deg at each flow increment
          do ihval = 1, maxint
          !! downcutting bottom
          ! bedlaod capacity - simplified Meyers-Peter-Mueler from Chap 4 on Sediment Transport
          if (sd_chd(isd_db)%clay >= 10.) then
            chns = .0156
          else
            chns = (sd_chd(isd_db)%d50 / 25.4) ** .16666 / 39.
          end if
          shear_btm_cr = sd_chd(isd_db)%d50
          shear_btm = 9800. * hyd_rad(ihval) * sd_ch(ich)%chs   !! Pa = N/m^2 * m * m/m
          if (shear_btm > shear_btm_cr) then
            bedld_cap = 0.253 * (shear_btm - shear_btm_cr) ** 1.5
            shear_btm_adj = shear_btm * (1. - bedld / bedld_cap) !!* (chns / sd_chd(isd_db)%chn) ** 2
          else
            shear_btm_adj = 0.
          end if
          
          perim_bed = sd_ch(ich)%chw
          shear_btm_adj = shear_btm   !take out bedld_cap adjustment
	      if (shear_btm_adj > shear_btm_cr) then
            e_btm = timeint(ihval) *  sd_chd(isd_db)%cherod * (shear_btm_adj - shear_btm_cr)    !! cm = hr * cm/hr/Pa * Pa
            erode_btm = erode_btm + e_btm
            !! calculate mass of sediment eroded
            ! t = cm * m/100cm * width (m) * length (km) * 1000 m/km * bd (t/m3)
            !! degradation of the bottom (downcutting)
            perim_bed = sd_ch(ich)%chw
            deg_btm = deg_btm + 10. * e_btm * perim_bed * sd_ch(ich)%chl * sd_chd(isd_db)%bd
          end if

          !! widening sides
          perim_bank = 2. * ((sd_chd(isd_db)%chd ** 2) * (1. + sd_chd(isd_db)%chss ** 2)) ** 0.5
          tw = perim_bed + 2. * sd_chd(isd_db)%chss * rchdep
          s_bank = 1.77 * (perim_bed / perim_bank + 1.5) ** - 1.4
          shear_bank = shear_btm * .75                              !s_bank * (tw * perim_bed) / (2. * perim_bank)
          shear_bank_adj = sd_ch(ich)%shear_bnk * (1. - sd_chd(isd_db)%cov)      !* (chns / sd_chd(isd_db)%chn) ** 2
          shear_bank_cr = 0.493 * 10. ** (.0182 * sd_chd(isd_db)%clay)
          if (shear_bank_adj > shear_bank_cr) then
            e_bank = timeint(ihval) * sd_chd(isd_db)%cherod * (shear_bank_adj - shear_bank_cr)    !! cm = hr * cm/hr/Pa * Pa
            erode_bank = erode_bank + e_bank
            !! calculate mass of sediment eroded
            ! t = cm * m/100cm * width (m) * length (km) * 1000 m/km * bd (t/m3)
            !! degradation of the bank (widening)
            deg_bank = deg_bank + 10. * e_bank * perim_bank * sd_ch(ich)%chl * sd_chd(isd_db)%bd
          end if

          end do
          
          !! adjust for incoming bedload
          bedld = sd_chd(isd_db)%bedldcoef * ht1%sed
          erode_btm = (deg_btm - bedld) / (10. * perim_bed * sd_ch(ich)%chl * sd_chd(isd_db)%bd)
          erode_bank = MAX(0., erode_bank)
          sd_ch(ich)%chd = sd_ch(ich)%chd + erode_btm / 100.
          if (sd_ch(ich)%chd < 0.) then
            !! stream is completely filled in
            sd_ch(ich)%chd = 0.01
          end if
          sd_ch(ich)%chw = sd_ch(ich)%chw + 2. * erode_bank / 100.
          sd_ch(ich)%chs = sd_ch(ich)%chs - (erode_btm / 100.) / (sd_ch(ich)%chl * 1000.)
          sd_ch(ich)%chs = MAX(sd_chd(isd_db)%chseq, sd_ch(ich)%chs)

        end if

      !! compute sediment leaving the channel
	  washld = (1. - sd_chd(isd_db)%bedldcoef) * ht1%sed
	  sedout = washld + hc_sed + deg_btm + deg_bank
      dep = bedld - deg_btm - deg_bank
      ht2%sed = sedout

      !! set values for outflow hydrograph
      !! calculate flow velocity and travel time
      idb = ob(icmd)%props
      jrch = ich
      vc = 0.001
      if (rcharea > 1.e-4 .and. ht1%flo > 1.e-4) then
        vc = peakrate / rcharea
        if (vc > sd_ch_vel(ich)%celerity_bf) vc = sd_ch_vel(ich)%celerity_bf
        rttime = sd_ch(jhyd)%chl * 1000. / (3600. * vc)
        if (time%step == 0) rt_delt = 1.
        if (bsn_cc%wq == 1) then
          !! use modified qual-2e routines
          ht3 = ht1
          call hyd_convert_mass_to_conc (ht3)
          jnut = sd_dat(ich)%nut
          ben_area = sd_ch(ich)%chw * sd_ch(ich)%chl
          call ch_watqual4
          !! convert mass to concentration
          call hyd_convert_conc_to_mass (ht2)
         
          !! compute nutrient losses using 2-stage ditch model
          !call sd_channel_nutrients
        end if

        !! route constituents
        call ch_rtpest
        call ch_rtpath
      end if
      
      end if    ! peakrate > 0
      
      !! compute water balance - precip, evap and seep
      !! km * m * 1000 m/km * ha/10000 m2 = ha
      ch_wat_d(ich)%area_ha = sd_ch(ich)%chl * sd_ch(ich)%chw / 10.
      !! mm * ha * m/1000 mm = ha-m
      ch_wat_d(ich)%precip = wst(iwst)%weat%precip * ch_wat_d(ich)%area_ha / 1000.
      ch_wat_d(ich)%evap = bsn_prm%evrch * wst(iwst)%weat%pet * ch_wat_d(ich)%area_ha / 1000.
      ch_wat_d(ich)%seep = 24. * sd_chd(isd_db)%chk * ch_wat_d(ich)%area_ha / 1000.
      ht1%flo = ht1%flo + 10. * ch_wat_d(ich)%precip      !ha-m * 10 = m3
      !! subtract evaporation
      if (ht1%flo < 10. * ch_wat_d(ich)%evap) then
        ch_wat_d(ich)%evap = ht1%flo / 10.      !m3 -> ha-m
        ht1%flo = 0.
      else
        ht1%flo = ht1%flo - 10. * ch_wat_d(ich)%evap
      end if
      !! subtract seepage
      if (ht1%flo < 10. * ch_wat_d(ich)%seep) then
        ch_wat_d(ich)%seep = ht1%flo / 10.      !m3 -> ha-m
        ht1%flo = 0.
      else
        ht1%flo = ht1%flo - 10. * ch_wat_d(ich)%seep
      end if
      
      !! calculate hydrograph leaving reach and storage in channel
      if (time%step == 0) rt_delt = 1.
      det = 24.* rt_delt
      scoef = bsn_prm%scoef * det / (rttime + det)
      frac = 1. - scoef
      if (rttime > det) then      ! ht1 = incoming + storage
        !! travel time > timestep -- then all incoming is stored and frac of stored is routed
        ht2 = scoef * ch_stor(ich)
        ch_stor(ich) = frac * ch_stor(ich) + ht1
        hcs2 = scoef * ch_water(ich)
        ch_water(ich) = frac * ch_water(ich) + hcs1
      else
        !! travel time < timestep -- route all stored and frac of incoming
        ht2 = scoef * ht1
        ht2 = ht2 + ch_stor(ich)
        ch_stor(ich) = frac * ht1
        hcs2 = scoef * hcs1
        hcs2 = hcs2 + ch_water(ich)
        ch_water(ich) = frac * hcs1
      end if

      ob(icmd)%hd(1) = ht2
      ob(icmd)%hd(1)%temp = 5. + .75 * wst(iwst)%weat%tave
      
      if (cs_db%num_pests > 0) then
        obcs(icmd)%hd(1)%pest = hcs2%pest
      end if
      
      !! output channel organic-mineral
      ch_out_d(ich) = ht2                       !set inflow om hydrograph
      ch_out_d(ich)%flo = ht2%flo / 86400.      !m3 -> m3/s
      
      !! output channel morphology
      chsd_d(ich)%flo = ht2%flo / 86400.        !adjust if overbank flooding is moved to landscape
      chsd_d(ich)%peakr = peakrate 
      chsd_d(ich)%sed_in = ob(icmd)%hin%sed
      chsd_d(ich)%sed_out = sedout
      chsd_d(ich)%washld = washld
      chsd_d(ich)%bedld = bedld
      chsd_d(ich)%dep = dep
      chsd_d(ich)%deg_btm = deg_btm
      chsd_d(ich)%deg_bank = deg_bank
      chsd_d(ich)%hc_sed = hc_sed
      chsd_d(ich)%width = sd_ch(ich)%chw
      chsd_d(ich)%depth = sd_ch(ich)%chd
      chsd_d(ich)%slope = sd_ch(ich)%chs
      chsd_d(ich)%deg_btm_m = erode_btm
      chsd_d(ich)%deg_bank_m = erode_bank
      chsd_d(ich)%hc_m = hc
      
      !! set pesticide output variables
      do ipest = 1, cs_db%num_pests
        chpst_d(ich)%pest(ipest)%tot_in = obcs(icmd)%hin%pest(ipest)
        chpst_d(ich)%pest(ipest)%sol_out = frsol * obcs(icmd)%hd(1)%pest(ipest)
        chpst_d(ich)%pest(ipest)%sor_out = frsrb * obcs(icmd)%hd(1)%pest(ipest)
        chpst_d(ich)%pest(ipest)%react = chpst%pest(ipest)%react
        chpst_d(ich)%pest(ipest)%volat = chpst%pest(ipest)%volat
        chpst_d(ich)%pest(ipest)%settle = chpst%pest(ipest)%settle
        chpst_d(ich)%pest(ipest)%resus = chpst%pest(ipest)%resus
        chpst_d(ich)%pest(ipest)%difus = chpst%pest(ipest)%difus
        chpst_d(ich)%pest(ipest)%react_bot = chpst%pest(ipest)%react_bot
        chpst_d(ich)%pest(ipest)%bury = chpst%pest(ipest)%bury 
        chpst_d(ich)%pest(ipest)%water = ch_water(ich)%pest(ipest)
        chpst_d(ich)%pest(ipest)%benthic = ch_benthic(ich)%pest(ipest)
      end do
        
      !! set values for recharge hydrograph - should be trans losses
      !ob(icmd)%hd(2)%flo = perc  

      return
      
      end subroutine sd_channel_control