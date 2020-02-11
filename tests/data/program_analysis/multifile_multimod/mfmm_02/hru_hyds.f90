      subroutine hru_hyds
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine summarizes data for subbasins with multiple HRUs and
!!    prints the daily output.hru file

      use hru_module, only : cbodu, chl_a, clayld, doxq, hhsurfq, hru, ihru, itb, lagyld, latq, peakr, percn, qday,  &
         sagyld, sanyld, silyld, sedminpa, sedminps, sedorgn, sedorgp, sepbtm, surqno3, surqsolp, tileno3,     &
         tmpav, uh, sedyld, latno3, qtile
      use hydrograph_module
      use basin_module
      use time_module
      use constituent_mass_module
      use output_ls_pesticide_module
      
      implicit none

      integer :: j                   !none          |same as ihru (hru number)
      integer :: kk                  !none          |counter
      integer :: ii                  !none          |counter
      real :: cnv                    !none          |conversion factor (mm/ha => m^3)
      real :: wtmp                   !deg C         |temperature of water in reach
      ! hqd, hsd locally defined. J.Jeong 4/26/2009
      !real :: hqd(4*time%step)
      !real :: hsd(4*time%step)
      !real :: hqdtst(time%step)  
      real :: cnv_m3                 !              |
      real :: cnv_kg                 !              |
      integer :: ihdmx               !              | 
      integer :: iob                 !              |
      integer :: ihyd                !none          |counter  
      integer :: iday                !              | 
      integer :: iday_prev           !              |
      real :: ssq                    !              |
      real :: sumflo                 !              |
      real :: sumflo_day             !              |
      integer :: itot                !              |
      integer :: ib                  !none          |counter
      real :: rto                    !none          |cloud cover factor
      real :: ratio                  !              |   
      integer :: iadj                !none          |counter
      integer :: istep               !none          |counter 
      integer :: ipest               !none          |counter
      integer :: ipath               !none          |counter 
      

      j = ihru
      cnv_m3 = hru(j)%area_ha * 10.
      cnv_kg = hru(j)%area_ha
      ihdmx = 2

      !! assign reach loadings for subbasin
      !! zero out hydrograph storage locations
      iob = icmd 
      ob(icmd)%hd(3) = hz

      !! surface runoff hydrograph (3)
      ob(icmd)%peakrate = peakr
      ob(icmd)%hd(3)%temp = 5. + .75 * tmpav(j)       !!wtmp
      ob(icmd)%hd(3)%flo = qday * cnv_m3              !!qdr m3/d
      ob(icmd)%hd(3)%sed = sedyld(j)                  !!sedyld
      ob(icmd)%hd(3)%orgn = sedorgn(j) * cnv_kg       !!sedorgn
      ob(icmd)%hd(3)%sedp = (sedorgp(j) + sedminpa(j) +                 &
                        sedminps(j)) * cnv_kg         !!sedorgp & sedminps
      ob(icmd)%hd(3)%no3 = surqno3(j) * cnv_kg        !!surqno3 & latno3 & no3gw
      ob(icmd)%hd(3)%solp = surqsolp(j) * cnv_kg      !!surqsolp & sedminpa
      ob(icmd)%hd(3)%chla = chl_a(j) *cnv_kg          !!chl_a
      ob(icmd)%hd(3)%nh3 = 0.                         !! NH3
      ob(icmd)%hd(3)%no2 = 0.                         !! NO2
      ob(icmd)%hd(3)%cbod = cbodu(j) * cnv_kg         !!cbodu
      ob(icmd)%hd(3)%dox = doxq(j) *cnv_kg            !!doxq & soxy

      ob(icmd)%hd(3)%san = sanyld(j)                  !! detached sand
      ob(icmd)%hd(3)%sil = silyld(j)                  !! detached silt
      ob(icmd)%hd(3)%cla = clayld(j)                  !! detached clay
      ob(icmd)%hd(3)%sag = sagyld(j)                  !! detached small aggregates
      ob(icmd)%hd(3)%lag = lagyld(j)                  !! detached large aggregates
      !set constituents
      do ipest = 1, cs_db%num_pests
        obcs(icmd)%hd(3)%pest(ipest) = (hpestb_d(j)%pest(ipest)%surq + hpestb_d(j)%pest(ipest)%sed) * cnv_kg
      end do
      do ipath = 1, cs_db%num_paths
        obcs(icmd)%hd(3)%path(ipath) = 0
      end do
      
      !recharge hydrograph (2)
      ob(icmd)%hd(2)%flo = sepbtm(j) * cnv_m3           !! recharge flow
      ob(icmd)%hd(2)%no3 = percn(j) * cnv_kg            !! recharge nitrate
      !set constituents
      do ipest = 1, cs_db%num_pests
        obcs(icmd)%hd(2)%pest(ipest) = hpestb_d(j)%pest(ipest)%perc * cnv_kg
      end do
      do ipath = 1, cs_db%num_paths
        obcs(icmd)%hd(2)%path(ipath) = 0
      end do
      
      !lateral soil flow hydrograph (4)
      ob(icmd)%hd(4)%flo = latq(j) * cnv_m3             !! lateral flow
      ob(icmd)%hd(4)%no3 = latno3(j) * cnv_kg
      !set constituents
      do ipest = 1, cs_db%num_pests
        obcs(icmd)%hd(4)%pest(ipest) = hpestb_d(j)%pest(ipest)%latq * cnv_kg
      end do
      do ipath = 1, cs_db%num_paths
        obcs(icmd)%hd(4)%path(ipath) = 0
      end do
      
      !tile flow hydrograph (5)
      ob(icmd)%hd(5)%flo = qtile * cnv_m3               !! tile flow
      ob(icmd)%hd(5)%no3 = tileno3(j) * cnv_kg          !! tile flow nitrate 
      !set constituents
      do ipest = 1, cs_db%num_pests
        obcs(icmd)%hd(5)%pest(ipest) = hpestb_d(j)%pest(ipest)%tileq * cnv_kg
      end do
      do ipath = 1, cs_db%num_paths
        obcs(icmd)%hd(5)%path(ipath) = 0.
      end do
      
      !sum to obtain the total outflow hydrograph (1)
      ob(icmd)%hd(1) = hz
      do ihyd = 3, 5
        ob(icmd)%hd(1) = ob(icmd)%hd(1) + ob(icmd)%hd(ihyd)
      end do
      !set constituents
      do ipest = 1, cs_db%num_pests
        obcs(icmd)%hd(1)%pest(ipest) = obcs(icmd)%hd(3)%pest(ipest) + obcs(icmd)%hd(4)%pest(ipest) +    &
                                                                      obcs(icmd)%hd(5)%pest(ipest)
      end do
      do ipath = 1, cs_db%num_paths
        obcs(icmd)%hd(1)%path(ipath) = 0
      end do
      
      !! set subdaily hydrographs
      if (time%step > 0) then
      iday = ob(icmd)%day_cur
      iday_prev = iday - 1
      if (iday_prev < 1) iday_prev = 2
        
      !! subsurface flow = lateral + tile
      ssq = (ob(icmd)%hd(4)%flo + ob(icmd)%hd(5)%flo) * cnv_m3  / time%step
        
      !! zero previous days hyds - current day is the hyd from yesterday so its set
      do kk = 1, time%step
        ob(icmd)%ts(iday_prev,kk) = hz
      end do

      if (qday > 1.e-9) then
          
        !! use unit hydrograph to compute subdaily flow hydrographs
        sumflo = 0.  !sum flow in case hydrograph exceeds max days 
        
        do ii = 1, time%step !loop for total time steps in a day
          itot = ii
          do ib = 1, itb(j)  !loop for number of steps in the unit hydrograph base time
            itot = itot + ib - 1
            if (itot > time%step) then
              iday = iday + 1
              if (iday > ihdmx) iday = 1
              itot = 1
            end if

            !! check to see if day has gone past the max allocated days- uh > 1 day
            if (iday <= ihdmx) then
              ob(icmd)%ts(iday,itot)%flo = ob(icmd)%ts(iday,itot)%flo + hhsurfq(j,ii) * uh(j,ib) * cnv_m3
              sumflo = sumflo + ob(icmd)%ts(iday,itot)%flo
            else
              !! adjust if flow exceeded max days
              rto = Max (1., ob(icmd)%hd(3)%flo / sumflo)
              do iadj = 1, itot - 1
                iday = iadj / time%step + 1
                istep = iadj - (iday - 1) * time%step
                ob(icmd)%ts(iday,itot)%flo = ob(icmd)%ts(iday,itot)%flo * rto
              end do
            end if
          end do
        end do
        
        sumflo_day = 0.
        iday = ob(icmd)%day_cur
        do istep = 1, time%step
          ob(icmd)%ts(iday,istep)%flo = ob(icmd)%ts(iday,istep)%flo + ssq
          sumflo_day = sumflo_day + ob(icmd)%ts(iday,istep)%flo
        end do

        !! set values for other routing variables - assume constant concentration
        !! storage locations set to zero are not currently used
        do ii = 1, time%step
          ratio = ob(icmd)%ts(iday,ii)%flo / sumflo_day
            if (ob(icmd)%hd(1)%flo > 0.) then
              ob(icmd)%ts(iday,ii)%temp = wtmp                                !!wtmp
              ob(icmd)%ts(iday,ii)%sed = ob(icmd)%hd(1)%sed * ratio           !!sedyld
              ob(icmd)%ts(iday,ii)%orgn = ob(icmd)%hd(1)%orgn * ratio         !!sedorgn
              ob(icmd)%ts(iday,ii)%sedp = ob(icmd)%hd(1)%sedp * ratio         !!sedorgp
              ob(icmd)%ts(iday,ii)%no3 = ob(icmd)%hd(1)%no3 * ratio           !!no3
              ob(icmd)%ts(iday,ii)%solp = ob(icmd)%hd(1)%solp * ratio         !!minp
              !ob(icmd)%ts(iday,ii)%psol = ob(icmd)%hd(1)%psol * ratio         !!sol pst
              !ob(icmd)%ts(iday,ii)%psor = ob(icmd)%hd(1)%psor * ratio         !!sorb pst
              ob(icmd)%ts(iday,ii)%chla = ob(icmd)%hd(1)%chla * ratio         !!chl_a
              ob(icmd)%ts(iday,ii)%nh3 = 0.                                   !! NH3
              ob(icmd)%ts(iday,ii)%no2 = 0.                                   !! NO2
              ob(icmd)%ts(iday,ii)%cbod = ob(icmd)%hd(1)%cbod * ratio         !!cbodu
              ob(icmd)%ts(iday,ii)%dox = ob(icmd)%hd(1)%dox * ratio           !!doxq & soxy 
            end if
          end do
        else
          !! no surface runoff on current day so zero hyds
          do istep = 1, time%step
            ob(icmd)%ts(iday,istep)%flo = ssq
          end do
        end if  ! qday > 0
      end if  ! time%step  > 0

      return   
      end subroutine hru_hyds