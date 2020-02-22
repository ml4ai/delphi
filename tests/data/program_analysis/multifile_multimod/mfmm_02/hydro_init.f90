      subroutine hydro_init 

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    This subroutine computes variables related to the watershed hydrology:
!!    the time of concentration for the subbasins, lagged surface runoff,
!!    the coefficient for the peak runoff rate equation, and lateral flow travel
!!    time.

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~1
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    ch_n(1,:)   |none          |Manning's "n" value for the tributary channels
!!    ch_s(1,:)   |m/m           |average slope of tributary channels
!!    gdrain(:)   |hrs           |drain tile lag time: the amount of time
!!                               |between the transfer of water from the soil
!!                               |to the drain tile and the release of the
!!                               |water from the drain tile to the reach.
!!    hru_km(:)   |km2           |area of HRU in square kilometers
!!    lat_ttime(:)|days          |lateral flow travel time
!!    slsoil(:)   |m             |slope length for lateral subsurface flow
!!    slsubbsn(:) |m             |average slope length for subbasin
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 


!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    lat_ttime(:)|none          |Exponential of the lateral flow travel time
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~  
!!    SWAT: Ttcoef

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only : dormhr, hru, i_sep, isep, isep_ly, iseptic
      use soil_module
      use plant_module
      use climate_module
      use septic_data_module
      use plant_data_module
      use tiles_data_module
      use pesticide_data_module
      use basin_module
      use channel_module
      use time_module
      use organic_mineral_mass_module
      use hydrograph_module, only : sp_ob, ob
      use constituent_mass_module
      
      implicit none

      integer :: j              !none          |counter            
      integer :: l              !none          |counter
      integer :: idp            !              | 
      real :: t_ch              !hr            |time for flow entering the farthest upstream 
                                !              |channel to reach the subbasin outlet
      real :: scmx              !mm/hr         |maximum soil hydraulic conductivity
      real :: xx                !none          |variable to hold calculation result
      real :: tsoil             !              | 
      integer :: iob            !              | 
      integer :: iwst           !              | 
      integer :: iwgn           !              | 
      real :: sffc              !              | 
      integer :: nly            !none          |end of loop
      integer :: k              !none          |counter
      real :: plt_zmx           !              |
      integer :: ipl            !none          |counter
      real :: plt_zmxp          !              | 
      integer :: max            !              |
      integer :: min            !              |
      real :: dep_new           !              |
      integer :: jj             !none          |counter
      real :: solpst            !              |
      integer :: n              !              |
      real :: wt1               !none          |conversion factor to convert kg/ha to g/t(ppm) 
      integer :: ly             !none          |counter   
      integer :: isdr           !none          |conversion factor to convert kg/ha to g/t(ppm)
      real :: sd
      real :: dd
      real :: sdlat
      real :: h 
      real :: daylength

      do j = 1, sp_ob%hru
       iob = hru(j)%obj_no
       iwst = ob(iob)%wst
       iwgn = wst(iwst)%wco%wgn
       
       hru(j)%lumv%usle_mult = soil(j)%phys(1)%rock * soil(j)%ly(1)%usle_k *       &
                                 hru(j)%lumv%usle_p * hru(j)%lumv%usle_ls * 11.8

      tsoil = (wgn(iwgn)%tmpmx(12) + wgn(iwgn)%tmpmx(12)) / 2.
      !! should be beginning month of simulation and not 12 (December)

!!    set fraction of field capacity in soil
      if (bsn_prm%ffcb <= 0.) then
       sffc = wgn_pms(iwgn)%pcp_an / (wgn_pms(iwgn)%pcp_an + Exp(9.043 -   &
                                     .002135 * wgn_pms(iwgn)%pcp_an))
                         !!S-curve equation Jeff made up.
      else
        sffc = bsn_prm%ffcb
      end if
      
      !! set initial soil water and temperature for each layer
      nly = soil(j)%nly
      soil(j)%sw = 0.
      do k = 1, nly
        soil(j)%phys(k)%tmp = tsoil
        soil(j)%phys(k)%st = sffc * soil(j)%phys(k)%fc
        soil(j)%sw = soil(j)%sw + soil(j)%phys(k)%st
      end do
      
      !! set day length threshold for dormancy and initial dormancy
      dormhr(j) = wgn_pms(iwgn)%daylth
      sd = Asin(.4 * Sin((Real(time%day) - 82.) / 58.09))  !!365/2pi = 58.09
      dd = 1.0 + 0.033 * Cos(Real(time%day) / 58.09)
      sdlat = -wgn_pms(iwgn)%latsin * Tan(sd) / wgn_pms(iwgn)%latcos
      if (sdlat > 1.) then    !! sdlat will be >= 1. if latitude exceeds +/- 66.5 deg in winter
        h = 0.
      elseif (sdlat >= -1.) then
        h = Acos(sdlat)
      else
        h = 3.1416         !! latitude exceeds +/- 66.5 deg in summer
      endif 
      daylength = 7.6394 * h
      do ipl = 1, pcom(j)%npl
        if (pcom(j)%plcur(ipl)%gro == "y" .and. daylength - dormhr(j) < wgn_pms(iwgn)%daylmn) then
          pcom(j)%plcur(ipl)%idorm = "y"
        else
          pcom(j)%plcur(ipl)%idorm = "n"
        end if
      end do

!!    set maximum depth in soil to maximum rooting depth of plant
      soil(j)%zmx = soil(j)%phys(nly)%d
      
!! create a biozone layer in septic HRUs
      isep = iseptic(j)
      if (sep(isep)%opt  /= 0) then 
	 if (sep(isep)%z + sep(isep)%thk > soil(j)%phys(nly)%d) then
	   if (soil(j)%phys(nly)%d > sep(isep)%thk + 10.) then !min. soil thickness for biozone layer (10mm top+biozone layer thickness)
	      sep(isep)%z = soil(j)%phys(nly)%d - sep(isep)%thk
	   else
	      sep(isep)%z = soil(j)%phys(nly)%d
	      soil(j)%phys(nly)%d = soil(j)%phys(nly)%d + sep(isep)%thk
	   endif
       endif 
       if (sep(isep)%z > 0.) then 
         call layersplit (sep(isep)%z)
         dep_new = sep(isep)%z + sep(isep)%thk
         call layersplit (dep_new)  
         i_sep(j) = isep_ly
       endif    
      endif

!!    compute lateral flow travel time
        if (hru(j)%hyd%lat_ttime <= 0.) then
            scmx = 0.
            do l = 1, soil(j)%nly
              if (soil(j)%phys(l)%k > scmx) then
                scmx = soil(j)%phys(l)%k
              endif
            end do
            !! unit conversion:
            !! xx = m/(mm/h) * 1000.(mm/m)/24.(h/d) / 4.
            xx = 0.
            xx = 10.4 * hru(j)%topo%lat_len / scmx
            if (xx < 1.) xx = 1.
            hru(j)%hyd%lat_ttime = 1. - Exp(-1./xx)
        else
          hru(j)%hyd%lat_ttime = 1. - Exp(-1. / hru(j)%hyd%lat_ttime)
        end if

        isdr = hru(j)%tiledrain
        if (hru(j)%lumv%ldrain > 0 .and. sdr(isdr)%lag > 0.01) then
          hru(j)%lumv%tile_ttime = 1. - Exp(-24. / sdr(isdr)%lag)
        else
          hru(j)%lumv%tile_ttime = 0.
        end if
      end do

      return
      end subroutine hydro_init