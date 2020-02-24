      subroutine swr_satexcess(j1)
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine is the master soil percolation component.

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    sep         |mm H2O        |micropore percolation from soil layer
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Max
!!    SWAT: percmacro, percmicro

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use septic_data_module
      use hru_module, only : hru, ihru, cbodu, surfq, surqno3, surqsolp, sep_tsincefail, i_sep,  &
        isep, qday, sepday
      use soil_module
      use hydrograph_module
      use basin_module
      use organic_mineral_mass_module
      
      implicit none

      integer :: j                 !none          |HRU number
      integer :: j1                !none          |counter
      integer :: ii                !none          |counter
      integer :: isp               !              | 
      real:: ul_excess             !              |
      real:: qlyr                  !              |
      real:: pormm                 !mm            |porosity in mm depth 
      real:: rtof                  !none          |weighting factor used to partition the 
                                   !              |organic N & P concentration of septic effluent
                                   !              |between the fresh organic and the stable organic pools
      real :: qvol                 !              |     
      real :: xx                   !              |
      integer :: jj                !              |
      integer :: l                 !              | 
      integer :: nn                !none          |number of soil layers
      integer :: ly                !none          |counter

      j = ihru
      isp = sep(isep)%typ 	   !! J.Jeong 3/09/09
      rtof = 0.5

 	if (sep(isep)%opt == 2 .and. j1 == i_sep(j)) then
	  
	  ii = j1 
	  qlyr = soil(j)%phys(ii)%st
	  
	  ! distribute excess STE to upper soil layers 
	  do while (qlyr > 0 .and. ii > 1)
		 ! distribute STE to soil layers above biozone layer
        if (soil(j)%phys(ii)%st > soil(j)%phys(ii)%ul) then
	      qlyr = soil(j)%phys(ii)%st - soil(j)%phys(ii)%ul 	! excess water moving to upper layer
	      soil(j)%phys(ii)%st = soil(j)%phys(ii)%ul  ! layer saturated
	      soil(j)%phys(ii-1)%st = soil(j)%phys(ii-1)%st + qlyr ! add excess water to upper layer
	    else 
	      qlyr = 0.
	    endif
	  
	    ! Add surface ponding to the 10mm top layer when the top soil layer is saturated
		 !  and surface ponding occurs.
		 if (ii == 2) then
	     qlyr = soil(j)%phys(1)%st - soil(j)%phys(1)%ul
	     ! excess water makes surface runoff
	     if (qlyr > 0) then
            soil(j)%phys(1)%st = soil(j)%phys(1)%ul
            !add septic effluent cbod (mg/l) concentration to HRU runoff, Jaehak Jeong 2016
            cbodu(j) = (cbodu(j) * surfq(j) + sepdb(sep(isep)%typ)%bodconcs * qlyr) / (qday + qlyr) 
            surfq(j) = surfq(j) + qlyr 
            qvol = qlyr * hru(j)%area_ha * 10.
            ! nutrients in surface runoff
            xx = qvol / hru(j)%area_ha / 1000.
            surqno3(j) = surqno3(j) + xx * (sepdb(sep(isep)%typ)%no3concs + sepdb(sep(isep)%typ)%no2concs) 
            surqsolp(j) =  surqsolp(j) +  xx * sepdb(sep(isep)%typ)%minps 
            
            ! Initiate counting the number of days the system fails and makes surface ponding of STE
            if(sep_tsincefail(j)==0) sep_tsincefail(j) = 1
	     endif
	     qlyr = 0.
           !nutrients in the first 10mm layer
		   qvol = soil(j)%phys(1)%st * hru(j)%area_ha * 10.
		   xx = qvol / hru(j)%area_ha / 1000.
           rsd1(j)%mn%no3 = rsd1(j)%mn%no3 + xx * (sepdb(sep(isep)%typ)%no3concs + sepdb(sep(isep)%typ)%no2concs)  
           rsd1(j)%mn%nh4 = rsd1(j)%mn%nh4 + xx * sepdb(sep(isep)%typ)%nh4concs                  
           soil1(j)%hp(1)%n = soil1(j)%hp(1)%n + xx * sepdb(sep(isep)%typ)%orgnconcs * rtof
           rsd1(j)%tot(1)%n = rsd1(j)%tot(1)%n + xx * sepdb(sep(isep)%typ)%orgnconcs * (1.-rtof)
           soil1(j)%hp(1)%p = soil1(j)%hp(1)%p + xx * sepdb(sep(isep)%typ)%orgps * rtof
           rsd1(j)%tot(1)%p = rsd1(j)%tot(1)%p + xx * sepdb(sep(isep)%typ)%orgps*(1.-rtof)
           rsd1(j)%mp%lab = rsd1(j)%mp%lab + xx * sepdb(sep(isep)%typ)%minps  
		 endif

         ! volume water in the current layer: m^3
         qvol = soil(j)%phys(ii)%st * hru(j)%area_ha * 10. 
         
		 ! add nutrient to soil layer
		 xx = qvol / hru(j)%area_ha / 1000.
         soil1(j)%mn(ii)%no3 = soil1(j)%mn(ii)%no3 + xx * sepdb(sep(isep)%typ)%no3concs + sepdb(sep(isep)%typ)%no2concs
	   soil1(j)%mn(ii)%nh4 = soil1(j)%mn(ii)%nh4 + xx * sepdb(sep(isep)%typ)%nh4concs
	   soil1(j)%hp(ii)%n = soil1(j)%hp(ii)%n + xx * sepdb(sep(isep)%typ)%orgnconcs*rtof
       soil1(j)%tot(ii)%n = soil1(j)%tot(ii)%n + xx * sepdb(sep(isep)%typ)%orgnconcs * (1.-rtof)
       soil1(j)%hp(ii)%p = soil1(j)%hp(ii)%p + xx * sepdb(sep(isep)%typ)%orgps * rtof
	   soil1(j)%tot(ii)%p = soil1(j)%tot(ii)%p + xx * sepdb(sep(isep)%typ)%orgps*(1.-rtof)
       soil1(j)%mp(ii)%lab = soil1(jj)%mp(l)%lab + xx * sepdb(sep(isep)%typ)%minps

	    ii = ii - 1
	  end do
	endif
	       
      if (sep(isep)%opt == 0) then
      if (j1 < soil(j)%nly) then
        if (soil(j)%phys(j1)%st - soil(j)%phys(j1)%ul > 1.e-4) then
          sepday = (soil(j)%phys(j1)%st - soil(j)%phys(j1)%ul)
          soil(j)%phys(j1)%st = soil(j)%phys(j1)%ul
          soil(j)%phys(j1+1)%st = soil(j)%phys(j1+1)%st + sepday
        end if
      else

        if (soil(j)%phys(j1)%st - soil(j)%phys(j1)%ul > 1.e-4) then
          ul_excess = soil(j)%phys(j1)%st - soil(j)%phys(j1)%ul
          soil(j)%phys(j1)%st = soil(j)%phys(j1)%ul
          nn = soil(j)%nly
          do ly = nn - 1, 1, -1
            soil(j)%phys(ly)%st = soil(j)%phys(ly)%st + ul_excess
            if (soil(j)%phys(ly)%st > soil(j)%phys(ly)%ul) then
              ul_excess = soil(j)%phys(ly)%st - soil(j)%phys(ly)%ul
              soil(j)%phys(ly)%st = soil(j)%phys(ly)%ul
            else
              ul_excess = 0.
              exit
            end if
            if (ly == 1 .and. ul_excess > 0.) then
              !! add ul_excess to depressional storage and then to surfq
              wet(j)%flo = wet(j)%flo + ul_excess
            end if
          end do
          !compute tile flow again after saturation redistribution
        end if
      end if
      end if

      return
      end subroutine swr_satexcess