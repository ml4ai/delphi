      subroutine unit_hyd

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    This subroutine computes variables related to the watershed hydrology:
!!    the time of concentration for the subbasins, lagged surface runoff,
!!    the coefficient for the peak runoff rate equation, and lateral flow travel
!!    time.

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~1
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    ch_n(1,:)   |none          |Manning"s "n" value for the tributary channels
!!    ch_s(1,:)   |m/m           |average slope of tributary channels
!!    gdrain(:)   |hrs           |drain tile lag time: the amount of time
!!                               |between the transfer of water from the soil
!!                               |to the drain tile and the release of the
!!                               |water from the drain tile to the reach.
!!    hru_km(:)   |km2           |area of HRU in square kilometers
!!    lat_ttime(:)|days          |lateral flow travel time
!!    slsoil(:)   |m             |slope length for lateral subsurface flow
!!    slsubbsn(:) |m             |average slope length for subbasin
!!    tconc(:)     |hr           |time of concentration
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 


!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    lat_ttime(:)|none          |Exponential of the lateral flow travel time
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    l           |none          |counter
!!    scmx        |mm/hr         |maximum soil hydraulic conductivity
!!    t_ch        |hr            |time for flow entering the farthest upstream 
!!                               |channel to reach the subbasin outlet
!!    xx          |none          |variable to hold calculation result
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~  
!!    SWAT: Ttcoef

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only : itb, tconc, uh
    
      use climate_module
      use basin_module
      use channel_module
      use time_module
      use hydrograph_module, only : sp_ob
      
      implicit none
      
      integer :: j             !none          |counter
      real :: ql               !              | 
      real :: sumq             !              |
      real :: tb               !              |
      real :: tp               !              |
      integer :: int           !              |
      integer :: i             !none          |counter
      real :: xi               !              |
      real :: q                !              |
      integer :: max           !              |
      integer :: ij            !              |
 
!!    compute unit hydrograph for computing subbasin hydrograph from direct runoff
      do j = 1, sp_ob%hru
        ql = 0.
        sumq = 0.
        tb = .5 + .6 * tconc(j) + bsn_prm%tb_adj    !baseflow time, hr
        if (tb > 48.) tb = 48.			   !maximum 48hrs
        tp = .375 * tb                       ! time to peak flow
	  !! convert to time step (from hr), J.Jeong March 2009
	  tb = ceiling(tb * 60./ real(time%dtm))
	  tp = int(tp * 60./ real(time%dtm))         
	  
	  if(tp==0) tp = 1
	  if(tb==tp) tb = tb + 1
	  itb(j) = int(tb) 
        
	  ! Triangular Unit Hydrograph
	  if (bsn_cc%uhyd == 0) then
	    do i = 1, itb(j)
          xi = float(i)
 	      if (xi < tp) then           !! rising limb of hydrograph
            q = xi / tp
          else                        !! falling limb of hydrograph
            q = (tb - xi) / (tb - tp)
          end if
          q = Max(0.,q)
          uh(j,i) = (q + ql) / 2.
          ql = q
          sumq = sumq + uh(j,i)
        end do
          
		do i = 1, itb(j)
            uh(j,i) = uh(j,i) / sumq
        end do
	  
	  ! Gamma Function Unit Hydrograph
	  elseif (bsn_cc%uhyd == 1) then
          i = 1; q = 1.
		do while (q > 0.0001)
            xi = float(i)
		   q = (xi / tp) ** bsn_prm%uhalpha * exp((1.- xi / tp) *             &    
                          bsn_prm%uhalpha)
            q = Max(0.,q)
            uh(j,i) = (q + ql) / 2.
            ql = q
            sumq = sumq + uh(j,i)
	      i = i + 1
	      if (i>3.*time%step) exit
	    end do
	    itb(ij) = i - 1
          do i = 1, itb(j)
            uh(j,i) = uh(j,i) / sumq
          end do
	  endif 

      end do

      return
      end subroutine unit_hyd