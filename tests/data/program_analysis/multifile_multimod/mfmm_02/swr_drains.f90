	subroutine swr_drains

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine finds the effective lateral hydraulic conductivity 
!!    and computes drainage or subirrigation flux

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    drain_co(:) |mm/day        |drainage coefficient 
!!    ddrain(:)   |mm            |depth of drain tube from the soil surface							 
!!    latksatf(:) |none          |multiplication factor to determine conk(j1,j) from sol_k(j1,j) for HRU
!!    pc(:)       |mm/hr         |pump capacity (default pump capacity = 1.042mm/hr or 25mm/day)
!!    sdrain(:)   |mm            |distance between two drain tubes or tiles
!!    sstmaxd(:)  |mm            |static maximum depressional storage; read from .sdr
!!    stmaxd(:)   |mm            |maximum surface depressional storage for the day in a given HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    qtile       |mm H2O        |drainage tile flow in soil profile for the day

!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ddarnp      |mm            |a variable used to indicate distance slightly less
!!                               |than ddrain. Used to prevent calculating subirrigation
!!                               |when water table is below drain bottom or when it is empty
!!    i           |none          |counter
!!    w           |mm            |thickness of saturated zone in layer considered
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: 
!!    SWAT: depstor

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use tiles_data_module
      use basin_module
      use hydrograph_module
      use climate_module, only : wst
      use hru_module, only : hru, ihru, wnan, stmaxd, sstmaxd, surfq, etday, inflpcp, &  
          mlyr, precip_eff, qtile, wt_shall
      use soil_module
      use time_module
      use reservoir_module
      
      implicit none

      integer :: j1              !none          |counter
      integer :: j               !none          |HRU number 
      integer :: m               !none          |counter
      real:: cone                !mm/hr         |effective saturated lateral conductivity - based
                                 !              |on water table depth and conk/sol_k of layers
      real:: depth               !mm            |actual depth from surface to impermeable layer 
      real:: dg                  !mm            |depth of soil layer
      real:: ad                  !              | 
      real:: ap                  !              |
      real:: hdrain              !mm            |equivalent depth from water surface in drain tube to
                                 !              |impermeable layer
      real:: gee                 !none          |factor -g- in Kirkham equation
      real:: e                   !              |
      real:: gee1                !              | 
      real:: gee2                !              | 
      real:: gee3                !              | 
      real:: pi	                 !              |
      real:: k2                  !              |
      real:: k3                  !              |
      real:: k4                  !              |
      real:: k5                  !              |
      real:: k6                  !              |
      real :: y1                 !mm            |dummy variable for dtwt 
      integer :: isdr            !              |
      real :: above              !mm            |depth of top layer considered
      integer :: nlayer          !none          |number of layers to be used to determine cone 
      real :: x                  !              |
      real :: sum                !              | 
      real :: deep               !mm            |total thickness of saturated zone
      real :: xx                 !              |
      real :: hdmin              !              |
      real :: storro             !mm            |surface storage that must b
                                 !              |can move to the tile drain tube
      real :: stor               !mm            |surface storage for the day in a given HRU
      real :: dflux              !mm/hr         |drainage flux
      real :: em                 !mm            |distance from water level in the drains to water table 
                                 !              |at midpoint: em is negative during subirrigation
      real :: ddranp             !              |
      real :: dot                !mm            |actual depth from impermeable layer to water level
                                 !              |above drain during subsurface irrigation
   
      !! initialize variables

      j = ihru
      isdr = hru(j)%tiledrain
      wnan = 0
      y1 = soil(j)%zmx - wt_shall 
      if (y1 > soil(j)%zmx) y1 = soil(j)%zmx
      above = 0.
      pi = 22./7.
      gee1 =0.

!! find number of soil layers 
      do j1 = 1, mlyr
        if(soil(j)%phys(j1)%d > 0.) nlayer = j1	    
      end do

!! find effective lateral hydraulic conductivity for the profile in hru j
      do j1 = 1, nlayer
        if(y1 > soil(j)%phys(j1)%d) then  
          wnan(j1) = 0.
        else
	    wnan(j1) = soil(j)%phys(j1)%d - y1  
	    x = soil(j)%phys(j1)%d -  above  
          if(wnan(j1) > x) wnan(j1) = x
	  end if
	  above = soil(j)%phys(j1)%d
      end do
      sum = 0.
      deep = 0.
      do j1=1,nlayer
        soil(j)%ly(j1)%conk = soil(j)%phys(j1)%k * sdr(isdr)%latksat !Daniel 2/26/09
        sum = sum + wnan(j1) * soil(j)%ly(j1)%conk
        deep = deep + wnan(j1)
      end do
      if((deep <= 0.001).or.(sum <= 0.001)) then
        sum = 0.
        deep = 0.001
        do j1=1,nlayer
		  sum = sum + soil(j)%ly(j1)%conk * soil(j)%phys(j1)%thick !Daniel 10/09/07
		  deep = deep + dg   !Daniel 10/09/07
        end do
	  cone=sum/deep
	else
	  cone=sum/deep
      end if

      !!	calculate parameters hdrain and gee1
      ad = soil(j)%zmx - hru(j)%lumv%sdr_dep
      ad = Max (10., ad)
	  ap = 3.55 - ((1.6 * ad) / sdr(isdr)%dist) + 2 *                   &
                                           ((2 / sdr(isdr)%dist)**2)
	  if (ad / sdr(isdr)%dist < 0.3) then
        hdrain= ad / (1 + ((ad / sdr(isdr)%dist) * (((8 / pi) *       &
     	    Log(ad / sdr(isdr)%radius) - ap))))
      else
        hdrain = ad
          !hdrain = (sdr(isdr)%dist * pi) / (8 * ((log(sdr(isdr)%dist /  &
          !         sdr(isdr)%radius)/ log(e)) - 1.15))
	  end if
      !! calculate Kirkham G-Factor, gee
        k2 = tan((pi * ((2. * ad) -sdr(isdr)%radius)) / (4. * soil(j)%zmx))
        k3 = tan((pi * sdr(isdr)%radius) / (4. * soil(j)%zmx))
      do m=1,2
         k4 = (pi * m * sdr(isdr)%dist) / (2. * soil(j)%zmx)
         k5 = (pi *sdr(isdr)%radius) / (2. * soil(j)%zmx)
         k6 = (pi * (2. * ad - sdr(isdr)%radius)) / (2. * soil(j)%zmx)
         gee2 = (cosh(k4) + cos(k5)) / (cosh(k4) - cos(k5))
         gee3 = (cosh(k4) - cos(k6)) / (cosh(k4) + cos(k6))
         gee1 = gee1 + Log(gee2 * gee3)
      end do
       xx = k2 / k3
       if (xx < 1.) then
         gee = 1.
       else
         gee = 2 * Log(k2 / k3) + 2 * gee1
       end if
       if (gee < 1.) gee = 1.
       if (gee > 12.) gee = 12.

      !! calculate drainage and subirrigation flux section
      !	drainage flux for ponded surface
      depth = hru(j)%lumv%sdr_dep + hdrain
      hdmin = depth - hru(j)%lumv%sdr_dep
      if (bsn_cc%smax == 1) then
        call swr_depstor ! dynamic stmaxd(j): compute current HRU stmaxd based 
	           ! on cumulative rainfall and cum. intensity
	else
	  stmaxd(j) = sstmaxd(j)
	end if 
      storro = 0.2 * stmaxd(j) !surface storage that must be filled before surface
                   !water can move to the tile drain tube
      !! Determine surface storage for the day in a given HRU (stor)
        !initialize stor on the beginning day of simulation, Daniel 9/20/2007
      if (time%yrs == 1 .and. time%day == time%day_start) then 
        stor= 0.
      end if
      if (wet_ob(j)%area_ha <= 0.) then ! determine stor
        stor = precip_eff - inflpcp - etday !Daniel 10/05/07
        if(surfq(j) > 0.0) stor = stmaxd(j)
      else
        stor = wet(j)%flo / (wet_ob(j)%area_ha * 1000.)
      endif
      if(hdrain < hdmin) hdrain=hdmin
      if((stor > storro).and.(y1 < 5.0)) then
        dflux= (12.56637*24.0*cone*(depth-hdrain+stor))/                   &
            (gee*sdr(isdr)%dist) !eq.10
        if(dflux > sdr(isdr)%drain_co) dflux = sdr(isdr)%drain_co !eq.11
      else
!	subirrigation flux 
        em=depth-y1-hdrain
        if(em < -1.0) then
!!          ddranp=ddrain(j)-1.0
          ddranp = hru(j)%lumv%sdr_dep - 1.0
          dot = hdrain + soil(j)%zmx - depth
          dflux=4.0*24.0*cone*em*hdrain*(2.0+em/dot)/sdr(isdr)%dist**2 
          if((depth-hdrain) >= ddranp) dflux=0.
          if(abs(dflux) > sdr(isdr)%pumpcap) then
            dflux = - sdr(isdr)%pumpcap * 24.0 
          end if
!	drainage flux - for WT below the surface and for ponded depths < storro (S1)
        else
        dflux=4.0*24.0*cone*em*(2.0*hdrain+em) / sdr(isdr)%dist**2 !eq.5
        if(dflux > sdr(isdr)%drain_co) dflux = sdr(isdr)%drain_co !eq.11
        if(dflux < 0.) dflux=0.
        if(em < 0.) dflux=0.
        end if
	end if
 	qtile = dflux

       return
       end subroutine swr_drains