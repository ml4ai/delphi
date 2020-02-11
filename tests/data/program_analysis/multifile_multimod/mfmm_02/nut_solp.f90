      subroutine nut_solp
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine calculates the amount of phosphorus lost from the soil
!!    profile in runoff and the movement of soluble phosphorus from the first
!!    to the second layer via percolation

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name          |units        |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ihru          |none         |HRU number
!!    surfq(:)      |mm H2O       |surface runoff generated on day in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Min, Max

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use basin_module
      use organic_mineral_mass_module
      use hru_module, only : hru, surqsolp, percp, surfq, i_sep, ihru, qtile 
      use soil_module
      
      implicit none 

      integer :: j           !none          |HRU number
      real :: xx             !none          |variable to hold intermediate calculation
                             !              |result
      real :: vap            !kg P/ha       |amount of P leached from soil layer 
      integer :: ii          !none          |counter 

      j = ihru

!! compute soluble P lost in surface runoff
      xx = soil(j)%phys(1)%bd * soil(j)%phys(1)%d * bsn_prm%phoskd
      surqsolp(j) = rsd1(j)%mp%lab  * surfq(j) / (xx + 1.)   !dont merge
        !!units ==> surqsolp = [kg/ha * mm] / [t/m^3 * mm * m^3/t] = kg/ha
      surqsolp(j) = Min(surqsolp(j), rsd1(j)%mp%lab)
      surqsolp(j) = Max(surqsolp(j), 0.)
      rsd1(j)%mp%lab = rsd1(j)%mp%lab - surqsolp(j)


!! compute soluble P leaching
      vap = rsd1(j)%mp%lab * soil(j)%ly(1)%prk /                   &
        ((soil(j)%phys(1)%conv_wt/ 1000.) * bsn_prm%pperco + .1)   !dont merge
      vap = Min(vap, .5 * rsd1(j)%mp%lab)
      rsd1(j)%mp%lab = rsd1(j)%mp%lab - vap
      if (soil(j)%nly >= 2) then
        soil1(j)%mp(2)%lab = soil1(j)%mp(2)%lab + vap
      end if
   
      do ii = 2, soil(j)%nly-1
        vap = 0.
	   if (ii/=i_sep(j)) then
         vap = soil1(j)%mp(ii)%lab * soil(j)%ly(ii)%prk /                &
          ((soil(j)%phys(ii)%conv_wt / 1000.) * bsn_prm%pperco + .1)
	     vap = Min(vap, .2 * soil1(j)%mp(ii)%lab)
	     soil1(j)%mp(ii)%lab = soil1(j)%mp(ii)%lab - vap
	     soil1(j)%mp(ii+1)%lab = soil1(j)%mp(ii+1)%lab + vap
         if (ii == hru(j)%lumv%ldrain) then
           vap = soil1(j)%mp(ii)%lab * qtile /                        &
              (soil(j)%phys(ii)%conv_wt / 1000. * bsn_prm%pperco + .1)  !dont merge
           vap = Min(vap, soil1(j)%mp(ii)%lab)
           soil1(j)%mp(ii)%lab = soil1(j)%mp(ii)%lab - vap
         endif
	   endif
	 end do
	 percp(j) = vap
     
      return
      end subroutine nut_solp