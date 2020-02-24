      subroutine nut_nlch
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine simulates the loss of nitrate via surface runoff, 
!!    lateral flow, tile flow, and percolation out of the profile

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    nperco      |none          |nitrate percolation coefficient (0-1)
!!                               |0:concentration of nitrate in surface runoff
!!                               |  is zero
!!                               |1:surface runoff has same concentration of
!!                               |  nitrate as percolate
!!    surfq(:)    |mm H2O        |surface runoff generated on day in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Exp, Max, Min

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use basin_module
      use organic_mineral_mass_module
      use hru_module, only : hru, latno3, percn, surqno3, tileno3, surfq, ihru, qtile
      use soil_module
      
      implicit none 

      integer :: j         !none          |HRU number
      integer :: jj        !none          |counter   
      real :: sro          !mm H2O        |surface runoff 
      real :: ssfnlyr      !kg N/ha       |nitrate transported in lateral flow from layer
      real :: percnlyr     !kg N/ha       |nitrate leached to next lower layer with
                           !              |percolation
      real :: vv           !mm H2O        |water mixing with nutrient in layer
      real :: vno3         !              |
      real :: co           !kg N/mm       |concentration of nitrate in solution
      real :: cosurf       !kg N/mm       |concentration of nitrate in surface runoff 
      real :: nloss        !frac          |nloss based on half life
      real :: ww           !varies        |variable to hold intermediate calculation
                           !              |result

      j = ihru

      percnlyr = 0.

      do jj = 1, soil(j)%nly

        !! add nitrate leached from layer above
        soil1(j)%mn(jj)%no3 = soil1(j)%mn(jj)%no3 + percnlyr
	  if (soil1(j)%mn(jj)%no3 < 1.e-6) soil1(j)%mn(jj)%no3 = 0.0

        !! determine concentration of nitrate in mobile water
        if (jj == 1) then
          sro = surfq(j)
        else
          sro = 0.
        end if
        vv = soil(j)%ly(jj)%prk + sro + soil(j)%ly(jj)%flat + 1.e-10
        if (hru(j)%lumv%ldrain == jj) vv = vv + qtile
        ww = -vv / ((1. - soil(j)%anion_excl) * soil(j)%phys(jj)%ul)
        vno3 = soil1(j)%mn(jj)%no3 * (1. - Exp(ww))
        co = Max(vno3 / vv, 0.)

        !! calculate nitrate in surface runoff
        cosurf = bsn_prm%nperco * co
        if (jj == 1) then
          surqno3(j) = surfq(j) * cosurf
          surqno3(j) = Min(surqno3(j), soil1(j)%mn(jj)%no3)
          soil1(j)%mn(jj)%no3 = soil1(j)%mn(jj)%no3 - surqno3(j)
        endif
        !Daniel 1/2012    
        !! calculate nitrate in tile flow 
        if (hru(j)%lumv%ldrain == jj) then
           ! tileno3(j) = bsn_prm%nperco * co * qtile     !Daniel 1/2012
           tileno3(j) = co * qtile     !Daniel 1/2012
          tileno3(j) = Min(tileno3(j), soil1(j)%mn(jj)%no3)
          soil1(j)%mn(jj)%no3 = soil1(j)%mn(jj)%no3 - tileno3(j)
        end if
        !Daniel 1/2012                  

        !! calculate nitrate in lateral flow
        if (jj == 1) then
          ssfnlyr = cosurf * soil(j)%ly(jj)%flat
        else
          ssfnlyr = co * soil(j)%ly(jj)%flat
        end if
        ssfnlyr = Min(ssfnlyr, soil1(j)%mn(jj)%no3)
        latno3(j) = latno3(j) + ssfnlyr
        soil1(j)%mn(jj)%no3 = soil1(j)%mn(jj)%no3 - ssfnlyr

        !! calculate nitrate in percolate
        percnlyr = co * soil(j)%ly(jj)%prk
        percnlyr = Min(percnlyr, soil1(j)%mn(jj)%no3)
        soil1(j)%mn(jj)%no3 = soil1(j)%mn(jj)%no3 - percnlyr
      end do

      !! calculate nitrate leaching from soil profile
      percn(j) = percnlyr


      nloss = (2.18 * hru(j)%topo%dis_stream - 8.63) / 100.
      nloss = Max(0.,nloss)
      nloss = Amin1(1.,nloss)
      latno3(j) = (1. - nloss) * latno3(j)

      return
      end subroutine nut_nlch