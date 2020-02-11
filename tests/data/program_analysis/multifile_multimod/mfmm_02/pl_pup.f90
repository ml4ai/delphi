      subroutine pl_pup

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine calculates plant phosphorus uptake

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units          |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    pup1(:)   |none           |1st shape parameter for plant P uptake
!!                                |equation
!!    pup2(:)   |none           |2st shape parameter for plant P uptake
!!                                |equation
!!    ihru        |none           |HRU number
!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~     
!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    uapd        |kg P/ha       |plant demand of phosphorus
!!    up2         |kg P/ha       |optimal plant phosphorus content
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Exp, Min
!!    SWAT: nuts

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use basin_module
      use organic_mineral_mass_module
      use hru_module, only : uapd, up2, pplnt, ihru, ipl, rto_solp, uptake
      use soil_module
      use plant_module

      implicit none

      integer :: icrop       !none      |land cover code
      integer :: j           !none      |hru number
      integer :: l           !none      |counter (soil layer)
      real :: root_depth     !mm        |root depth
      real :: uapl           !kg P/ha   |amount of phosphorus removed from layer
      real :: gx             !mm        |lowest depth in layer from which nitrogen
                             !          |may be removed
      real :: upmx           !kg P/ha   |maximum amount of phosphorus that can be
                             !          |removed from the soil layer
      real :: uobp           !none      |phosphorus uptake normalization parameter
                             !          |This variable normalizes the phosphorus
                             !          |uptake so that the model can easily verify
                             !          |that uptake from the different soil layers
                             !          |sums to 1.0
      
      j = ihru

      pcom(j)%plstr(ipl)%strsp = 1.
      if (uapd(ipl) < 1.e-6) return

      do l = 1, soil(j)%nly
        root_depth = amax1 (10., pcom(j)%plg(ipl)%root_dep)
        if (pcom(j)%plg(ipl)%root_dep <= soil(j)%phys(l)%d) then
          exit
        else
          gx = soil(j)%phys(l)%d
        end if

        upmx = uapd(ipl) * rto_solp * (1. - Exp(-bsn_prm%p_updis * gx / root_depth)) / uptake%p_norm
        uapl = Min(upmx - pplnt(j), soil1(j)%mp(l)%lab)
        pplnt(j) = pplnt(j) + uapl
        soil1(j)%mp(l)%lab = soil1(j)%mp(l)%lab - uapl
      end do
      if (pplnt(j) < 0.) pplnt(j) = 0.

      pl_mass(j)%tot(ipl)%p = pl_mass(j)%tot(ipl)%p + pplnt(j)
      pl_mass_up%p = pplnt(j)

      !! compute phosphorus stress
      call nuts(pl_mass(j)%tot(ipl)%p, up2(ipl), pcom(j)%plstr(ipl)%strsp)

      return
      end subroutine pl_pup