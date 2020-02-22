      subroutine soil_nutcarb_write

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine writes daily HRU output to the output.hru file

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name          |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    etday         |mm H2O        |actual amount of evapotranspiration that
!!                                 |occurs on day in HRU
!!    gw_q(:)       |mm H2O        |groundwater contribution to streamflow from
!!                                 |HRU on current day
!!    hru_ha(:)     |ha            |area of HRU in hectares
!!    hru_km(:)     |km^2          |area of HRU in square kilometers
!!    ihru          |none          |HRU number
!!    rchrg(:)      |mm H2O        |amount of water recharging both aquifers on
!!                                 |current day in HRU
!!    surfq(:)      |mm H2O        |surface runoff generated on day in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ii          |none          |counter
!!    j           |none          |HRU number
!!    sb          |none          |subbasin number
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only : etday, ihru, tillage_factor
      use soil_module
      use time_module
      use basin_module
      use organic_mineral_mass_module
      use hydrograph_module
      
      implicit none
      
      integer :: ly           !none        |counter
      
      ! sum the output for the entire soil profile
      do ihru = 1, sp_ob%hru
        soil_prof_tot = soil_org_z
        soil_prof_mn = soil_mn_z
        soil_prof_mp = soil_mp_z
        do ly = 1, soil(ihru)%nly
          soil_prof_mn = soil_prof_mn + soil1(ihru)%mn(ly)
          soil_prof_mp = soil_prof_mp + soil1(ihru)%mp(ly)
          soil_prof_tot = soil_prof_tot + soil1(ihru)%tot(ly)
          soil_prof_str = soil_prof_str + soil1(ihru)%str(ly)
          soil_prof_lig = soil_prof_lig + soil1(ihru)%lig(ly)
          soil_prof_meta = soil_prof_meta + soil1(ihru)%meta(ly)
          soil_prof_man = soil_prof_man + soil1(ihru)%man(ly)
          soil_prof_hs = soil_prof_hs + soil1(ihru)%hs(ly)
          soil_prof_hp = soil_prof_hp + soil1(ihru)%hp(ly)
          soil_prof_microb = soil_prof_microb + soil1(ihru)%microb(ly)
          soil_prof_water = soil_prof_water + soil1(ihru)%water(ly)
        end do
        
        ! write all carbon, organic n and p, and mineral n and p for the soil profile
        write (2610,*) time%day, time%yrc, ihru, soil_prof_mn, soil_prof_mp, soil_prof_tot, soil_prof_str,  &
          soil_prof_lig, soil_prof_meta, soil_prof_man, soil_prof_hs, soil_prof_hp, soil_prof_microb,       &
          soil_prof_water
      end do

      return

      end subroutine soil_nutcarb_write