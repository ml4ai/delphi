      subroutine rls_routetile (iob)
      
!!    ~ ~ ~ PURPOSE ~ ~ ~

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
!!
      use hru_module, only : ihru, hru
      use soil_module
      use hydrograph_module
      use organic_mineral_mass_module
      
      implicit none
 
      integer, intent (in) :: iob   !           |object number
      integer :: j                  !           |hru number
      integer :: lyr                !           |tile soil layer 

      j = ihru

      !! add tile inflow and nitrate to soil layer the tile is in
      !! if exceeds saturation, it will be redistributed in swr_satexcess
      lyr = hru(j)%lumv%ldrain
      soil(j)%phys(lyr)%st = soil(j)%phys(lyr)%st + ob(iob)%hin_til%flo
      soil1(j)%mn(lyr)%no3 = soil1(j)%mn(lyr)%no3 + ob(iob)%hin_til%no3

      return
      end subroutine rls_routetile