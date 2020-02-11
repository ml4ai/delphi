      subroutine sq_volq

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    Call subroutines to calculate the current day"s CN for the HRU and
!!    to calculate surface runoff

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    SWAT: surq_daycn, surq_breakcn, surq_greenampt, dir_rnff
!!    SWAT: surq_hourly

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use basin_module
      use time_module
      
      implicit none

!! Compute surface runoff for day
      if (time%step == 0) then
          call sq_daycn
      else
        call sq_greenampt
      end if

      return
      end subroutine sq_volq