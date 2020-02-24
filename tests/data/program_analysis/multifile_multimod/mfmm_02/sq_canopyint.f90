      subroutine sq_canopyint

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine computes canopy interception of rainfall
!!    used for methods other than curve number

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    blai(:)     |none          |maximum (potential) leaf area index
!!    canmx(:)    |mm H2O        |maximum canopy storage
!!    canstor(:)  |mm H2O        |amount of water held in canopy storage
!!    wst(:)%weat%ts(:) |mm H2O        |precipitation in time step for HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    canstor(:)  |mm H2O        |amount of water held in canopy storage
!!    wst(:)%weat%ts(:) |mm H2O        |precipitation reaching soil surface in
!!                               |time step
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use basin_module
      use time_module
      use climate_module, only : wst
      use hru_module, only : hru, canstor, blai_com, ihru, precipday
      use plant_module
      
      implicit none

      real :: xx                 !mm H2O        |precipitation prior to canopy interception 
      integer :: j               !none          |HRU number
      integer :: ii              !none          |counter
      real ::canmxl              !mm H2O        |maximum canopy storage at current day's leaf
                                 !              |area
      real :: canstori           !mm H2O        |initial canopy storage water content 
      integer :: iwst            !none          |counter
      real :: precip_eff         !mm            |daily effective precip for runoff calculations = precipday + ls_overq + snomlt - canstor
                                 !     |precip_eff = precipday + ls_overq - snofall + snomlt - canstor
      
      j = ihru

      if (blai_com(j) < 0.001) return

      if (time%step > 0) then
          canstori = canstor(j)
          canmxl = hru(j)%hyd%canmx * pcom(j)%lai_sum / blai_com(j)
          do ii = 2, time%step+1
            xx = 0.
            xx = wst(iwst)%weat%ts(ii)
            wst(iwst)%weat%ts(ii) = wst(iwst)%weat%ts(ii) - (canmxl - canstor(j))

            if (wst(iwst)%weat%ts(ii) < 0.) then
              canstor(j) = canstor(j) + xx
              wst(iwst)%weat%ts(ii) = 0.
            else
              canstor(j) = canmxl
            endif
          end do
          if (canstor(j) > canstori) then
            do ii = 1, time%step
              xx = 0.
              xx = wst(iwst)%weat%ts(ii)
              wst(iwst)%weat%ts(ii) = wst(iwst)%weat%ts(ii) - (canstor(j) - canstori)

              if (wst(iwst)%weat%ts(ii) < 0.) then
                canstori = canstori + xx
                wst(iwst)%weat%ts(ii) = 0.
              else
                canstori = canstor(j)
              endif
            end do
          end if

        else
          canmxl = hru(j)%hyd%canmx * pcom(j)%lai_sum / blai_com(j)
          precip_eff = precip_eff - (canmxl - canstor(j))
          if (precipday < 0.) then
            canstor(j) = canstor(j) + precipday
            precip_eff = 0.
          else
            canstor(j) = canmxl
          endif
       end if

      return
      end subroutine sq_canopyint