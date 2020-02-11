      subroutine surface

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine models surface hydrology at any desired time step

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ovrlnd(:)   |mm H2O        |overland flow onto HRU from upstream
!!                               |routing unit
!!    peakr       |mm/hr         |peak runoff rate
!!    surfq(:)    |mm H2O        |surface runoff generated in HRU during
!!                               |the day
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Max
!!    SWAT: canopyint, snom, crackvol, dailycn, volq, crackflow, surfst_h2o,
!!    SWAT: alph, pkq, tran, eiusle, ysed

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~
      
      use basin_module
      use time_module
      use hydrograph_module
      use climate_module, only:  wst
      use hru_module, only : hru, surfq, ovrlnd_dt, ihru, &
        peakr, precipday, precip_eff, qday
      use soil_module
      use urban_data_module
      
      implicit none

      integer :: j                !none          |HRU number 
      real :: ulu                 !              |
      real :: hruirrday           !              |
      integer :: irmmdt           !              |
      integer :: ii               !none          |counter

      j = ihru
      ulu = hru(j)%luse%urb_lu
      hruirrday = 0.
      irmmdt = 0.

      !!compute canopy interception
      call sq_canopyint

      !! compute snow melt
      call sq_snom

      !! compute crack volume
      if (bsn_cc%crk == 1) call sq_crackvol

      if (time%step > 0) then
        do ii = 1, time%step
          wst(iwst)%weat%ts(ii+1) = wst(iwst)%weat%ts(ii+1) + ovrlnd_dt(j,ii)
        end do
      end if

      !!calculate subdaily curve number value
      call sq_dailycn

      !! compute runoff - surfq in mm H2O
      if (precip_eff > 0.1) then
         call sq_volq 

        !! adjust runoff for loss into crack volume
         if (surfq(j) > 0. .and. bsn_cc%crk == 1) call sq_crackflow
      end if

      !! add irrigation runoff and surface runon runoff
      surfq(j) = surfq(j) + irrig(j)%runoff
      irrig(j)%runoff = 0.

      !! calculate amount of surface runoff reaching main channel during day
      !! (qday) and store the remainder
      call sq_surfst
      qday =  surfq(j)

      if (qday > 0.0001) then
        !! compute peak rate - peakr in m3/s  
        call ero_pkq 
      end if  

      if (qday > 0.0001 .and. peakr > 0.) then
        call ero_eiusle

	!! calculate sediment erosion by rainfall and overland flow
		call ero_ovrsed
      end if

      call ero_cfactor
      if (surfq(j) > 1.e-6 .and. peakr > 1.e-6) call ero_ysed

      if (qday < 0.) qday = 0.

1010  format (2(i4,1x),a5,a4,1x,10f8.3)
      return
      end subroutine surface