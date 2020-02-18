      subroutine sq_snom
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine predicts daily snom melt when the average air
!!    temperature exceeds 0 degrees Celcius

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ihru         |none          |HRU number
!!    snocov1      |none          |1st shape parameter for snow cover equation
!!                                |This parameter is determined by solving the
!!                                |equation for 50% snow cover
!!    snocov2      |none          |2nd shape parameter for snow cover equation
!!                                |This parameter is determined by solving the
!!                                |equation for 95% snow cover
!!    snotmp(:)    |deg C         |temperature of snow pack in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    wst(:)%weat%ts(:)  |mm H2O        |precipitation for the time step during day
!!    snofall      |mm H2O        |amount of precipitation falling as freezing 
!!                                |rain/snow on day in HRU
!!    snomlt       |mm H2O        |amount of water in snow melt for the day in 
!!                                |HRU
!!    snotmp(:)    |deg C         |temperature of snow pack in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Real, Sin, Exp

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use time_module
      use hydrograph_module
      use hru_module, only : hru, snotmp, tmpav, tmx, ihru, precip_eff, snocov1, snocov2,  &
         snofall, snomlt 
      use hydrology_data_module
      use climate_module, only: wst
      
      implicit none

      integer :: ib      !none          |counter
      integer :: j       !none          |HRU number
      real :: sum        !mm H2O        |snow water content in HRU on current day 
      real :: smp        !mm H2O        |precipitation on current day for HRU
      real :: smfac      !              |
      real :: smleb      !mm H2O        |amount of snow melt in elevation band on 
                         !              |current day
      real :: xx         !none          |ratio of amount of current day"s snow water
                         !              |content to the minimum amount needed to
                         !              |cover ground completely 
      real :: snocov     !none          |fraction of HRU area covered with snow
      integer :: isno    !none          |counter
      integer :: ii      !none          |counter

      j = ihru
      sum = 0.
      smp = 0.
      isno = hru(j)%dbs%snow

        !! estimate snow pack temperature
        snotmp(j)=snotmp(j) * (1. - snodb(isno)%timp) + tmpav(j) * snodb(isno)%timp

        if (tmpav(j) <= snodb(isno)%falltmp) then
          !! calculate snow fall
          hru(j)%sno_mm = hru(j)%sno_mm + precip_eff
          snofall = precip_eff
          precip_eff = 0.
          if (time%step > 0) wst(iwst)%weat%ts = 0.
        endif
 
        if (tmx(j) > snodb(isno)%melttmp .and. hru(j)%sno_mm > 0.) then
          !! adjust melt factor for time of year
          smfac = (snodb(isno)%meltmx + snodb(isno)%meltmn) / 2. +      &     
             Sin((time%day - 81) / 58.09) *                                 &
             (snodb(isno)%meltmx - snodb(isno)%meltmn) / 2.             !! 365/2pi = 58.09
          snomlt = smfac * (((snotmp(j)+tmx(j))/2.)-snodb(isno)%melttmp)

          !! adjust for areal extent of snow cover
          if (hru(j)%sno_mm < snodb(isno)%covmx) then
            xx = hru(j)%sno_mm / snodb(isno)%covmx
            snocov = xx / (xx + Exp(snocov1 - snocov2 * xx))
          else
            snocov = 1.
          endif
          snomlt = snomlt * snocov
          if (snomlt < 0.) snomlt = 0.
          if (snomlt > hru(j)%sno_mm) snomlt = hru(j)%sno_mm
          hru(j)%sno_mm = hru(j)%sno_mm - snomlt
          precip_eff = precip_eff + snomlt
          if (time%step > 0) then
            do ii = 1, time%step
             wst(iwst)%weat%ts(ii+1) = wst(iwst)%weat%ts(ii+1) + snomlt / time%step
            end do
          end if
          if (precip_eff < 0.) precip_eff = 0.
        else
          snomlt = 0.
        end if
 
      return
      end subroutine sq_snom