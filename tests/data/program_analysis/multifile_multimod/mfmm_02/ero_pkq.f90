      subroutine ero_pkq
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine computes the peak runoff rate for each HRU
!!    and the entire subbasin using a modification of the rational formula

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    hru_km(:)    |km^2          |area of HRU in square kilometers
!!    ihru         |none          |HRU number
!!    tconc(:)     |hr            |time of concentration for HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name         |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    peakr        |m^3/s         |peak runoff rate
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Log, Expo

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only: hru, tconc, ihru, peakr, qday
      use hydrograph_module
      use climate_module
      
      implicit none

      integer :: j      !none          |HRU number
      real :: altc      !              |
      real :: expo      !              | 
      integer :: iob    !              | 
      
      j = ihru
      iob = hru(j)%obj_no
      iwst = ob(iob)%wst
      
      altc = 1. - expo(2. * tconc(j) * Log(1. - wst(iwst)%weat%precip_half_hr))
      peakr = altc * qday / tconc(j)           !! mm/h
      peakr = peakr * hru(j)%km / 3.6          !! m^3/s

      return
      end subroutine ero_pkq