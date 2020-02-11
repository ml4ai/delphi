      subroutine swr_latsed

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine calculates the sediment load contributed in lateral flow

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    hru_km(:)   |km^2          |area of HRU in square kilometers
!!    lat_sed(:)  |g/L           |sediment concentration in lateral flow
!!    latq(:)     |mm H2O        |total lateral flow in soil profile for the
!!                               |day in HRU
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    j           |none          |HRU number
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only : hru, ihru, sedyld, sanyld, silyld, clayld, sagyld, lagyld, sedorgn,  &
         sedorgp, latq 
      use soil_module
      
      implicit none

      integer :: j              !none          |HRU number

      j = 0
      j = ihru

      !! update sediment yield for sediment in lateral flow
      sedyld(j)=sedyld(j) + latq(j) * hru(j)%km * hru(j)%hyd%lat_sed
      sanyld(j)=sanyld(j)+latq(j) * hru(j)%km * hru(j)%hyd%lat_sed * soil(j)%det_san
      silyld(j)=silyld(j)+latq(j) * hru(j)%km * hru(j)%hyd%lat_sed * soil(j)%det_sil
      clayld(j)=clayld(j)+latq(j) * hru(j)%km * hru(j)%hyd%lat_sed * soil(j)%det_cla
      sagyld(j)=sagyld(j)+latq(j) * hru(j)%km * hru(j)%hyd%lat_sed * soil(j)%det_sag
      lagyld(j)=lagyld(j)+latq(j) * hru(j)%km * hru(j)%hyd%lat_sed * soil(j)%det_lag

      !! organic n and p in the lateral flow     - by J.Jeong BREC 2011  
      !mm*mg/L*1000L/m3*kg/1000000mg*m3/10ha-mm=kg/ha
      sedorgn(j) = sedorgn(j) + latq(j) * hru(j)%hyd%lat_orgn /10000.
      sedorgp(j) = sedorgp(j) + latq(j) * hru(j)%hyd%lat_orgp /10000.

      if (sedyld(j) < 0.) sedyld(j) = 0.
      if (sanyld(j) < 0.) sanyld(j) = 0.0
      if (silyld(j) < 0.) silyld(j) = 0.0
      if (clayld(j) < 0.) clayld(j) = 0.0
      if (sagyld(j) < 0.) sagyld(j) = 0.0
      if (lagyld(j) < 0.) lagyld(j) = 0.0
      if (sedorgn(j) < 0.) sedorgn(j) = 0.0
      if (sedorgp(j) < 0.) sedorgp(j) = 0.0

      return
      end subroutine swr_latsed