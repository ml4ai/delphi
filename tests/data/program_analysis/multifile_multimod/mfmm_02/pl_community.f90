      subroutine pl_community

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine predicts daily potential growth of total plant
!!    biomass and roots and calculates leaf area index. Incorporates
!!    residue for tillage functions and decays residue on ground
!!    surface. Adjusts daily dry matter based on water stress.

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    phubase(:)  |heat units    |base zero total heat units (used when no
!!                               |land cover is growing)
!!    phutot(:)   |heat units    |total potential heat units for year (used
!!                               |when no crop is growing)
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    phubase(:)  |heat units    |base zero total heat units (used when no
!!                               |land cover is growing)
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Max
!!    SWAT: operatn, swu, grow

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use hru_module, only : ep_max, epmax, hru_ra, htfac, ihru, ipl, par, tmpav, translt
      use soil_module
      use plant_module
      use plant_data_module
      use organic_mineral_mass_module
      use time_module
      
      implicit none      

      integer :: j              !none          |HRU number
      integer :: idp            !              | 
      integer :: npl_gro        !              | 
      integer :: ip 
      integer :: jpl            !none          |counter
      real :: x1                !              |
      real :: sum               !              |
      real :: sumf              !              |
      real :: sumle             !              |
      real :: fi                !              |

      j = ihru  
      par = 0.

      !! calc total lai for plant community
      pcom(j)%lai_sum = 0.
      do ipl = 1, pcom(j)%npl
        pcom(j)%lai_sum = pcom(j)%lai_sum + pcom(j)%plg(ipl)%lai
      end do
      !! calc max water uptake for each plant
      do ipl = 1, pcom(j)%npl
        if (pcom(j)%lai_sum > 1.e-6) then
          epmax(ipl) = ep_max * pcom(j)%plg(ipl)%lai / pcom(j)%lai_sum
        else
          epmax(ipl) = 0.
        end if
      end do

      !! sum total masses for plant community
      pl_mass(j)%tot_com%m = 0.
      pl_mass(j)%ab_gr_com%m = 0.
      pl_mass(j)%leaf_com%m = 0.
      pl_mass(j)%stem_com%m = 0.
      pl_mass(j)%seed_com%m = 0.
      pl_mass(j)%root_com%m = 0.
      do ipl = 1, pcom(j)%npl
        pl_mass(j)%tot_com%m = pl_mass(j)%tot_com%m + pl_mass(j)%tot(ipl)%m
        pl_mass(j)%ab_gr_com%m = pl_mass(j)%ab_gr_com%m + pl_mass(j)%ab_gr(ipl)%m
        pl_mass(j)%leaf_com%m = pl_mass(j)%leaf_com%m + pl_mass(j)%leaf(ipl)%m
        pl_mass(j)%stem_com%m = pl_mass(j)%stem_com%m + pl_mass(j)%stem(ipl)%m
        pl_mass(j)%seed_com%m = pl_mass(j)%seed_com%m + pl_mass(j)%seed(ipl)%m
        pl_mass(j)%root_com%m = pl_mass(j)%root_com%m + pl_mass(j)%root(ipl)%m
      end do
      
      !! calc max height for penman pet equation
      pcom(j)%cht_mx = 0.
      do ipl = 1, pcom(j)%npl
        pcom(j)%cht_mx = amax1 (pcom(j)%cht_mx, pcom(j)%plg(ipl)%cht)
      end do
      
      !! calc total biomass for plant community
      pl_mass(j)%tot_com%m = 0.
      do ipl = 1, pcom(j)%npl
        pl_mass(j)%tot_com%m = amax1 (pl_mass(j)%tot_com%m, pl_mass(j)%tot(ipl)%m)
      end do
      
      npl_gro = 0
      do ipl = 1, pcom(j)%npl
        if (pcom(j)%plcur(ipl)%gro == "y") then
          call pl_waterup
          npl_gro = npl_gro + 1
          ip = ipl  !used for only one plant growing
        end if
      end do
 
      !! calculate photosynthetically active radiation during growth period
      if (npl_gro == 1) then
        !! calculate photosynthetically active radiation for one plant
        if (pcom(j)%plcur(ip)%idorm == "n" .and. pcom(j)%plcur(ip)%gro        & 
                                                              == "y")then
          idp = pcom(j)%plcur(ip)%idplt
          pl_db => pldb(idp)
          par(ip) = .5 * hru_ra(j) * (1. - Exp(-pldb(idp)%ext_coef *        &    
                (pcom(j)%plg(ip)%lai + .05)))
        end if
      else if (npl_gro > 1) then
        !! calculate photosynthetically active radiation for multiple plants
        if (pcom(j)%lai_sum > 1.e-6) then
          translt = 0.
          do ipl = 1, pcom(j)%npl
            do jpl = 1, pcom(j)%npl
              x1 = pcom(j)%plg(jpl)%cht - .5 * pcom(j)%plg(ipl)%cht
              if (x1 > 0.) then
                idp = pcom(j)%plcur(ipl)%idplt
                pl_db => pldb(idp)
                translt(ipl) = translt(ipl) + x1 / (pcom(j)%plg(ipl)%cht + 1.e-6) *   & 
                         pcom(j)%plg(ipl)%lai * (-pldb(idp)%ext_coef)
              end if
            end do
          end do
          sum = 0.
          do ipl = 1,pcom(j)%npl
            translt(ipl) = exp(translt(ipl))
            sum = sum + translt(ipl)
          end do
          sumf = 0.
          sumle = 0.
          do ipl = 1, pcom(j)%npl
            idp = pcom(j)%plcur(ipl)%idplt
            translt(ipl) = translt(ipl) / sum
            x1 = pcom(j)%plg(ipl)%lai * pldb(idp)%ext_coef
            sumle = sumle + x1
            sumf = sumf + (1. - exp(-x1)) * translt(ipl)
          end do
          fi = 1. - exp(-sumle)
          do ipl = 1, pcom(j)%npl
            idp = pcom(j)%plcur(ipl)%idplt
            if (sumf > 0.) then
              htfac(ipl) = (1. - exp(-pldb(idp)%ext_coef *                  &             
                          pcom(j)%plg(ipl)%lai)) * translt(ipl) / sumf
            else
              htfac(ipl) = 1.
            end if
            htfac(ipl) = fi * htfac(ipl)
            htfac(ipl) = 1.
            par(ipl) = .5 * htfac(ipl) * hru_ra(j) * (1. -                  &              
              Exp(-pldb(idp)%ext_coef * (pcom(j)%plg(ipl)%lai + .05)))
          end do  
        end if
      end if

      return
      
      end subroutine pl_community