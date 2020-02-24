      subroutine pl_biomass_gro
      
      use plant_data_module
      use basin_module
      use hru_module, only : hru, uapd, uno3d, lai_yrmx, par, bioday, ep_day, es_day,              &
         ihru, ipl, pet_day, rto_no3, rto_solp, sum_no3, sum_solp, uapd_tot, uno3d_tot, vpd
      use plant_module
      use carbon_module
      use organic_mineral_mass_module
      use climate_module
      use hydrograph_module
      
      implicit none 
      
      integer :: j              !none               |HRU number
      real :: ruedecl           !none               |decline in radiation use efficiency for the
                                !                   |plant
      real :: beadj             !(kg/ha)/(MJ/m**2)  |radiation-use efficiency for a given CO2
                                !                   |concentration
      real :: rto               !none               |ratio of current years of growth:years to maturity of perennial
      integer :: idp            !                   |
      integer :: iob            !                   |

      j = ihru
      idp = pcom(j)%plcur(ipl)%idplt
      rto = 1.
 
        !! if plant hasn't reached maturity
        if (pcom(j)%plcur(ipl)%phuacc <= 1.) then

          !! calculate optimal biomass
          !! adjust radiation-use efficiency for CO2
          if (hru(j)%parms%co2 > 330.) then
            beadj = 100. * hru(j)%parms%co2 / (hru(j)%parms%co2 +        &
                Exp(plcp(idp)%ruc1 - hru(j)%parms%co2 * plcp(idp)%ruc2))
          else
            beadj = pldb(idp)%bio_e
          end if

          !! adjust radiation-use efficiency for vapor pressure deficit
          !!assumes vapor pressure threshold of 1.0 kPa
          if (vpd > 1.0) then
            ruedecl = vpd - 1.0
            beadj = beadj - pldb(idp)%wavp * ruedecl
            beadj = Max(beadj, 0.27 * pldb(idp)%bio_e)
          end if

          beadj = pldb(idp)%bio_e
          
          !! adjust radiation-use efficiency for day length
          iob = hru(j)%obj_no
          iwst = ob(iob)%wst
          beadj = beadj * wst(iwst)%weat%daylength / 12.
          
          bioday = beadj * par(ipl)
          if (bioday < 0.) bioday = 0.
                    
          !! compute temperature stress    
          call pl_tstr

          !! compute n and p uptake and stresses
          if (uno3d_tot > sum_no3) then
            rto_no3 = uno3d(ipl) / uno3d_tot
          else 
            rto_no3 = 1.
          end if
          if (uapd_tot > sum_solp) then
            rto_solp = uapd(ipl) / uapd_tot
          else
            rto_solp = 1.
          end if
       
          uno3d(ipl) = Min(4. * pldb(idp)%pltnfr3 * bioday, uno3d(ipl))
          if (uapd(ipl) > 10.) then
            uapd(ipl) = Min(4. * pldb(idp)%pltpfr3 * bioday, uapd(ipl))
          end if
          ! uno3d(ipl) = uno3d(ipl) * rto_no3
          ! uapd(ipl) = uapd(ipl) * rto_solp
          call pl_nup
          call pl_pup

          !! code to turn off all plant stress
          if (bsn_cc%nostress == 1) then
            pcom(j)%plstr(ipl)%strsw = 1.
            pcom(j)%plstr(ipl)%strst = 1.
            pcom(j)%plstr(ipl)%strsn = 1.
            pcom(j)%plstr(ipl)%strsp = 1.
            pcom(j)%plstr(ipl)%strsa = 1.
          end if
          !! code to turn off nutrient plant stress only
          if (bsn_cc%nostress == 2) then
            pcom(j)%plstr(ipl)%strsn = 1.
            pcom(j)%plstr(ipl)%strsp = 1.
          end if
          
          !! reduce predicted biomass due to stress on plant
          pcom(j)%plstr(ipl)%reg = Min(pcom(j)%plstr(ipl)%strsw, pcom(j)%plstr(ipl)%strst,      &
            pcom(j)%plstr(ipl)%strsn, pcom(j)%plstr(ipl)%strsp, pcom(j)%plstr(ipl)%strsa)
          if (pcom(j)%plstr(ipl)%reg < 0.) pcom(j)%plstr(ipl)%reg = 0.
          if (pcom(j)%plstr(ipl)%reg > 1.) pcom(j)%plstr(ipl)%reg = 1.

          pl_mass_up%m = bioday * pcom(j)%plstr(ipl)%reg
          pl_mass_up%c = 0.42 * bioday * pcom(j)%plstr(ipl)%reg
                                      
          !! increase in plant c
          if (bsn_cc%cswat == 2) then
            cbn_loss(j)%nppc_d = cbn_loss(j)%nppc_d + bioday * pcom(j)%plstr(ipl)%reg * 0.42
          end if

          !! sum plant stresses
          pcom(j)%plstr(ipl)%sum_w = pcom(j)%plstr(ipl)%sum_w + (1. - pcom(j)%plstr(ipl)%strsw)
          pcom(j)%plstr(ipl)%sum_tmp = pcom(j)%plstr(ipl)%sum_tmp + (1.-pcom(j)%plstr(ipl)%strst)
          pcom(j)%plstr(ipl)%sum_n = pcom(j)%plstr(ipl)%sum_n + (1. - pcom(j)%plstr(ipl)%strsn)
          pcom(j)%plstr(ipl)%sum_p = pcom(j)%plstr(ipl)%sum_p + (1. - pcom(j)%plstr(ipl)%strsp) 
          pcom(j)%plstr(ipl)%sum_a = pcom(j)%plstr(ipl)%sum_a + (1. - pcom(j)%plstr(ipl)%strsa)

        end if
        
      return
      end subroutine pl_biomass_gro