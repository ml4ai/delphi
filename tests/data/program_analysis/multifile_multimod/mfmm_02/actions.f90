      subroutine actions (ob_cur, ob_num, idtbl)
      use conditional_module
      use climate_module
      use time_module
      use hru_module, only : fertno3, fertnh3, fertorgn, fertorgp, fertsolp,   &
        ihru, ipl, isol, ndeat, phubase, sol_sumno3, sol_sumsolp, hru, yield 
      use soil_module
      use plant_module
      use plant_data_module
      use mgt_operations_module  
      use tillage_data_module
      use reservoir_module
      use sd_channel_module
      use hru_lte_module
      use basin_module
      use organic_mineral_mass_module
      use hydrograph_module
      use tiles_data_module
      use output_landscape_module
      use conditional_module
      use constituent_mass_module
      use calibration_data_module

      implicit none

      integer, intent (in)  :: ob_cur      !none     |sequential number of individual objects
      integer, intent (in)  :: ob_num      !none     |sequential number for all objects
      integer, intent (in)  :: idtbl       !none     |
      integer :: icom                      !none     |
      integer :: iac                       !none     |counter
      integer :: ial                       !none     |counter
      integer :: jj                        !none     |counter
      integer :: i                         !none     |counter
      integer :: iburn                     !none     |burn type from fire data base
      integer :: idtill                    !none     |tillage type
      integer :: ifertop                   !         |surface application fraction from chem app data base
      integer :: ifrt                      !         |fertilizer type from fert data base
      integer :: ipestop                   !         |surface application fraction from chem app data base
      integer :: ipst                      !         |pesticide type from pest data base
      integer :: iharvop                   !         |harvest operation type
      integer :: iihru                     !         |
      integer :: ilu                       !         |landuse type 
      integer :: j                         !none     |counter
      integer :: iob
      integer :: idp                       !         |
      integer :: istr                      !         |
      integer :: iob_out
      integer :: inhyd                     !         |
      integer :: ihyd_in                   !         |
      integer :: icon                      !         |
      integer :: iplt_bsn
      integer :: irrop                     !         |
      integer :: igr
      integer :: ireg                      !         |
      integer :: ilum
      real :: hiad1                        !         |
      real :: irrig_m3                     !         |
      real :: amt_mm                       !         |
      real :: biomass                      !         |
      real :: frt_kg
      real :: wur                          !         |
      real :: frac                         !         |
      real :: rto                          !         |
      real :: rto1                         !         |
      real :: pest_kg                      !kg/ha    |amount of pesticide applied 
      character(len=1) :: action           !         |

      do iac = 1, d_tbl%acts
        action = "n"
        do ial = 1, d_tbl%alts
          if (d_tbl%act_hit(ial) == "y" .and. d_tbl%act_outcomes(iac,ial) == "y") then
            action = "y"
            exit
          end if
        end do
      
        if (action == "y") then
          select case (d_tbl%act(iac)%typ)
          
          !irrigation demand - hru action
          case ("irr_demand")
            ipl = 1
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur

            irrop = d_tbl%act_typ(iac)
            irrig(j)%demand = irrop_db(irrop)%amt_mm * hru(j)%area_ha / 1000.       ! ha-m = mm * ha / 1000.
            
            !! if unlimited source, set irrigation applied directly to hru
            if (d_tbl%act(iac)%file_pointer == "unlim") then
              irrig(j)%applied = irrop_db(irrop)%amt_mm * irrop_db(irrop)%eff * (1. - irrop_db(irrop)%surq)
              irrig(j)%runoff = irrop_db(irrop)%amt_mm * irrop_db(irrop)%eff * irrop_db(irrop)%surq
              !set organics and constituents from irr.ops ! irrig(j)%water =  cs_irr(j) = 
              if (pco%mgtout == "y") then
                write (2612, *) j, time%yrc, time%mo, time%day, "        ", "IRRIGATE", phubase(j),  &
                    pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m, rsd1(j)%tot(ipl)%m, &
                    sol_sumno3(j), sol_sumsolp(j), irrig(j)%applied
              end if
            else
              !! set demand for irrigation from channel, reservoir or aquifer
              if (pco%mgtout == "y") then
                write (2612, *) j, time%yrc, time%mo, time%day, "        ", "IRRIG_DMD", phubase(j), &
                    pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m, rsd1(j)%tot(ipl)%m, &
                    sol_sumno3(j), sol_sumsolp(j), irrig(j)%demand
              end if
            end if

          !irrigate - hru action
          case ("irrigate")
            ipl = 1
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur

            irrop = mgt%op4                        !irrigation amount (mm) from irr.ops data base
            irrig(j)%applied = irrop_db(irrop)%amt_mm * irrop_db(irrop)%eff * (1. - irrop_db(irrop)%surq)
            irrig(j)%runoff = irrop_db(irrop)%amt_mm * irrop_db(irrop)%surq

            irrig(j)%demand = mgt%op3 * hru(j)%area_ha / 1000.        ! ha-m = mm * ha / 1000.

            !select object type
            iob = d_tbl%act(iac)%const
            select case (d_tbl%act(iac)%option)
            case ("aqu")
              irrig_m3 = amin1 (ch_stor(j)%flo, irrig(j)%demand)
              rto = irrig_m3 / ch_stor(j)%flo                               ! ratio of water removed from aquifer volume
              rto1 = (1. - rto)
              irrig(j)%applied = irrig(j)%applied * hru(j)%area_ha / 10.    ! convert to mm for hru application
              irrig(j)%water = rto * ch_stor(j)                             ! organics in irrigation water
              aqu(j) = rto1 * ch_stor(j)                                    ! remainder stays in aquifer
              cs_irr(j) = rto * cs_aqu(j)                                   ! constituents in irrigation water
              cs_aqu(j) = rto1 * cs_aqu(j)                                  ! remainder stays in aquifer
              
            case ("cha")
              irrig_m3 = amin1 (ch_stor(j)%flo, irrig(j)%demand)
              rto = irrig_m3 / ch_stor(j)%flo                               ! ratio of water removed from channel volume
              rto1 = (1. - rto)
              irrig(j)%applied = irrig(j)%applied * hru(j)%area_ha / 10.    ! convert to mm for hru application
              irrig(j)%water = rto * ch_stor(j)                             ! organics in irrigation water
              !ch_stor(j) = rto1 * ch_water(j)                              ! remainder stays in channel
              cs_irr(j) = rto * ch_water(j)                                 ! constituents in irrigation water
              ch_water(j) = rto1 * ch_water(j)                              ! remainder stays in channel
              
            case ("res")
              irrig_m3 = amin1 (res(j)%flo, irrig(j)%demand)
              rto = irrig_m3 / res(j)%flo                                   ! ratio of water removed from res volume
              rto1 = (1. - rto)
              irrig(j)%applied = irrig(j)%applied * hru(j)%area_ha / 10.    ! convert to mm for hru application
              irrig(j)%water = rto * res(j)                                 ! organics in irrigation water
              res(j) = rto1 * res(j)                                        ! remainder stays in reservoir
              cs_irr(j) = rto * res_water(j)                                ! constituents in irrigation water
              res_water(j) = rto1 * res_water(j)                            ! remainder stays in reservoir
              
            end select
                  
            if (pco%mgtout == "y") then
              write (2612, *) j, time%yrc, time%mo, time%day, "        ", "IRRIGATE", phubase(j),  &
                  pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m, rsd1(j)%tot(ipl)%m, &
                  sol_sumno3(j), sol_sumsolp(j), irrig(j)%demand
            end if

          !fertilize
          case ("fertilize")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              ipl = 1
              ifrt = d_tbl%act_typ(iac)               !fertilizer type from fert data base
              frt_kg = d_tbl%act(iac)%const           !amount applied in kg/ha
              ifertop = d_tbl%act_app(iac)            !surface application fraction from chem app data base
              call pl_fert (j, ifrt, frt_kg, ifertop)

              if (pco%mgtout == "y") then
                !write (2612, *) j, time%yrc, time%mo, time%day, chemapp_db(mgt%op4)%name, "    FERT", &
                write (2612, *) j, time%yrc, time%mo, time%day, chemapp_db(j)%name, "    FERT",       &
                  phubase(j),pcom(j)%plcur(ipl)%phuacc, soil(j)%sw, pl_mass(j)%tot(ipl)%m,            &
                  rsd1(j)%tot(ipl)%m, sol_sumno3(j), sol_sumsolp(j), frt_kg, fertno3, fertnh3,        &
                  fertorgn, fertsolp, fertorgp
              endif
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if

          !tillage
          case ("till")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              idtill = d_tbl%act_app(iac)
              ipl = 1
              call mgt_newtillmix(j, 0., idtill)
            
              if (pco%mgtout == "y") then
                write (2612, *) j, time%yrc, time%mo, time%day, tilldb(idtill)%tillnm, "TILLAGE",    &
                    phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw, pl_mass(j)%tot(ipl)%m,        &
                    rsd1(j)%tot(ipl)%m, sol_sumno3(j), sol_sumsolp(j), tilldb(idtill)%effmix
              end if
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if

          !plant
          case ("plant")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            icom = pcom(j)%pcomdb
            pcom(j)%days_plant = 1       !reset days since last planting
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              do ipl = 1, pcom(j)%npl
                idp = pcomdb(icom)%pl(ipl)%db_num
                if (d_tbl%act(iac)%option == pcomdb(icom)%pl(ipl)%cpnm) then
                  pcom(j)%plcur(ipl)%gro = "y"
                  pcom(j)%plcur(ipl)%idorm = "n"
                if (pco%mgtout == "y") then
                  write (2612, *) j, time%yrc, time%mo, time%day, pldb(idp)%plantnm, pcomdb(icom)%name,  &
                      phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(ihru)%sw,                              &
                      pl_mass(j)%tot(ipl)%m, rsd1(j)%tot_com%m, sol_sumno3(j),                           &
                      sol_sumsolp(j), pcom(j)%plg(ipl)%lai, pcom(j)%plcur(ipl)%laimx_pop
                  end if
                end if
              end do
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
            
          !harvest only
          case ("harvest")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              iharvop = d_tbl%act_typ(iac)
              icom = pcom(j)%pcomdb
              pcom(j)%days_harv = 1       !reset days since last harvest
            
              do ipl = 1, pcom(j)%npl
                biomass = pl_mass(j)%tot(ipl)%m
                if (d_tbl%act(iac)%option == pcomdb(icom)%pl(ipl)%cpnm .or. d_tbl%act(iac)%option == "all") then
                          
                  !harvest specific type
                  select case (harvop_db(iharvop)%typ)
                  case ("biomass")    
                    call mgt_harvbiomass (j, ipl, iharvop)
                  case ("grain")
                    call mgt_harvgrain (j, ipl, iharvop)
                  case ("residue")
                  case ("tree")
                  case ("tuber")
                    call mgt_harvtuber (j, ipl, iharvop)
                  end select

                  !! sum yield and num. of harvest to calc ave yields
                  pl_mass(j)%yield_tot(ipl) = pl_mass(j)%yield_tot(ipl) + pl_yield
                  pcom(j)%plcur(ipl)%harv_num = pcom(j)%plcur(ipl)%harv_num + 1
                            
                  !! sum basin crop yields and area harvested
                  iplt_bsn = pcom(j)%plcur(ipl)%bsn_num
                  bsn_crop_yld(iplt_bsn)%area_ha = bsn_crop_yld(iplt_bsn)%area_ha + hru(j)%area_ha
                  bsn_crop_yld(iplt_bsn)%yield = bsn_crop_yld(iplt_bsn)%yield + yield * hru(j)%area_ha / 1000.
                  !! sum regional crop yields for soft calibration
                  ireg = hru(j)%crop_reg
                  do ilum = 1, plcal(ireg)%lum_num
                    if (plcal(ireg)%lum(ilum)%meas%name == mgt%op_char) then
                      plcal(ireg)%lum(ilum)%ha = plcal(ireg)%lum(ilum)%ha + hru(j)%area_ha
                      plcal(ireg)%lum(ilum)%sim%yield = plcal(ireg)%lum(ilum)%sim%yield + pl_yield%m * hru(j)%area_ha / 1000.
                    end if
                  end do
            
                  idp = pcom(j)%plcur(ipl)%idplt
                  if (pco%mgtout == "y") then
                    write (2612, *) j, time%yrc, time%mo, time%day,  pldb(idp)%plantnm, "HARVEST",      &
                        phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw, biomass, rsd1(j)%tot(ipl)%m, &
                        sol_sumno3(j), sol_sumsolp(j), pl_yield%m, pcom(j)%plstr(ipl)%sum_n, &
                        pcom(j)%plstr(ipl)%sum_p, pcom(j)%plstr(ipl)%sum_tmp, pcom(j)%plstr(ipl)%sum_w, &
                        pcom(j)%plstr(ipl)%sum_a
                  end if 
                end if
                pcom(j)%plcur(ipl)%phuacc = 0.
              end do
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if

          !kill plant
          case ("kill")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              icom = pcom(j)%pcomdb
              do ipl = 1, pcom(j)%npl
                biomass = pl_mass(j)%tot(ipl)%m
                if (d_tbl%act(iac)%option == pcomdb(icom)%pl(ipl)%cpnm .or. d_tbl%act(iac)%option == "all") then

                  call mgt_killop (j, ipl)

                  idp = pcom(j)%plcur(ipl)%idplt
                  if (pco%mgtout == "y") then
                    write (2612, *) j, time%yrc, time%mo, time%day,  pldb(idp)%plantnm, "HARV/KILL",     &
                        phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw, biomass, rsd1(j)%tot(ipl)%m,  &
                        sol_sumno3(j), sol_sumsolp(j), yield, pcom(j)%plstr(ipl)%sum_n,                  &
                        pcom(j)%plstr(ipl)%sum_p, pcom(j)%plstr(ipl)%sum_tmp, pcom(j)%plstr(ipl)%sum_w,  &
                        pcom(j)%plstr(ipl)%sum_a
                  end if 
                end if
                pcom(j)%plcur(ipl)%phuacc = 0.
                phubase(j) = 0.
              end do
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
  
          !harvest and kill
          case ("harvest_kill")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              iharvop = d_tbl%act_typ(iac)
              icom = pcom(j)%pcomdb
              pcom(j)%days_harv = 1       !reset days since last harvest
            
              do ipl = 1, pcom(j)%npl
                biomass = pl_mass(j)%tot(ipl)%m
                if (d_tbl%act(iac)%option == pcomdb(icom)%pl(ipl)%cpnm .or. d_tbl%act(iac)%option == "all") then
                          
                  !harvest specific type
                  select case (harvop_db(iharvop)%typ)
                  case ("biomass")    
                    call mgt_harvbiomass (j, ipl, iharvop)
                  case ("grain")
                    call mgt_harvgrain (j, ipl, iharvop)
                  case ("residue")
                  case ("tree")
                  case ("tuber")
                    call mgt_harvtuber (j, ipl, iharvop)
                  end select
            
                  call mgt_killop (j, ipl)

                  !! sum yield and num. of harvest to calc ave yields
                  pl_mass(j)%yield_tot(ipl) = pl_mass(j)%yield_tot(ipl) + pl_yield
                  pcom(j)%plcur(ipl)%harv_num = pcom(j)%plcur(ipl)%harv_num + 1
                            
                  !! sum basin crop yields and area harvested
                  iplt_bsn = pcom(j)%plcur(ipl)%bsn_num
                  bsn_crop_yld(iplt_bsn)%area_ha = bsn_crop_yld(iplt_bsn)%area_ha + hru(j)%area_ha
                  bsn_crop_yld(iplt_bsn)%yield = bsn_crop_yld(iplt_bsn)%yield + yield * hru(j)%area_ha / 1000.
                  !! sum regional crop yields for soft calibration
                  ireg = hru(j)%crop_reg
                  do ilum = 1, plcal(ireg)%lum_num
                    if (plcal(ireg)%lum(ilum)%meas%name == d_tbl%act(iac)%option) then
                      plcal(ireg)%lum(ilum)%ha = plcal(ireg)%lum(ilum)%ha + hru(j)%area_ha
                      plcal(ireg)%lum(ilum)%sim%yield = plcal(ireg)%lum(ilum)%sim%yield + pl_yield%m * hru(j)%area_ha / 1000.
                    end if
                  end do
            
                  idp = pcom(j)%plcur(ipl)%idplt
                  if (pco%mgtout == "y") then
                    write (2612, *) j, time%yrc, time%mo, time%day,  pldb(idp)%plantnm, "HARV/KILL",        &
                        phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw, biomass, rsd1(j)%tot(ipl)%m,     &
                        sol_sumno3(j), sol_sumsolp(j), pl_yield%m, pcom(j)%plstr(ipl)%sum_n,   &
                        pcom(j)%plstr(ipl)%sum_p, pcom(j)%plstr(ipl)%sum_tmp, pcom(j)%plstr(ipl)%sum_w,     &
                        pcom(j)%plstr(ipl)%sum_a
                  end if 
                end if
                pcom(j)%plcur(ipl)%phuacc = 0.
                phubase(j) = 0.
              end do
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
  
          !reset rotation year
          case ("rot_reset")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            if (d_tbl%act(iac)%const < 1) d_tbl%act(iac)%const = 1
            pcom(j)%rot_yr = d_tbl%act(iac)%const
              
          !apply pesticide
          case ("pest_apply")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              ipl = 1
              ipst = d_tbl%act_typ(iac)                                     !pesticide type from fert data base
              ipestop = d_tbl%act_app(iac)                                  !surface application fraction from chem app data base
              pest_kg = d_tbl%act(iac)%const * chemapp_db(ipestop)%app_eff  !amount applied in kg/ha
              call pest_apply (j, ipst, pest_kg, ipestop)

              if (pco%mgtout == "y") then
                write (2612, *) j, time%yrc, time%mo, time%day_mo, d_tbl%act(iac)%option, "    PEST ",        &
                 phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m,           &
                 rsd1(j)%tot(ipl)%m, sol_sumno3(j), sol_sumsolp(j), pest_kg
              endif
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
          
          case ("graze")    !! grazing operation
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur

            igr = d_tbl%act_typ(iac)
            graze = grazeop_db(igr)
            call pl_graze
            
              !if (pco%mgtout == "y") then
              !  write (2612, *) j, time%yrc, time%mo, time%day, "         ", "    GRAZE",         &
              !    phubase(j), pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m,        &
              !    rsd1(j)%tot(ipl)%m, sol_sumno3(j), sol_sumsolp(j), grazeop_db(igr)%eat, grazeop_db(igr)%manure
              !end if
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1

          !initiate growing season for hru_lte
          case ("grow_init")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            hlt(j)%gro = "y"
            hlt(j)%g = 0.
            hlt(j)%alai = 0.
            hlt(j)%dm = 0.
            hlt(j)%hufh = 0.

          !end growing season for hru_lte
          case ("grow_end")
            !calculate yield - print lai, biomass and yield
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
              idp = hlt(j)%iplant
              if (hlt(j)%pet < 10.) then
                wur = 100.
              else
                wur = 100. * hlt(j)%aet / hlt(j)%pet
              endif
              hiad1 = (pldb(idp)%hvsti - pldb(idp)%wsyf) *                            &   
                        (wur / (wur + Exp(6.13 - .0883 * wur))) + pldb(idp)%wsyf
              hiad1 = amin1 (hiad1, pldb(idp)%hvsti)
              yield = 0.8 * hlt(j)%dm * hiad1  ! * hlt(isd)%stress
              hlt(j)%yield = yield / 1000.
              hlt(j)%npp = hlt(j)%dm / 1000.
              hlt(j)%lai_mx = hlt(j)%alai
              !compute annual net primary productivity (npp) for perennial non-harvested?
              !use output.mgt print code
              !write() isd, time%day, time%yrc, pldb(iplt)%plantnm, hlt(isd)%alai, hlt(isd)%dm, yield
              hlt(j)%gro = "n"
              hlt(j)%g = 0.
              hlt(j)%alai = 0.
              hlt(j)%dm = 0.     !adjust for non-harvested perennials?
              hlt(j)%hufh = 0.
              hlt(j)%aet = 0.
              hlt(j)%pet = 0.

          !drainage water management
          case ("drain_control") !! set drain depth for drainage water management
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
             istr = hru(j)%tiledrain
              hru(j)%lumv%sdr_dep = d_tbl%act(iac)%const * sdr(istr)%depth
              !if (hru(j)%lumv%sdr_dep > 0) then
              !  do jj = 1, soil(j)%nly
              !    if (hru(j)%lumv%sdr_dep < soil(j)%phys(jj)%d) hru(j)%lumv%ldrain = jj
              !    if (hru(j)%lumv%sdr_dep < soil(j)%phys(jj)%d) exit
              !  end do
              !else
              !  hru(j)%lumv%ldrain = 0
              !end if
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
                           
          !tile flow control for saturated buffers
          case ("flow_control") !! set flow fractions to buffer tile and direct to channel
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            select case (d_tbl%act(iac)%option)
            case ("min_flo")    
              if (hwb_d(j)%qtile < d_tbl%act(iac)%const) then
                frac = 1.
              else
                frac = d_tbl%act(iac)%const / hwb_d(j)%qtile
              end if
              ! set inflow hydrograph fraction of recieving objects - used for dtbl flow fractions
              ! set first object hyd fractin as defined in decision table
              inhyd = dtbl_flo(idtbl)%act(iac)%ob_num
              ihyd_in = ob(ob_num)%rcvob_inhyd(inhyd)
              iob_out = ob(ob_num)%obj_out(inhyd)
              ob(iob_out)%frac_in(ihyd_in) = frac
              
              ! set second hydrograph fraction
              if (inhyd < ob(ob_num)%src_tot .and. dtbl_flo(idtbl)%act(iac)%typ /= "irrigate_direct") then
                inhyd = inhyd + 1
                ihyd_in = ob(ob_num)%rcvob_inhyd(inhyd)
                iob_out = ob(ob_num)%obj_out(inhyd)
                ob(iob_out)%frac_in(ihyd_in) = 1. - frac
              end if

            case ("linear")

            case ("power")
                
            end select
                                       
          !tile flow control for saturated buffers
          case ("tile_control") !! set flow fractions to buffer tile and direct to channel
            icon = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            select case (d_tbl%act(iac)%option)
            case ("min_flo")    
              if (hwb_d(j)%qtile < d_tbl%act(iac)%const) then
                frac = 1.
              else
                frac = d_tbl%act(iac)%const / hwb_d(j)%qtile
              end if
              ! set inflow hydrograph fraction of recieving objects - used for dtbl flow fractions
              ! set first object hyd fractin as defined in decision table
              inhyd = dtbl_flo(idtbl)%act(iac)%ob_num
              ihyd_in = ob(ob_num)%rcvob_inhyd(inhyd)
              iob_out = ob(ob_num)%obj_out(inhyd)
              ob(iob_out)%frac_in(ihyd_in) = frac
              
              ! set second hydrograph fraction
              if (inhyd < ob(ob_num)%src_tot .and. dtbl_flo(idtbl)%act(iac)%typ /= "irrigate_direct") then
                inhyd = inhyd + 1
                ihyd_in = ob(ob_num)%rcvob_inhyd(inhyd)
                iob_out = ob(ob_num)%obj_out(inhyd)
                ob(iob_out)%frac_in(ihyd_in) = 1. - frac
              end if

            case ("linear")

            case ("power")
                
            end select
            
          !land use change
          case ("lu_change")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            ilu = d_tbl%act_typ(iac)
            hru(j)%dbs%land_use_mgt = ilu
            hru(j)%land_use_mgt_c = d_tbl%act(iac)%file_pointer
            isol = hru(j)%dbs%soil  
            call plant_init (1)
                     
          !channel change
          case ("chan_change")
            ich = ob_cur
            !set new cover and name for calibration
            sd_ch(ich)%cov = d_tbl%act(iac)%const
            sd_ch(ich)%order = d_tbl%act(iac)%file_pointer
        
          ! burning
          case ("burn")
            j = d_tbl%act(iac)%ob_num
            if (j == 0) j = ob_cur
            
            if (pcom(j)%dtbl(idtbl)%num_actions(iac) <= Int(d_tbl%act(iac)%const2)) then
              iburn = d_tbl%act_typ(iac)           !burn type from fire data base
              do ipl = 1, pcom(j)%npl
                call pl_burnop (j, ipl, iburn)
              end do
                        
              if (pco%mgtout == "y") then
                write (2612, *) j, time%yrc, time%mo, time%day, "        ", "    BURN", phubase(j),    &
                    pcom(j)%plcur(ipl)%phuacc, soil(j)%sw,pl_mass(j)%tot(ipl)%m, rsd1(j)%tot(ipl)%m,   &
                    sol_sumno3(j), sol_sumsolp(j)
              end if
              pcom(j)%dtbl(idtbl)%num_actions(iac) = pcom(j)%dtbl(idtbl)%num_actions(iac) + 1
            end if
          
          !herd management - move the herd
          case ("herd")
            
          !water rights decision to move water
          case ("water_rights")
            
          end select
        end if
      end do

      return
      end subroutine actions