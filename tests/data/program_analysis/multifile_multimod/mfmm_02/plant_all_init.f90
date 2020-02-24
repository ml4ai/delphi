      subroutine plant_all_init
    
      use plant_module
      use plant_data_module
      use hru_module, only : hru, ihru, isol, ilu
      use hydrograph_module, only : sp_ob
      use maximum_data_module
            
      implicit none

      integer :: ipl
      integer :: iplt
      integer :: ipl_bsn
      integer :: num_plts_cur   !none       |temporary counter for number of different plants in basin

      allocate (plts_bsn(db_mx%plantparm))
      
      !!assign land use pointers for the hru
      !!allocate and initialize land use and management
      do ihru = 1, sp_ob%hru
        !!ihru, ilu and isol are in modparm
        ilu = hru(ihru)%dbs%land_use_mgt
        isol = hru(ihru)%dbs%soil
        !send 0 value in when initializing- 1 for updating to deallocate
        call plant_init (0)
      end do

      !! find all different plants in simulation for yield output
      do ihru = 1, sp_ob%hru
        do ipl = 1, pcom(ihru)%npl
          if (basin_plants == 0) then
            plts_bsn(1) = pcom(ihru)%plg(ipl)%cpnm
            basin_plants = 1
          end if
          num_plts_cur = basin_plants
          do iplt = 1, num_plts_cur
            if (pcom(ihru)%plg(ipl)%cpnm == plts_bsn(iplt)) exit
            if (iplt == num_plts_cur) then
              plts_bsn(iplt+1) = pcom(ihru)%plg(ipl)%cpnm
              basin_plants = basin_plants + 1
            end if
          end do
        end do
      end do

      !! set all plants simulated in the basin
      allocate (plants_bsn(basin_plants))
      allocate (bsn_crop_yld(basin_plants))
      allocate (bsn_crop_yld_aa(basin_plants))

      !! zero basin crop yields and harvested areas
      do ipl_bsn = 1, basin_plants
        bsn_crop_yld(ipl_bsn) = bsn_crop_yld_z
        bsn_crop_yld_aa(ipl_bsn) = bsn_crop_yld_z
      end do

      plants_bsn = plts_bsn(1:basin_plants)
      deallocate (plts_bsn)
      do ihru = 1, sp_ob%hru
        do ipl = 1, pcom(ihru)%npl
          do ipl_bsn = 1, basin_plants
            if (pcom(ihru)%plg(ipl)%cpnm == plants_bsn(ipl_bsn)) then
              pcom(ihru)%plcur(ipl)%bsn_num = ipl_bsn
              exit
            end if
          end do
        end do
      end do
      
      end subroutine plant_all_init