      subroutine soiltest_all_init
      
      use hru_module, only : hru, wfsh, ihru, sol_plt_ini
      use soil_module
      use plant_module
      use maximum_data_module
      use soil_data_module
      use hydrograph_module, only : sp_ob
      
      implicit none 
      
      integer :: isolt            !           | 
      integer :: isol_pl          !           |

      do ihru = 1, sp_ob%hru
        isol_pl = hru(ihru)%dbs%soil_plant_init
        isolt = sol_plt_ini(isol_pl)%nut
        if (isolt > 0) then
          call soiltest_init (ihru, isolt)
        end if
      end do
      
      return
      end subroutine soiltest_all_init