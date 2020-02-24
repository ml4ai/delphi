      subroutine hru_output(ihru)

      use plant_module
      use plant_data_module
      use time_module
      use basin_module
      use output_landscape_module
      use hydrograph_module, only : sp_ob1, ob
      use organic_mineral_mass_module
      
      implicit none
      
      integer, intent (in) :: ihru             !            |
      integer :: idp                           !            |
      integer :: j
      integer :: iob
      integer :: ipl
      real :: const
                         
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine outputs HRU variables on daily, monthly and annual time steps

      j = ihru
      
      iob = sp_ob1%hru + j - 1   !!!!!! added for new output write
          
        hwb_m(j) = hwb_m(j) + hwb_d(j)
        hnb_m(j) = hnb_m(j) + hnb_d(j)
        hls_m(j) = hls_m(j) + hls_d(j) 
        hpw_m(j) = hpw_m(j) + hpw_d(j)

      !! daily print
         if (pco%day_print == "y" .and. pco%int_day_cur == pco%int_day) then
          if (pco%wb_hru%d == "y") then
             write (2000,100) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_d(j)   !! waterbal
             if (pco%csvout == "y") then
            !! changed write unit below (2004 to write file data)
               write (2004,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_d(j)  !! waterbal
               !write (4015,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_d(j)  !! waterbal
             end if
          end if
          if (pco%nb_hru%d == "y") then
            !write (2020,*) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_d(j)  !! nutrient bal
            write (2020,104) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_d(j)  !! nutrient bal
              if (pco%csvout == "y") then
                write (2024,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_d(j)  !! nutrient bal
              end if
          end if
          if (pco%ls_hru%d == "y") then
            write (2030,102) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_d(j)  !! losses
              if (pco%csvout == "y") then
                write (2034,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_d(j)  !! losses
              end if
          end if
          if (pco%pw_hru%d == "y") then
            write (2040,101) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_d(j)  !! plant weather 
              if (pco%csvout == "y") then 
                write (2044,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_d(j)  !! plant weather
              end if 
          end if
        end if

        !! check end of month
        if (time%end_mo == 1) then
          hwb_y(j) = hwb_y(j) + hwb_m(j)
          hnb_y(j) = hnb_y(j) + hnb_m(j)
          hls_y(j) = hls_y(j) + hls_m(j)
          hpw_y(j) = hpw_y(j) + hpw_m(j)
          
          const = float (ndays(time%mo + 1) - ndays(time%mo))
          hpw_m(j) = hpw_m(j) // const
          hwb_m(j) = hwb_m(j) // const
          
          
          !! monthly print
           if (pco%wb_hru%m == "y") then
             write (2001,100) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_m(j)
               if (pco%csvout == "y") then
                 write (2005,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_m(j)
               end if
           end if
           if (pco%nb_hru%m == "y") then
             write (2021,104) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_m(j)
               if (pco%csvout == "y") then
                 write (2025,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_m(j)
               end if
           end if
           if (pco%ls_hru%m == "y") then
             write (2031,102) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_m(j)
               if (pco%csvout == "y") then 
                 write (2035,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_m(j)
               end if
           end if
           if (pco%pw_hru%m == "y") then
             write (2041,101) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_m(j)
               if (pco%csvout == "y") then 
                 write (2045,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_m(j)
               end if 
           end if
          
          hwb_m(j) = hwbz
          hnb_m(j) = hnbz
          hpw_m(j) = hpwz
          hls_m(j) = hlsz
        end if
        
        !! check end of year
        if (time%end_yr == 1) then
          hwb_a(j) = hwb_a(j) + hwb_y(j)
          hnb_a(j) = hnb_a(j) + hnb_y(j)
          hls_a(j) = hls_a(j) + hls_y(j)
          hpw_a(j) = hpw_a(j) + hpw_y(j)
          
          const = time%day_end_yr
          hwb_y(j) = hwb_y(j) // const
          hpw_y(j) = hpw_y(j) // const
          
          !! yearly print
           if (time%end_yr == 1 .and. pco%wb_hru%y == "y") then
             write (2002,100) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_y(j)
               if (pco%csvout == "y") then
                 write (2006,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_y(j)
               end if
           end if
           if (time%end_yr == 1 .and. pco%nb_hru%y == "y") then
             write (2022,104) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_y(j)
               if (pco%csvout == "y") then
                 write (2026,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_y(j)
               end if
           end if
           if (time%end_yr == 1 .and. pco%ls_hru%y == "y") then
             write (2032,102) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_y(j)
               if (pco%csvout == "y") then
                 write (2036,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_y(j)
               end if
           end if
           if (time%end_yr == 1 .and. pco%pw_hru%y == "y") then
             write (2042,101) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_y(j)
               if (pco%csvout == "y") then 
                 write (2046,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_y(j)
               end if 
           end if
          
        end if
        
!!!!! average annual print
         if (time%end_sim == 1 .and. pco%wb_hru%a == "y") then
           hwb_a(j) = hwb_a(j) / time%yrs_prt
           hwb_a(j) = hwb_a(j) // time%days_prt
           write (2003,100) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_a(j)
           if (pco%csvout == "y") then
             write (2007,100) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hwb_a(j)
           end if
           hwb_a(j) = hwbz
         end if
        
         if (time%end_sim == 1 .and. pco%nb_hru%a == "y") then 
           hnb_a(j) = hnb_a(j) / time%yrs_prt
           write (2023,104) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_a(j)
             if (pco%csvout == "y") then 
               write (2027,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hnb_a(j)
             end if
             hnb_a(j) = hnbz
         end if
        
         if (time%end_sim == 1 .and. pco%ls_hru%a == "y") then
           hls_a(j) = hls_a(j) / time%yrs_prt 
           write (2033,101) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_a(j)
             if (pco%csvout == "y") then 
               write (2037,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hls_a(j)
             end if
             hls_a(j) = hlsz
         end if
        
         if (time%end_sim == 1 .and. pco%pw_hru%a == "y") then     
           hpw_a(j) = hpw_a(j) / time%yrs_prt
           hpw_a(j) = hpw_a(j) // time%days_prt
           write (2043,102) time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_a(j)
             if (pco%csvout == "y") then 
               write (2047,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j, ob(iob)%gis_id, ob(iob)%name, hpw_a(j)
             end if
             hpw_a(j) = hpwz
         end if

         if (time%end_sim == 1) then
           do ipl = 1, pcom(j)%npl
             idp = pcom(j)%plcur(ipl)%idplt
             if (pcom(j)%plcur(ipl)%harv_num > 0) then 
               pl_mass(j)%yield_tot(ipl) = pl_mass(j)%yield_tot(ipl) / float(pcom(j)%plcur(ipl)%harv_num)
             endif
            write (4008,103) time%day, time%mo, time%day_mo, time%yrc, j,pldb(idp)%plantnm, pl_mass(j)%yield_tot(ipl)
            if (pco%csvout == "y") then
              write (4009,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, j,pldb(idp)%plantnm, pl_mass(j)%yield_tot(ipl) 
            end if
           end do
         end if
      return
      
100   format (4i6,2i8,2x,a,28f12.3)
101   format (4i6,2i8,2x,a,20f12.3)
102   format (4i6,2i8,2x,a,20f12.3)
103   format (4i6,i8,4x,a,5x,4f12.3)
104   format (4i6,2i8,2x,a8,4f12.3,23f17.3)
       
      end subroutine hru_output