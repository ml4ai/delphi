      subroutine lsu_output
      
      use time_module
      use basin_module
      use maximum_data_module
      use calibration_data_module
      use hydrograph_module
      use output_landscape_module
      
      implicit none
      
      integer :: ilsu          !none        |counter
      integer :: ielem         !none        |counter
      integer :: ihru          !none        |counter 
      integer :: iob           !none        |counter 
      real :: const            !            |  
                            
      do ilsu = 1, db_mx%lsu_out
        ! summing HRU output for the landscape unit(
        do ielem = 1, lsu_out(ilsu)%num_tot
          ihru = lsu_out(ilsu)%num(ielem)
          iob = sp_ob1%ru + ilsu - 1
          if (lsu_elem(ihru)%ru_frac > 1.e-9) then
            const = lsu_elem(ihru)%ru_frac
            if (lsu_elem(ihru)%obtyp == "hru") then
              ruwb_d(ilsu) = ruwb_d(ilsu) + hwb_d(ihru) * const
              runb_d(ilsu) = runb_d(ilsu) + hnb_d(ihru) * const
              ruls_d(ilsu) = ruls_d(ilsu) + hls_d(ihru) * const
              rupw_d(ilsu) = rupw_d(ilsu) + hpw_d(ihru) * const
            end if
            ! summing HRU_LTE output
            if (lsu_elem(ihru)%obtyp == "hlt") then
              ruwb_d(ilsu) = ruwb_d(ilsu) + hltwb_d(ihru) * const
              runb_d(ilsu) = runb_d(ilsu) + hltnb_d(ihru) * const
              ruls_d(ilsu) = ruls_d(ilsu) + hltls_d(ihru) * const
              rupw_d(ilsu) = rupw_d(ilsu) + hltpw_d(ihru) * const
            end if
          end if
        end do    !ielem
      
        !! sum monthly variables
        ruwb_m(ilsu) = ruwb_m(ilsu) + ruwb_d(ilsu)
        runb_m(ilsu) = runb_m(ilsu) + runb_d(ilsu)
        ruls_m(ilsu) = ruls_m(ilsu) + ruls_d(ilsu)
        rupw_m(ilsu) = rupw_m(ilsu) + rupw_d(ilsu)
        
        !! daily print - LANDSCAPE UNIT
         if (pco%day_print == "y" .and. pco%int_day_cur == pco%int_day) then
          if (pco%wb_lsu%d == "y") then
            write (2140,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_d(ilsu)  !! waterbal
              if (pco%csvout == "y") then 
                write (2144,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_d(ilsu)  !! waterbal
              end if 
          end if 
          if (pco%nb_lsu%d == "y") then
            write (2150,103) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_d(ilsu)  !! nutrient bal
            if (pco%csvout == "y") then 
              write (2154,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_d(ilsu)  !! nutrient bal
            end if 
          end if
          if (pco%ls_lsu%d == "y") then
            write (2160,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_d(ilsu)  !! losses
            if (pco%csvout == "y") then 
              write (2164,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_d(ilsu)  !! losses
            end if 
          end if
          if (pco%pw_lsu%d == "y") then
            write (2170,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_d(ilsu)  !! plant weather
            if (pco%csvout == "y") then 
              write (2175,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_d(ilsu)  !! plant weather 
            end if
          end if 
        end if

        ruwb_d(ilsu) = hwbz
        runb_d(ilsu) = hnbz
        ruls_d(ilsu) = hlsz
        rupw_d(ilsu) = hpwz
        
        !! check end of month
        if (time%end_mo == 1) then
          ruwb_y(ilsu) = ruwb_y(ilsu) + ruwb_m(ilsu)
          runb_y(ilsu) = runb_y(ilsu) + runb_m(ilsu)
          ruls_y(ilsu) = ruls_y(ilsu) + ruls_m(ilsu)
          rupw_y(ilsu) = rupw_y(ilsu) + rupw_m(ilsu)
          
          const = float (ndays(time%mo + 1) - ndays(time%mo)) 
          rupw_m(ilsu) = rupw_m(ilsu) // const
          ruwb_m(ilsu) = ruwb_m(ilsu) // const 
          
          if (pco%wb_lsu%m == "y") then
            write (2141,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_m(ilsu)
            if (pco%csvout == "y") then 
              write (2145,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_m(ilsu)
            end if 
          end if
          if (pco%nb_lsu%m == "y") then 
            write (2151,103) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_m(ilsu)
            if (pco%csvout == "y") then 
              write (2155,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_m(ilsu)
            end if 
          end if
          if (pco%ls_lsu%m == "y") then
            write (2161,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_m(ilsu)
            if (pco%csvout == "y") then 
              write (2165,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_m(ilsu)
            end if 
          end if
          if (pco%pw_lsu%m == "y") then
            write (2171,100) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_m(ilsu)
            if (pco%csvout == "y") then 
              write (2175,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_m(ilsu)
            end if 
          end if
  
          ruwb_m(ilsu) = hwbz
          runb_m(ilsu) = hnbz
          ruls_m(ilsu) = hlsz
          rupw_m(ilsu) = hpwz
        end if

        !!  check end of year
        if (time%end_yr == 1) then
           ruwb_a(ilsu) = ruwb_a(ilsu) + ruwb_y(ilsu)
           runb_a(ilsu) = runb_a(ilsu) + runb_y(ilsu)
           ruls_a(ilsu) = ruls_a(ilsu) + ruls_y(ilsu)
           rupw_a(ilsu) = rupw_a(ilsu) + rupw_y(ilsu)
           const = time%day_end_yr
           ruwb_y(ilsu) = ruwb_y(ilsu) // const
           rupw_y(ilsu) = rupw_y(ilsu) // const

           if (pco%wb_lsu%y == "y") then
             write (2142,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_y(ilsu)
             if (pco%csvout == "y") then 
               write (2146,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_y(ilsu)
             end if 
           end if
           if (pco%nb_lsu%y == "y") then
             write (2152,103) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_y(ilsu)
             if (pco%csvout == "y") then 
               write (2156,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_y(ilsu)
             end if 
           end if
           if (pco%ls_lsu%y == "y") then
             write (2162,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_y(ilsu)
             if (pco%csvout == "y") then 
               write (2166,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_y(ilsu)
             end if 
           end if
           if (pco%pw_lsu%y == "y") then
             write (2172,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_y(ilsu)
             if (pco%csvout == "y") then 
               write (2176,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_y(ilsu)
             end if 
           end if
 
          !! zero yearly variables        
          ruwb_y(ilsu) = hwbz
          runb_y(ilsu) = hnbz
          ruls_y(ilsu) = hlsz
          rupw_y(ilsu) = hpwz
        end if
        
      !! average annual print - LANDSCAPE UNIT
      if (time%end_sim == 1 .and. pco%wb_lsu%a == "y") then
        ruwb_a(ilsu) = ruwb_a(ilsu) / time%yrs_prt
        ruwb_a(ilsu) = ruwb_a(ilsu) // time%days_prt
        write (2143,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_a(ilsu)
        if (pco%csvout == "y") then 
          write (2147,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruwb_a(ilsu)
        end if 
      end if
      if (time%end_sim == 1 .and. pco%nb_lsu%a == "y") then
        runb_a(ilsu) = runb_a(ilsu) / time%yrs_prt
        write (2153,103) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_a(ilsu)
        if (pco%csvout == "y") then 
          write (2157,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, runb_a(ilsu)
        end if
      end if
      if (time%end_sim == 1 .and. pco%ls_lsu%a == "y") then     
        ruls_a(ilsu) = ruls_a(ilsu) / time%yrs_prt
        write (2163,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_a(ilsu)
        if (pco%csvout == "y") then 
          write (2167,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, ruls_a(ilsu)
        end if 
      end if
      if (time%end_sim == 1 .and. pco%pw_lsu%a == "y") then    
        rupw_a(ilsu) = rupw_a(ilsu) / time%yrs_prt
        rupw_a(ilsu) = rupw_a(ilsu) // time%days_prt
        write (2173,102) time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_a(ilsu) 
        if (pco%csvout == "y") then 
          write (2177,'(*(G0.3,:","))') time%day, time%mo, time%day_mo, time%yrc, ilsu, ob(iob)%gis_id, ob(iob)%name, rupw_a(ilsu)
        end if
      end if
      end do    !ilsu
      
      return
      
100   format (1x,4i6,i7,i8,2x,a,28f12.3)
102   format (1x,4i6,i7,i8,2x,a,28f12.3)
103   format (4i6,i8,i8,2x,a,6f12.3,22f17.3)
104   format (4i6,i8,i8,2x,a,6f12.3,22f17.3)
       
      end subroutine lsu_output