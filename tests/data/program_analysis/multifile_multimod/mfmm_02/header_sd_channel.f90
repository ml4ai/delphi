      subroutine header_sd_channel

      use sd_channel_module
      use basin_module
      use hydrograph_module
      
      implicit none 

!!!  SWAT-DEG CHANNEL
      if (sp_ob%chandeg > 0) then
        if (pco%sd_chan%d == "y") then
          open (2500,file="channel_sd_day.txt",recl = 1500)
          write (2500,*) bsn%name, prog
          write (2500,*) ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
          write (2500,*) ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
          write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_day.txt"
          if (pco%csvout == "y") then
            open (2504,file="channel_sd_day.csv",recl = 1500)
            write (2504,*) bsn%name, prog
            write (2504,'(*(G0.3,:,","))') ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
            write (2504,'(*(G0.3,:,","))') ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
            write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_day.csv"
          end if
        endif
      endif
      
        if (sp_ob%chandeg > 0) then
          if (pco%sd_chan%m == "y") then  
          open (2501,file="channel_sd_mon.txt",recl = 1500)
          write (2501,*) bsn%name, prog
          write (2501,*) ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
          write (2501,*) ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
          write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_mon.txt"
          if (pco%csvout == "y") then
            open (2505,file="channel_mon_sd.csv",recl = 1500)
            write (2505,*) bsn%name, prog
            write (2505,'(*(G0.3,:,","))') ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr  
            write (2505,'(*(G0.3,:,","))') ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
            write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_mon.csv"
          end if
          end if
         end if 
        
      if (sp_ob%chandeg > 0) then
        if (pco%sd_chan%y == "y") then
          open (2502,file="channel_sd_yr.txt",recl = 1500)
          write (2502,*) bsn%name, prog
          write (2502,*) ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
          write (2502,*) ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
          write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_yr.txt"
          if (pco%csvout == "y") then
            open (2506,file="channel_sd_yr.csv",recl = 1500)
            write (2506,*) bsn%name, prog
            write (2506,'(*(G0.3,:,","))') ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
            write (2506,'(*(G0.3,:,","))') ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
            write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_yr.csv"
          end if
        endif
      endif
      
        if (sp_ob%chandeg > 0) then
          if (pco%sd_chan%a == "y") then
          open (2503,file="channel_sd_aa.txt",recl = 1500)
          write (2503,*) bsn%name, prog
          write (2503,*) ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr
          write (2503,*) ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
          write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_aa.txt"
          if (pco%csvout == "y") then
            open (2507,file="channel_sd_aa.csv",recl = 1500)
            write (2507,*) bsn%name, prog
            write (2507,'(*(G0.3,:,","))') ch_wbod_hdr, hyd_stor_hdr, hyd_in_hdr, hyd_out_hdr   
            write (2507,'(*(G0.3,:,","))') ch_wbod_hdr_units, hyd_hdr_units1, hyd_hdr_units1, hyd_hdr_units1
            write (9000,*) "SWAT-DEG_CHANNEL          channel_sd_aa.csv"
          end if
          end if
         end if 
         
!!!!!!!! SD_CHANMORPH
      if (sp_ob%chandeg > 0) then
        if (pco%sd_chan%d == "y") then
          open (4800,file="channel_sdmorph_day.txt",recl = 1500)
          write (4800,*) bsn%name, prog
          write (4800,*) sdch_hdr !! swat deg channel morph
          write (4800,*) sdch_hdr_units
          write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_day.txt"
          if (pco%csvout == "y") then
            open (4804,file="channel_sdmorph_day.csv",recl = 1500)
            write (4804,*) bsn%name, prog
            write (4804,'(*(G0.3,:,","))') sdch_hdr 
            write (4804,'(*(G0.3,:,","))') sdch_hdr_units
            write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_day.csv"
          end if
        endif
      endif
      
        if (sp_ob%chandeg > 0) then
          if (pco%sd_chan%m == "y") then  
          open (4801,file="channel_sdmorph_mon.txt",recl = 1500)
          write (4801,*) bsn%name, prog
          write (4801,*) sdch_hdr   !! swat deg channel morph
          write (4801,*) sdch_hdr_units
          write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_mon.txt"
          if (pco%csvout == "y") then
            open (4805,file="channel_mon_sdmorph.csv",recl = 1500)
            write (4805,*) bsn%name, prog
            write (4805,'(*(G0.3,:,","))') sdch_hdr   
            write (4805,'(*(G0.3,:,","))') sdch_hdr_units
            write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_mon.csv"
          end if
          end if
         end if 
        
      if (sp_ob%chandeg > 0) then
        if (pco%sd_chan%y == "y") then
          open (4802,file="channel_sdmorph_yr.txt",recl = 1500)
          write (4802,*) bsn%name, prog
          write (4802,*) sdch_hdr !! swat deg channel morph
          write (4802,*) sdch_hdr_units
          write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_yr.txt"
          if (pco%csvout == "y") then
            open (4806,file="channel_sdmorph_yr.csv",recl = 1500)
            write (4806,*) bsn%name, prog
            write (4806,'(*(G0.3,:,","))') sdch_hdr !! swat deg channel morph csv
            write (4806,'(*(G0.3,:,","))') sdch_hdr_units
            write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_yr.csv"
          end if
        endif
      endif
      
        if (sp_ob%chandeg > 0) then
          if (pco%sd_chan%a == "y") then
          open (4803,file="channel_sdmorph_aa.txt",recl = 1500)
          write (4803,*) bsn%name, prog
          write (4803,*) sdch_hdr   !! swat deg channel morph
          write (4803,*) sdch_hdr_units
          write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_aa.txt"
          if (pco%csvout == "y") then
            open (4807,file="channel_sdmorph_aa.csv",recl = 1500)
            write (4807,*) bsn%name, prog
            write (4807,'(*(G0.3,:,","))') sdch_hdr   
            write (4807,'(*(G0.3,:,","))') sdch_hdr_units
            write (9000,*) "SWAT-DEG_CHANNEL_MORPH    channel_sdmorph_aa.csv"
          end if
          end if
         end if 
!!!!!!!! SD_CHANMORPH
       
      return
      end subroutine header_sd_channel