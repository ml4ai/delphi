     subroutine header_snutc
    
     use basin_module

     implicit none 
!!   open mgt.out file 
      if (pco%snutc == "d" .or. pco%snutc == "m" .or. pco%snutc == "y" .or. pco%snutc == "a") then
        open (2610,file="soil_nutcarb_out.txt",recl=800)
        write (2610,*) bsn%name, prog
        write (2610,*) snutc_hdr
        write (9000,*) "SNUTC                     soil_nutcarb_out.txt"
      end if
          
      return
      end subroutine header_snutc  