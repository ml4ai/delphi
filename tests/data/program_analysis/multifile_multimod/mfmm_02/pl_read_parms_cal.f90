       subroutine pl_read_parms_cal
      
       use calibration_data_module
       use input_file_module
       
       implicit none        
      
       character (len=80) :: titldum   !           |title of file
       character (len=80) :: header    !           |header of file
       integer :: eof                  !           |end of file
       logical :: i_exist              !none       |check to determine if file exists
       integer :: mlsp                 !none       |end of loop
       integer :: i                    !none       |counter
       
       eof = 0

       inquire (file=in_chg%plant_parms_sft, exist=i_exist)
       if (.not. i_exist .or. in_chg%plant_parms_sft == "null") then
        allocate (pl_prms(0:0))	   	   
       else   
       do 
         open (107,file=in_chg%plant_parms_sft)
         read (107,*,iostat=eof) titldum
         if (eof < 0) exit
         read (107,*,iostat=eof) mlsp
         if (eof < 0) exit
         read (107,*,iostat=eof) header
         if (eof < 0) exit
         allocate (pl_prms(mlsp))
         exit
       enddo
       
       do i = 1, mlsp
         read (107,*,iostat=eof) pl_prms(i)%name, pl_prms(i)%chg_typ, pl_prms(i)%neg, pl_prms(i)%pos, pl_prms(i)%lo, pl_prms(i)%up
         if (eof < 0) exit 
       end do

       end if	   
    
       close(107)
       return
       end subroutine pl_read_parms_cal