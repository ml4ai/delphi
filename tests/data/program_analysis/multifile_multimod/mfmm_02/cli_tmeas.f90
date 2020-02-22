      subroutine cli_tmeas
      
      use input_file_module
      use climate_module
      use maximum_data_module
      use time_module
      
      implicit none
      
      character (len=80) :: titldum   !           |title of file
      character (len=80) :: header    !           |header of file
      integer :: eof                  !           |end of file
      integer :: imax                 !none       |determine max number for array (imax) and total number in file
      integer :: i                    !none       |counter 
      integer :: iyr                  !none       |number of years
      logical :: i_exist              !none       |check to determine if file exists
      integer :: istep                !           | 
      integer :: mtmp                 !           |
      real :: tempx                   !           |
      real :: tempn                   !           |
      integer :: iyr_prev             !none       |previous year
      integer :: iyrs                 !           |
       
       mtmp = 0
       eof = 0
       imax = 0

      !! read all measured daily temperature data
      inquire (file=in_cli%tmp_cli, exist=i_exist)
      if (.not. i_exist .or. in_cli%tmp_cli == "null") then
         allocate (tmp(0:0))
         allocate (tmp_n(0))
      else
      do 
        open (107,file=in_cli%tmp_cli)
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) header
        if (eof < 0) exit
 !       imax = 0
          do while (eof == 0)
            read (107,*,iostat=eof) titldum
            if (eof < 0) exit
            imax = imax + 1
          end do
          
      allocate (tmp(0:imax))
      allocate (tmp_n(imax))
      
      rewind (107)
      read (107,*,iostat=eof) titldum
      if (eof < 0) exit
      read (107,*,iostat=eof) header
      if (eof < 0) exit
      do i = 1, imax
        read (107,*,iostat = eof) tmp_n(i)
        if (eof < 0) exit
      end do
      
      rewind (107)
      read (107,*,iostat=eof) titldum
      if (eof < 0) exit
      read (107,*,iostat=eof) header
      if (eof < 0) exit
        
      do i = 1, imax
        read (107,*,iostat = eof) tmp(i)%filename
        if (eof < 0) exit
        
!!!!!weather path code
       if (in_path_tmp%tmp == "null") then
         open (108,file = tmp(i)%filename)
       else
         open (108,file = TRIM(ADJUSTL(in_path_tmp%tmp))//tmp(i)%filename)
       endif
!!!!!weather path code  
        
        read (108,*,iostat=eof) titldum
        if (eof < 0) exit
        read (108,*,iostat=eof) header
        if (eof < 0) exit
        read (108,*,iostat=eof) tmp(i)%nbyr, tmp(i)%tstep, tmp(i)%lat, tmp(i)%long,      &
                                     tmp(i)%elev

        if (eof < 0) exit
       
        allocate (tmp(i)%ts(366,tmp(i)%nbyr))
        allocate (tmp(i)%ts2(366,tmp(i)%nbyr))
        
        ! read and save start jd and yr
        read (108,*,iostat=eof) iyr, istep
        if (eof < 0) exit
        
        tmp(i)%start_day = istep
        tmp(i)%start_yr = iyr
        
        backspace (108)

      if (iyr > time%yrc) then
        tmp(i)%yrs_start = iyr - time%yrc
      else
        ! read and store entire year
        tmp(i)%yrs_start = 0
      end if
      
        ! read and store entire year
       do 
         read (108,*,iostat=eof) iyr, istep, tempx, tempn
         if (eof < 0) exit
         if (iyr == time%yrc .and. istep == time%day_start) exit
       end do

       backspace (108)
       iyr_prev = iyr
       iyrs = 1
       
       do 
         read (108,*,iostat=eof) iyr, istep, tmp(i)%ts(istep,iyrs),      &
                             tmp(i)%ts2(istep,iyrs)
         if (eof < 0) exit
         if (istep == 365 .or. istep == 366) then
           read (108,*,iostat=eof) iyr, istep
           if (eof < 0) exit
           backspace (108)
           if (iyr /= iyr_prev) then
             iyr_prev = iyr
             iyrs = iyrs + 1
           end if
         end if
       end do
       close (108)
              
       ! save end jd and year
       tmp(i)%end_day = istep
       tmp(i)%end_yr = iyr
       
      end do
      close (107)
      exit
      end do
      endif
      
      db_mx%tmpfiles = imax
      
      return
      end subroutine cli_tmeas