      subroutine res_read_hyd
      
      use basin_module
      use input_file_module
      use maximum_data_module
      use reservoir_data_module
      
      implicit none     
         
      character (len=80) :: titldum     !             |title of file
      character (len=80) :: header      !             |header of file
      integer :: eof                    !             |end of file
      integer :: imax                   !             |determine max number for array (imax) and total number in file
      logical :: i_exist                !none         |check to determine if file exists
      integer :: i                      !none         |counter
      integer :: ires                   !none         |counter
      
      eof = 0
      imax = 0

      inquire (file=in_res%hyd_res, exist=i_exist)
      if (.not. i_exist .or. in_res%hyd_res == "null") then
        allocate (res_hyd(0:0))
      else   
      do
       open (105,file=in_res%hyd_res)
       read (105,*,iostat=eof) titldum
       if (eof < 0) exit
       read (105,*,iostat=eof) header
       if (eof < 0) exit
        do while (eof == 0)
          read (105,*,iostat=eof) titldum
          if (eof < 0) exit
          imax = imax + 1
        end do
        
      db_mx%res_hyd = imax
      
      allocate (res_hyd(0:imax))
      rewind (105)
      read (105,*,iostat=eof) titldum
      if (eof < 0) exit
      read (105,*,iostat=eof) header
      if (eof < 0) exit
      
       do ires = 1, imax
         
         !read (105,*,iostat=eof) titldum
         !backspace (105)
         read (105,*,iostat=eof) res_hyd(ires)
         if (eof < 0) exit

        if (res_hyd(ires)%pvol + res_hyd(ires)%evol > 0.) then
          if(res_hyd(ires)%pvol <= 0) res_hyd(ires)%pvol = 0.9 * res_hyd(ires)%evol
        else
          if (res_hyd(ires)%pvol <= 0) res_hyd(ires)%pvol = 60000.0
        end if
        if (res_hyd(ires)%evol <= 0.0) res_hyd(ires)%evol = 1.11 *res_hyd(ires)%pvol
        if (res_hyd(ires)%psa <= 0.0) res_hyd(ires)%psa = 0.08 * res_hyd(ires)%pvol
        if (res_hyd(ires)%esa <= 0.0) res_hyd(ires)%esa = 1.5 * res_hyd(ires)%psa
        if (res_hyd(ires)%evrsv <= 0.) res_hyd(ires)%evrsv = 0.6

       end do
       close (105)
      exit
      enddo
      endif
  
      return
      end subroutine res_read_hyd