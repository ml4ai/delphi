      subroutine overbank_read
    
      use hydrograph_module
      use input_file_module
      use maximum_data_module
      implicit none 

      character (len=80) :: titldum   !           |title of file
      character (len=80) :: header    !           |header of file
      character (len=16) :: namedum   !           |
      character (len=3)::obtyp
      integer :: eof                  !           |end of file
      integer :: imax                 !none       |determine max number for array (imax) and total number in file
      integer :: nspu                 !           |
      logical :: i_exist              !none       |check to determine if file exists
      integer :: max                  !           |
      integer :: mcha_sp              !           |
      integer :: i                    !none       |counter
      integer :: isp                  !none       |counter
      integer :: numb                 !           |
      integer :: ise                  !none       |counter
      integer :: ichan  
          
      eof = 0
      imax = 0
      
    !!read data for surface elements in the floodplain-for overbank flooding
      inquire (file=in_link%chan_surf, exist=i_exist)
      if (i_exist .or. in_link%chan_surf /= "null") then
      do
        open (107,file=in_link%chan_surf)
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) mcha_sp
        if (eof < 0) exit
        read (107,*,iostat=eof) header
        if (eof < 0) exit
        imax = 0
        do while (eof == 0)
          read (107,*,iostat=eof) i
          if (eof < 0) exit
          imax = imax + 1
        end do
        imax=max(imax, mcha_sp)
          
        allocate (ch_sur(imax))
        rewind (107)
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) mcha_sp
        if (eof < 0) exit
        read (107,*,iostat=eof) header
        if (eof < 0) exit

        db_mx%ch_surf = imax
        
                !db_mx%ch_surf
        do ise = 1, imax
          read (107,*,iostat=eof) i, ichan, namedum, nspu
          if (eof < 0) exit
          allocate (ch_sur(i)%obtyp(nspu))
          allocate (ch_sur(i)%obtypno(nspu))
          allocate (ch_sur(i)%wid(nspu))
          allocate (ch_sur(i)%dep(nspu))
          allocate (ch_sur(i)%flood_volmx(nspu))
          allocate (ch_sur(i)%hd(nspu))
        
          if (nspu > 0) then
            backspace (107)
            read (107,*,iostat=eof) numb, ch_sur(i)%chnum, ch_sur(i)%name,    &
            ch_sur(i)%num, (ch_sur(i)%obtyp(isp), ch_sur(i)%obtypno(isp), isp = 1, nspu)
            if (eof < 0) exit
          end if

        end do
        exit
      end do
      close (107)
      end if
      return
      end subroutine overbank_read