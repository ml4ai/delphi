      subroutine recall_read

      use hydrograph_module
      use input_file_module
      use organic_mineral_mass_module
      use constituent_mass_module
      use maximum_data_module
      use time_module
      use exco_module
      
      implicit none      
 
      character (len=80) :: titldum   !           |title of file
      character (len=80) :: header    !           |header of file
      integer :: imax                 !none       |end of loop
      integer :: iyr                  !           |
      integer :: eof                  !           |end of file
      logical :: i_exist              !none       |check to determine if file exists
      integer :: nbyr                 !none       !number of years the land use occurred 
      integer :: k                    !           |
      integer :: iyrs                 !           | 
      integer :: iyr_prev             !none       |previous year
      integer :: istep                !           | 
      integer :: ipestcom_db          !none       !pointer to pestcom_db - fix*** ?? 
      integer :: ipc                  !none       |counter
      integer :: ii                   !none       |counter
      integer :: i                    !           |
      integer :: iexco_om
      integer :: ifirst               !           |

      eof = 0
      imax = 0

      !read all recall files
      inquire (file=in_rec%recall_rec, exist=i_exist)
      if (i_exist .or. in_rec%recall_rec /= "null") then
      do
        open (107,file=in_rec%recall_rec)
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) header
        if (eof < 0) exit
        imax = 0
          do while (eof == 0)
            read (107,*,iostat=eof) i
            if (eof < 0) exit
            imax = Max(imax,i) 
          end do
          
      allocate (recall(0:imax))
      allocate (rec_d(imax))
      allocate (rec_m(imax))
      allocate (rec_y(imax))
      allocate (rec_a(imax))
      
      rewind (107)
      read (107,*,iostat=eof) titldum
      if (eof < 0) exit
      read (107,*,iostat=eof) header
      if (eof < 0) exit
      
      do ii = 1, imax
        read (107,*,iostat=eof) i
        if (eof < 0) exit
        backspace (107)
 !       read (107,*,iostat = eof) k, rec_om(i)%name, rec_om(i)%typ, rec_om(i)%filename
        read (107,*,iostat = eof) k, recall(i)%name, recall(i)%typ, recall(i)%filename
        if (eof < 0) exit
        
        if (recall(i)%typ /= 4) then
          open (108,file = recall(i)%filename)
          read (108,*,iostat=eof) titldum
          if (eof < 0) exit
          read (108,*,iostat=eof) nbyr
          if (eof < 0) exit
          read (108,*,iostat=eof) header
          if (eof < 0) exit 
        
!        select case (rec_om(i)%typ)
        select case (recall(i)%typ)
           case (1) !! daily
!            allocate (rec_om(i)%hd_om(366,nbyr))
            allocate (recall(i)%hd(366,nbyr))
            
           case (2) !! monthly
            !allocate (rec_om(i)%hd_om(12,nbyr))
            allocate (recall(i)%hd(12,nbyr))
            
           case (3) !! annual
            !allocate (rec_om(i)%hd_om(1,nbyr))
            allocate (recall(i)%hd(1,nbyr))

        end select
           
        ! read and store entire year
       ! ifirst = 0
       do 
         read (108,*,iostat=eof) iyr, istep
         if (ifirst == 0) then
           recall(i)%start_ts = istep
           recall(i)%start_yr = iyr
           ifirst = 1
         end if
         if (eof < 0) exit
         if (iyr == time%yrc) exit
       end do
       recall(i)%end_ts = istep
       recall(i)%end_yr = iyr
       
       backspace (108)
       iyr_prev = iyr
       iyrs = 1
       
       do
         iyr_prev = iyr
 !        read (108,*,iostat=eof) iyr, istep, rec_om(i)%hd_om(istep,iyrs)
         read (108,*,iostat=eof) iyr, istep, recall(i)%hd(istep,iyrs)
         if (eof < 0) exit
         !call hyd_convert_mass (rec_om(i)%hd_om(istep,iyrs))
         !check to see when next year
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
      else
          
     if (recall(i)%typ == 4) then
        allocate (recall(i)%hd(1,1))
        !! xwalk with exco file to get sequential number
        do iexco_om = 1, db_mx%exco_om
          if (exco_db(iexco_om)%name == recall(i)%filename) then
            recall(i)%hd(1,1) = exco(iexco_om)
            exit
          end if
        end do
     end if 
     end if
      end do
      close (107)
      exit
      enddo
      endif
      
      !read all rec_pest files
      inquire (file="pest.com", exist=i_exist)
      if (i_exist ) then
      do
        open (107,file="pest.com")
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) header
        if (eof < 0) exit
        imax = 0
          do while (eof == 0)
            read (107,*,iostat=eof) i
            if (eof < 0) exit
            imax = Max(imax,i) 
          end do
          
      allocate (rec_pest(0:imax))
      rewind (107)
      read (107,*,iostat=eof) titldum
      if (eof < 0) exit
      read (107,*,iostat=eof) header
      if (eof < 0) exit
      
      do ipc = 1, db_mx%pestcom
        read (107,*,iostat=eof) ipestcom_db   !pointer to pestcom_db - fix***
        if (eof < 0) exit

         do ii = 1, imax
           read (107,*,iostat=eof) i
           if (eof < 0) exit
           backspace (107)
           read (107,*,iostat = eof) k, rec_pest(i)%name, rec_pest(i)%typ, rec_pest(i)%filename
           if (eof < 0) exit
           open (108,file = rec_pest(i)%filename)
           read (108,*,iostat=eof) titldum
           if (eof < 0) exit
           read (108,*,iostat=eof) nbyr
           if (eof < 0) exit
           read (108,*,iostat=eof) header
           if (eof < 0) exit
        
        select case (rec_pest(i)%typ)
           case (1) !! daily
            allocate (rec_pest(i)%hd_pest(366,nbyr))
            
           case (2) !! monthly
            allocate (rec_pest(i)%hd_pest(12,nbyr))
            
           case (3) !! annual
            allocate (rec_pest(i)%hd_pest(1,nbyr))
        end select
           
        ! read and store entire year
       do 
         read (108,*,iostat=eof) iyr, istep
         if (eof < 0) exit
         if (iyr == time%yrc) exit
       end do
       
       backspace (108)
       iyr_prev = iyr
       iyrs = 1
       
       do 
         read (108,*,iostat=eof) iyr, istep, recall(i)%hd(istep,iyrs)
         if (eof < 0) exit
         !call hyd_convert_mass (recall(i)%hd(istep,iyrs))
         if (iyr /= iyr_prev) then
           iyr_prev = iyr
           iyrs = iyrs + 1
         endif
       end do
       close (108)
         end do 
        end do
        close (107)
      end do
      end if     
      
      return
      end subroutine recall_read