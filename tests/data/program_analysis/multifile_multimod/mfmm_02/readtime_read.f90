       subroutine readtime_read 
      
       use time_module
       use input_file_module
       
       implicit none
       
       character (len=80) :: titldum    !           |title of file
       character (len=500) :: header    !           |header of file       
       integer :: eof                   !           |end of file
       logical :: i_exist              !none       |check to determine if file exists
       
       eof = 0

       !! read weather codes
       inquire (file=in_sim%time, exist=i_exist)
       if (i_exist .or. in_sim%time /= "null") then   
       do
         open (107,file=in_sim%time)
         read (107,*,iostat=eof) titldum
         if (eof < 0) exit
         read (107,*,iostat=eof) header
         if (eof < 0) exit
         read (107,*,iostat=eof) time%day_start, time%yrc_start, time%day_end, time%yrc_end, time%step
         if (eof < 0) exit
         if (time%day_start <= 0) time%day_start = 1
         time%nbyr = time%yrc_end - time%yrc_start + 1
         call xmon
         time%mo_start = time%mo
         if (eof < 0) exit
         exit
         close (107) 
       enddo
       endif
      
       time%yrc = time%yrc_start
       
       return
       end subroutine readtime_read            