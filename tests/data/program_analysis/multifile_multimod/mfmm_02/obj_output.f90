      subroutine obj_output
      
      use time_module
      use hydrograph_module
      
      implicit none
        
      integer :: ihd           !            |
      integer :: iob           !            | 
      integer :: iunit         !            |
      integer :: itot          !none        |counter
      
      do
        do itot = 1, mobj_out
          iob = ob_out(itot)%objno
          ihd = ob_out(itot)%hydno
          iunit = ob_out(itot)%unitno          
                
          if (iob <= sp_ob%objs) then
            write (iunit+itot,*) time%day, time%mo, time%day_mo, time%yrc,  ob(itot)%name,  ob_out(itot)%obtyp, ob(iob)%hd(ihd)  
!            write (iunit+itot,100) time%day, time%yrc, ob_out(itot)%obtyp, ob_out(itot)%obtypno, ob(iob)%hd(ihd)  
          end if

        end do
        exit
      end do
      
      return
         
!      100   format (4i8,a18,i18, 1(1x, f17.3), 24(1x,e17.3))

      end subroutine obj_output