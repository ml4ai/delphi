      subroutine ls_link
    
      use hydrograph_module
      
      implicit none
      
      integer :: isdc            !none    |counter
      integer :: i               !        |  
    
     !set parms for sd-channel-landscape linkage
      do isdc = 1, sp_ob%chandeg
        i = sp_ob1%chandeg + isdc - 1
        if (ob(i)%props2 > 0) then
          call sd_channel_surf_link (ob(i)%props, ob(i)%props2)
        end if
      end do
      return
      end subroutine ls_link