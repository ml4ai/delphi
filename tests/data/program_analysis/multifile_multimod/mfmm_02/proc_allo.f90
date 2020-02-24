      subroutine proc_allo
    
      use hydrograph_module
      
      implicit none

      call ru_allo

      call aqu_read
      call aqu_initial
      call aqu_read_init
      
	  return
      
      end subroutine proc_allo