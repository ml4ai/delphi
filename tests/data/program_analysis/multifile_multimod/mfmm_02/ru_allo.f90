      subroutine ru_allo
      
      use basin_module
      use time_module
      use ru_module
      use hydrograph_module, only : sp_ob
      
      implicit none       
      
      integer :: isb        !none        |counter
      real :: ql            !            |  
      real :: sumq          !            |
      real :: tb            !            |
      real :: tp            !none        !time to peak flow 
      integer :: i          !none        |counter
      integer :: int        !            |
      real :: xi            !            | 
      real :: q             !            |
      integer :: max        !            |maximum 
      real :: uhalpha       !none        !alpha coeff for est unit hydrograph using gamma func
 
      if (time%step > 0) then
!!    compute unit hydrograph for computing subbasin hydrograph from direct runoff
      do isb = 1, sp_ob%ru
        ql = 0.
        sumq = 0.
        tb = .5 + .6 * ru_tc(isb) + bsn_prm%tb_adj  !baseflow time, hr
        if (tb > 48.) tb = 48.			   !maximum 48hrs
        tp = .375 * tb                       ! time to peak flow
	  !! convert to time step (from hr), J.Jeong March 2009
	  tb = ceiling(tb * 60./ real(time%dtm))
	  tp = int(tp * 60./ real(time%dtm))         
	  
	  if(tp==0) tp = 1
	  if(tb==tp) tb = tb + 1
	  itsb(isb) = int(tb) 
        
	  ! Triangular Unit Hydrograph
	  if (bsn_cc%uhyd == 0) then
	    do i = 1, itsb(isb)
            xi = float(i)
 	      if (xi < tp) then           !! rising limb of hydrograph
              q = xi / tp
            else                        !! falling limb of hydrograph
              q = (tb - xi) / (tb - tp)
            end if
            q = Max(0.,q)
            uhs(isb,i) = (q + ql) / 2.
            ql = q
            sumq = sumq + uhs(isb,i)
          end do
          
		do i = 1, itsb(isb)
            uhs(isb,i) = uhs(isb,i) / sumq
          end do
	  
	  ! Gamma Function Unit Hydrograph
	  elseif (bsn_cc%uhyd == 2) then
          i = 1; q=1.
		do while (q>0.0001)
            xi = float(i)
            q = (xi / tp) ** bsn_prm%uhalpha * exp((1.- xi / tp) *      &     
                             uhalpha)
            q = Max(0.,q)
            uhs(isb,i) = (q + ql) / 2.
            ql = q
            sumq = sumq + uhs(isb,i)
	      i = i + 1
	      if (i>3.*time%step) exit
	    end do
	    itsb(isb) = i - 1
          do i = 1, itsb(isb)
            uhs(isb,i) = uhs(isb,i) / sumq
          end do
	  endif 
      end do
      end if

      return
      end subroutine ru_allo