	subroutine swr_origtile(wt_above_tile)

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine computes tile drainage using basic tile equations 

      use tiles_data_module
      use hru_module, only : hru, ihru, qtile, sw_excess, wt_shall
      use soil_module
      
      implicit none

      integer :: j                          !none       |HRU number
      integer :: isdr                       !none       |pointer to tile drainage data
      real, intent (in) :: wt_above_tile    !mm         |height of water table above tiles

      j = ihru
      isdr = hru(j)%tiledrain

      !! compute tile flow using the original tile equations

      if (soil(j)%sw > soil(j)%sumfc) then
        sw_excess = (wt_above_tile / wt_shall) * (soil(j)%sw - soil(j)%sumfc)
        qtile = sw_excess * (1. - Exp(-24. / sdr(isdr)%time))
        qtile = amin1(qtile, sdr(isdr)%drain_co)
      else
        qtile = 0.
      end if
     
	return
	end subroutine swr_origtile