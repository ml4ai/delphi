     subroutine soil_nutcarb_init (isol)

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine initializes soil chemical properties

      use soil_module
      use basin_module
      use organic_mineral_mass_module

      integer :: nly, j
      real :: actp, solp, ssp, zdst

      nly = sol(isol)%s%nly

      !calculate percent carbon for lower layers if only have upper layer
      do j = 2, nly
        if (sol1(isol)%cbn(j) == 0.) then
          sol1(isol)%cbn(j) = sol1(isol)%cbn(j-1) * Exp(-.001 * sol(isol)%phys(j)%d)
        end if
      end do

      !calculate initial nutrient contents of layers, profile and
      !average in soil for the entire watershed

      do j = 1, nly
        sol(isol)%phys(j)%conv_wt = sol(isol)%phys(j)%bd * sol(isol)%phys(j)%thick / 100.    ! mg/kg => kg/ha
        wt1 = sol(isol)%phys(j)%conv_wt
        
        !! set initial mineral pools
        !set initial no3 pools
        if (sol1(isol)%mn(j)%no3 <= 0.) then
          zdst = Exp(-sol(isol)%phys(j)%d / 1000.)
          sol1(isol)%mn(j)%no3 = 10. * zdst * .7
        end if
        sol1(isol)%mn(j)%no3 =  sol1(isol)%mn(j)%no3 * wt1      !! mg/kg => kg/ha

        !set initial labile P pool
        if (sol1(isol)%mp(j)%lab > 0.0001) then
          sol1(isol)%mp(j)%lab = sol1(isol)%mp(j)%lab * wt1   !! mg/kg => kg/ha
        else
          !! assume initial concentration of 5 mg/kg
          sol1(isol)%mp(j)%lab = 5. * wt1
        end if

        !! set active P pool based on dynamic PSP MJW
	    if (bsn_cc%sol_P_model == 0) then 
	      !! Allow Dynamic PSP Ratio
            !! convert to concentration
            solp = sol1(isol)%mp(j)%lab / sol(isol)%phys(j)%conv_wt * 1000000.
	      !! PSP = -0.045*log (% clay) + 0.001*(Solution P, mg kg-1) - 0.035*(% Organic C) + 0.43
	      if (sol(isol)%phys(j)%clay > 0.) then
              bsn_prm%psp = -0.045 * log(sol(isol)%phys(j)%clay) + (0.001 * solp) 
              bsn_prm%psp = bsn_prm%psp - (0.035 * sol1(isol)%tot(j)%c) + 0.43 
            else
              bsn_prm%psp = 0.4
            endif   		
            !! Limit PSP range
            if (bsn_prm%psp < .05) then
              bsn_prm%psp = 0.05
	      else if (bsn_prm%psp > 0.9) then
              bsn_prm%psp = 0.9
            end if
        end if
        sol1(isol)%mp(j)%act = sol1(isol)%mp(j)%lab * (1. - bsn_prm%psp) / bsn_prm%psp

        !! Set Stable pool based on dynamic coefficient
	    if (bsn_cc%sol_P_model == 0) then  !! From White et al 2009 
            !! convert to concentration for ssp calculation
	        actp = sol1(isol)%mp(j)%act / sol(isol)%phys(j)%conv_wt * 1000000.
		    solp = sol1(isol)%mp(j)%lab / sol(isol)%phys(j)%conv_wt * 1000000.
            !! estimate Total Mineral P in this soil based on data from sharpley 2004
		    ssp = 25.044 * (actp + solp)** -0.3833
		    !!limit SSP Range
		    if (ssp > 7.) ssp = 7.
		    if (ssp < 1.) ssp = 1.	      	  
		    sol1(isol)%mp(j)%sta = ssp * (sol1(isol)%mp(j)%act + sol1(isol)%mp(j)%lab)
         else
	      !! the original code
		  sol1(isol)%mp(j)%sta = 4. * sol1(isol)%mp(j)%act
	   end if
      end do

      
      !! set initial organic pools - originally by Zhang
	  do j = 1, nly

        !set soil humus fractions from DSSAT
        frac_hum_microb = 0.02
        frac_hum_slow = 0.54
        frac_hum_passive = 0.44

        !initialize total soil organic pool - no litter
        !kg/ha = mm * t/m3 * m/1,000 mm * 1,000 kg/t * 10,000 m2/ha
        sol1(isol)%tot(j)%m = 10000. * sol(isol)%phys(j)%thick * sol(isol)%phys(j)%bd
        sol1(isol)%tot(j)%c = sol1(isol)%tot(j)%m * sol1(isol)%cbn(j) / 100.
        sol1(isol)%tot(j)%n = sol1(isol)%tot(j)%c / 10.     !assume 10:1 C:N ratio
        sol1(isol)%tot(j)%p = sol1(isol)%tot(j)%c / 100.    !assume 100:1 C:P ratio
        
        !initialize passive humus pool
        sol1(isol)%hp(j)%m = frac_hum_passive * sol1(isol)%tot(j)%m
        sol1(isol)%hp(j)%c = .42 * frac_hum_passive * sol1(isol)%tot(j)%c   !assume 42% C
        sol1(isol)%hp(j)%n = sol1(isol)%hp(j)%c / 10.                       !assume 10:1 C:N ratio
        sol1(isol)%hp(j)%p = sol1(isol)%hp(j)%c / 80.                       !assume 80:1 C:P ratio
            
        !initialize stable humus pool
        sol1(isol)%hs(j)%m = frac_hum_slow * sol1(isol)%tot(j)%m
        sol1(isol)%hs(j)%c = .42 * frac_hum_slow * sol1(isol)%tot(j)%c  !assume 42% C
        sol1(isol)%hs(j)%n = sol1(isol)%hs(j)%c / 10.                   !assume 10:1 C:N ratio
        sol1(isol)%hs(j)%p = sol1(isol)%hs(j)%c / 80.                   !assume 80:1 C:P ratio
            
        !initialize microbial pool
        sol1(isol)%microb(j)%m = frac_hum_microb * sol1(isol)%tot(j)%m
        sol1(isol)%microb(j)%c = .42 * frac_hum_microb * sol1(isol)%tot(j)%c    !assume 42% C
        sol1(isol)%microb(j)%n = sol1(isol)%microb(j)%c / 8.                    !assume 8:1 C:N ratio
        sol1(isol)%microb(j)%p = sol1(isol)%microb(j)%c / 80.                   !assume 80:1 C:P ratio

        !initialize metabolic litter pool
        sol1(isol)%meta(j)%m = sol1(isol)%tot(j)%m / 1000.      !t/ha - kg/ha
        sol1(isol)%meta(j)%c = .42 * sol1(isol)%meta(j)%c       !assume 42% C
        sol1(isol)%meta(j)%n = sol1(isol)%meta(j)%c / 150.      !assume 150:1 C:N ratio
        sol1(isol)%meta(j)%p = sol1(isol)%meta(j)%c / 1500.     !assume 1500:1 C:P ratio
            
        !initialize structural litter pool
        sol1(isol)%str(j)%m = sol1(isol)%meta(j)%m
        sol1(isol)%str(j)%c = .42 * sol1(isol)%str(j)%c         !assume 42% C
        sol1(isol)%str(j)%n = sol1(isol)%lig(j)%c / 150.        !assume 150:1 C:N ratio
        sol1(isol)%str(j)%p = sol1(isol)%lig(j)%c / 1500.       !assume 1500:1 C:P ratio
            
        !initialize lignon litter pool
        sol1(isol)%lig(j)%m = .8 * sol1(isol)%str(j)%m
        sol1(isol)%lig(j)%c = .8 * sol1(isol)%str(j)%c
        sol1(isol)%lig(j)%n = .2 * sol1(isol)%str(j)%c
        sol1(isol)%lig(j)%p = .02 * sol1(isol)%str(j)%c
                        
        !initialize water soluble pool
        !sol1(isol)%water(j)%m = 
        !sol1(isol)%water(j)%c = 
        !sol1(isol)%water(j)%n = 
        !sol1(isol)%water(j)%p = 

	  end do	

      return
      end subroutine soil_nutcarb_init