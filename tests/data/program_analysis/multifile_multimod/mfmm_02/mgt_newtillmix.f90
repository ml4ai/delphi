      subroutine mgt_newtillmix (jj, bmix, idtill)

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine mixes residue and nutrients during tillage and 
!!    biological mixing
!!    New version developed by Armen R. Kemanian in collaboration with Stefan Julich and Cole Rossi
!!    Mixing was extended to all layers
!!    A subroutine to simulate stimulation of organic matter decomposition was added
!!    March 2009: testing has been minimal and further adjustments are expected
!!    use with caution and report anomalous results to akemanian@brc.tamus.edu and jeff.arnold@ars.usda.edu

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name          |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    npmx          |none          |number of different pesticides used in the simulation
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ LOCAL DEFINITIONS ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    bmix        |none          |biological mixing efficiency: this 
!!                               |number is zero for tillage operations
!!    dg          |mm            |depth of soil layer
!!    nl          |none          |number of layers being mixed
!!    thtill(:)   |none          |fraction of soil layer that is mixed
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Min, Max
!!    SWAT: curno 

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use tillage_data_module
      use basin_module
      use organic_mineral_mass_module
      use hru_module, only: tillage_days, tillage_depth, tillage_switch
      use soil_module
      use constituent_mass_module
      
      implicit none

      integer, intent (in) :: jj       !none           |HRU number
      integer, intent (in) :: idtill   !none           |tillage type
      real, intent (in) :: bmix        !               | 
      integer :: l                     !none           |counter
      integer :: k                     !none           |counter
      integer :: kk                    !               |
      integer :: npmx                  !               |
      !CB 12/2/09 nl and a are not used.
      real :: emix                     !none           |mixing efficiency
      real :: dtil                     !mm             |depth of mixing
      real :: XX                       !varies         |variable to hold calculation results
      real :: WW1                      !               |
      real :: WW2                      !               |
      real :: WW3                      !               |
      real :: WW4                      !               |
      real :: maxmix                   !none           | maximum mixing eff to preserve specified minimum residue cover
      !!by zhang
      !!=============   
      real :: smix(22+cs_db%num_pests+12)         !varies         |amount of substance in soil profile
                                       !               |that is being redistributed between mixed layers
      !CB 12/2/09 thtill is not used. mjw rev 490
      !!changed the dimension from 22 + npmx to 22 + npmx + 12
      !!by zhang
      !!=============
      real :: sol_mass(soil(jj)%nly)    !              | 
      real :: sol_thick(soil(jj)%nly)   !              |
      real :: sol_msm(soil(jj)%nly)     !              |sol_mass mixed
      real :: sol_msn(soil(jj)%nly)     !              |sol_mass not mixed 

      npmx = cs_db%num_pests

      XX = 0.
      WW1 = 0.
      WW2 = 0.
      WW3 = 0.
      WW4 = 0.
      emix = 0.
      dtil = 0.
      if (bmix > 1.e-6) then
        !! biological mixing
        emix = bmix !bmix MJW (rev 412)
        kk = soil(jj)%nly
        dtil = Min(soil(jj)%phys(kk)%d, 50.) ! it was 300.  MJW (rev 412)
      else 
        !! tillage operation
        emix = tilldb(idtill)%effmix
        dtil = tilldb(idtill)%deptil
      end if

      !!by zhang DSSAT tillage
      !!=======================
      if (bsn_cc%cswat == 2) then
          tillage_days(jj) = 0
          tillage_depth(jj) = dtil
          tillage_switch(jj) = 1
      end if
      !!by zhang DSSAT tillage
      !!=======================


      smix = 0.
      sol_mass = 0.
      sol_thick = 0.
      sol_msm = 0.
      sol_msn = 0.

	!! incorporate pathogens - no mixing - lost from transport
      if (dtil > 10.) then     
        !! incorporate pathogens
	  end if

      do l = 1, soil(jj)%nly
        if ( l == 1) then
          sol_thick(l) = soil(jj)%phys(l)%d
        else	
          sol_thick(l) = soil(jj)%phys(l)%d - soil(jj)%phys(l-1)%d
        end if
	
      sol_mass(l) = (sol_thick(l) / 1000.) * 10000. *                    &
          soil(jj)%phys(1)%bd * 1000. * (1.- soil(jj)%phys(l)%rock/ 100.)

      end do

      smix = 0.

      if (dtil > 0.) then
    ! added by Armen 09/10/2010 next line only
    if (dtil < 10.0) dtil = 11.0
	 do l = 1, soil(jj)%nly

          if (soil(jj)%phys(l)%d <= dtil) then
            !! msm = mass of soil mixed for the layer
            !! msn = mass of soil not mixed for the layer		
            sol_msm(l) = emix * sol_mass(l)	
            sol_msn(l) = sol_mass(l) - sol_msm(l)	
          else if (soil(jj)%phys(l)%d > dtil .AND. soil(jj)%phys(l-1)%d < dtil) then 
            sol_msm(l) = emix * sol_mass(l) * (dtil - soil(jj)%phys(l-1)%d) / sol_thick(l)
            sol_msn(l) =  sol_mass(l) -  sol_msm(l)
          else
            sol_msm(l) = 0.
            sol_msn(l) = sol_mass(l)
          end if

          !! calculate the mass or concentration of each mixed element 
          !! mass based mixing
          WW1 = sol_msm(l)/(sol_msm(l) + sol_msn(l))
          smix(1) = smix(1) + soil1(jj)%mn(l)%no3 * WW1
          smix(2) = smix(2) + soil1(jj)%hp(l)%n * WW1
          smix(3) = smix(3) + soil1(jj)%mn(l)%nh4 * WW1
          smix(4) = smix(4) + soil1(jj)%mp(l)%lab * WW1
          smix(5) = smix(5) + soil1(jj)%hp(l)%p * WW1
          smix(6) = smix(6) + soil1(jj)%hs(l)%n * WW1
          smix(7) = smix(7) + soil1(jj)%mp(l)%act * WW1
          smix(8) = smix(8) + soil1(jj)%tot(l)%n * WW1
          smix(9) = smix(9) + soil1(jj)%tot(l)%p * WW1
          smix(10) = smix(10) + soil1(jj)%mp(l)%sta * WW1
          smix(11) = smix(11) + soil1(jj)%tot(l)%m * WW1
          smix(12) = smix(12) + soil1(jj)%man(l)%c * WW1
          smix(13) = smix(13) + soil1(jj)%man(l)%n * WW1
          smix(14) = smix(14) + soil1(jj)%man(l)%p * WW1

		!! concentration based mixing
          WW2 = XX + sol_msm(l)
          smix(15) = (XX * smix(15) + soil1(jj)%tot(l)%c * sol_msm(l))/WW2
          smix(16) = (XX * smix(16) + soil1(jj)%tot(l)%n * sol_msm(l))/WW2
          smix(17) = (XX * smix(17) + soil(jj)%phys(l)%clay * sol_msm(l))/WW2
          smix(18) = (XX * smix(18) + soil(jj)%phys(l)%silt * sol_msm(l))/WW2
          smix(19)=(XX*smix(19)+soil(jj)%phys(l)%sand*sol_msm(l))/WW2

            !!by zhang
            !!============== 
            if (bsn_cc%cswat == 2) then         
	        smix(20+npmx+1) = smix(20+npmx+1) + soil1(jj)%str(l)%c * WW1
	        smix(20+npmx+2) = smix(20+npmx+2) + soil1(jj)%lig(l)%c * WW1
	        smix(20+npmx+3) = smix(20+npmx+3) + soil1(jj)%lig(l)%n* WW1
	        smix(20+npmx+4) = smix(20+npmx+4) + soil1(jj)%meta(l)%c * WW1
	        smix(20+npmx+5) = smix(20+npmx+5) + soil1(jj)%meta(l)%m * WW1
	        smix(20+npmx+6) = smix(20+npmx+6) + soil1(jj)%lig(l)%m * WW1
	        smix(20+npmx+7) = smix(20+npmx+7) + soil1(jj)%str(l)%m * WW1  
	        
	        smix(20+npmx+8) = smix(20+npmx+8) + soil1(jj)%str(l)%n * WW1
	        smix(20+npmx+9) = smix(20+npmx+9) + soil1(jj)%meta(l)%n * WW1
	        smix(20+npmx+10) = smix(20+npmx+10) +soil1(jj)%microb(l)%n* WW1
	        smix(20+npmx+11) = smix(20+npmx+11) + soil1(jj)%hs(l)%n * WW1
	        smix(20+npmx+12) = smix(20+npmx+12) + soil1(jj)%hp(l)%n * WW1  
	      end if
            !!by zhang 	
            !!=============
			
          XX = XX + sol_msm(l)

        end do

          do l = 1, soil(jj)%nly
			
            ! reconstitute each soil layer 
            WW3 = sol_msn(l) / sol_mass(l)
            WW4 = sol_msm(l) / XX

            soil1(jj)%mn(l)%no3 = soil1(jj)%mn(l)%no3 * WW3 + smix(1) * WW4
            soil1(jj)%hp(l)%n = soil1(jj)%hp(l)%n * WW3 + smix(2) * WW4
            soil1(jj)%mn(l)%nh4 = soil1(jj)%mn(l)%nh4 * WW3 + smix(3) * WW4
            soil1(jj)%mp(l)%lab = soil1(jj)%mp(l)%lab * WW3+smix(4) * WW4
            soil1(jj)%hp(l)%p = soil1(jj)%hp(l)%p * WW3 + smix(5) * WW4
            soil1(jj)%hs(l)%n = soil1(jj)%hs(l)%n * WW3+smix(6) * WW4
            soil1(jj)%mp(l)%act = soil1(jj)%mp(l)%act * WW3+smix(7) * WW4
            soil1(jj)%tot(l)%n = soil1(jj)%tot(l)%n * WW3 + smix(8) * WW4
            soil1(jj)%tot(l)%p = soil1(jj)%tot(l)%p * WW3 + smix(9)*WW4
            soil1(jj)%mp(l)%sta = soil1(jj)%mp(l)%sta * WW3+smix(10) * WW4
            soil1(jj)%tot(l)%m = soil1(jj)%tot(l)%m * WW3 + smix(11)*WW4
            if (soil1(jj)%tot(l)%m < 1.e-10) soil1(jj)%tot(l)%m = 1.e-10
            soil1(jj)%man(l)%c = soil1(jj)%man(l)%c * WW3 + smix(12)*WW4
            soil1(jj)%man(l)%n = soil1(jj)%man(l)%n * WW3 + smix(13)*WW4
            soil1(jj)%man(l)%p = soil1(jj)%man(l)%p * WW3 + smix(14)*WW4

            soil1(jj)%tot(l)%c = (soil1(jj)%tot(l)%c * sol_msn(l)+smix(15)    &
                 * sol_msm(l)) / sol_mass(l)
            soil1(jj)%tot(l)%n  = (soil1(jj)%tot(l)%n * sol_msn(l)+smix(16)     &
                 * sol_msm(l)) / sol_mass(l)
            soil(jj)%phys(l)%clay = (soil(jj)%phys(l)%clay                  &
                 * sol_msn(l)+smix(17) * sol_msm(l)) / sol_mass(l)
            soil(jj)%phys(l)%silt = (soil(jj)%phys(l)%silt                  &
                 * sol_msn(l)+smix(18) * sol_msm(l)) / sol_mass(l)
            soil(jj)%phys(l)%sand = (soil(jj)%phys(l)%sand                  &
                 * sol_msn(l) + smix(19) * sol_msm(l)) / sol_mass(l)

            do k = 1, npmx
              cs_soil(jj)%ly(l)%pest(k) = cs_soil(jj)%ly(l)%pest(k) * WW3 + smix(20+k) * WW4
            end do

             if (bsn_cc%cswat == 2) then
      soil1(jj)%str(l)%c = soil1(jj)%str(l)%c * WW3+smix(20+npmx+1) * WW4
      soil1(jj)%lig(l)%c = soil1(jj)%lig(l)%c * WW3 + smix(20+npmx+2) * WW4
      soil1(jj)%lig(l)%n = soil1(jj)%lig(l)%n * WW3 + smix(20+npmx+3) * WW4
      soil1(jj)%meta(l)%c = soil1(jj)%meta(l)%c * WW3 + smix(20+npmx+4) * WW4
      soil1(jj)%meta(l)%m = soil1(jj)%meta(l)%m * WW3 + smix(20+npmx+5) * WW4
      soil1(jj)%lig(l)%m = soil1(jj)%lig(l)%m * WW3 + smix(20+npmx+6) * WW4
       soil1(jj)%str(l)%m = soil1(jj)%str(l)%m * WW3 + smix(20+npmx+7)* WW4
       soil1(jj)%str(l)%n = soil1(jj)%str(l)%n * WW3 + smix(20+npmx+8) * WW4
       soil1(jj)%meta(l)%n = soil1(jj)%meta(l)%n * WW3 + smix(20+npmx+9) * WW4
       soil1(jj)%microb(l)%n = soil1(jj)%microb(l)%n * WW3 + smix(20 + npmx + 10) * WW4
       soil1(jj)%hs(l)%n = soil1(jj)%hs(l)%n * WW3 + smix(20 + npmx + 11) * WW4
       soil1(jj)%hp(l)%n = soil1(jj)%hp(l)%n * WW3 + smix(20 + npmx+12) * WW4
             end if
            !!by zhang 
            !!==============

	  end do
	
        if (bsn_cc%cswat == 1) then
            call mgt_tillfactor(jj,bmix,emix,dtil,sol_thick)
        end if
      end if

      return
      end subroutine mgt_newtillmix