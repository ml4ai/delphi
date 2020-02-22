      subroutine mgt_killop (jj, iplant)

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine performs the kill operation

      use basin_module
      use organic_mineral_mass_module
      use hru_module, only : hru, ihru, ipl
      use soil_module
      use plant_module
      use constituent_mass_module
      
      implicit none
   
      integer :: j                     !none           |HRU number
      integer :: k                     !none           |counter
      integer, intent (in) :: jj       !none           |counter
      integer, intent (in) :: iplant   !               |plant number xwalked from hlt_db()%plant and plants.plt
      integer :: ly                    !none           |soil layer

      j = jj
      ipl = iplant

      !! update root fractions in each layer
      call pl_rootfr
      
      !! allocate dead roots, N, P to soil layers
	  do ly = 1, soil(j)%nly
	      soil1(j)%tot(ly) = soil(j)%ly(ly)%rtfr * pl_mass(j)%root_com + soil1(j)%tot(ly)
      end do
      
      !! add above ground mass to residue pool
      rsd1(j)%tot(1) = pl_mass(j)%ab_gr(ipl) + rsd1(j)%tot(1)

      !! zero all plant mass
      pl_mass(j)%tot(ipl) = plt_mass_z
      pl_mass(j)%ab_gr(ipl) = plt_mass_z
      pl_mass(j)%leaf(ipl) = plt_mass_z
      pl_mass(j)%stem(ipl) = plt_mass_z
      pl_mass(j)%seed(ipl) = plt_mass_z
      pl_mass(j)%root(ipl) = plt_mass_z
      
      do k = 1, cs_db%num_pests
        cs_soil(j)%ly(1)%pest(k) = cs_soil(j)%ly(1)%pest(k) !+ cs_pl(j)%pest(k)
        cs_pl(j)%pest(k) = 0.
      end do

	  !! reset plant variables
      pcom(j)%plg(ipl) = plgz
      pcom(j)%plm(ipl) = plmz
      pcom(j)%plstr(ipl) = plstrz
      !! can't reset entire plcur - harv_num can't be zero'd
      pcom(j)%plcur(ipl)%gro = "n"
      pcom(j)%plcur(ipl)%idorm = "n"
      pcom(j)%plcur(ipl)%phuacc = 0.
      pcom(j)%plcur(ipl)%curyr_mat = 1

      return
      end subroutine mgt_killop