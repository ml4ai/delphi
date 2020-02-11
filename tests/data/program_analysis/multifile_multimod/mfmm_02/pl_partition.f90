      subroutine pl_partition
      
      use plant_data_module
      use basin_module
      use hru_module, only : hru, uapd, uno3d, lai_yrmx, par, bioday, ep_day, es_day,              &
         ihru, ipl, pet_day, rto_no3, rto_solp, sum_no3, sum_solp,uapd_tot, uno3d_tot, vpd
      use plant_module
      use carbon_module
      use organic_mineral_mass_module
      
      implicit none 
      
      integer :: j              !none               |HRU number
      integer :: idp            !                   |
      real :: root_frac         !none               |root mass fraction
      real :: ab_gr_frac        !none               |above ground mass fraction
      real :: leaf_mass_frac    !none               |leaf mass fraction of above ground biomass
      real :: stem_mass_frac    !none               |stem mass fraction of above ground biomass
      real :: seed_mass_frac    !none               |stem mass fraction of above ground biomass
      real :: n_left            !none               |n left after seed is removed
      real :: n_frac            !none               |n fraction in remainder of plant
      real :: p_left            !none               |p left after seed is removed
      real :: p_frac            !none               |p fraction in remainder of plant
      real :: m_left            !none               |mass left after seed is removed
      real :: leaf_frac_veg     !none               |fraction veg mass (stem+leaf) that is leaf
      real :: leaf_mass_frac_veg     !none               |fraction veg mass (stem+leaf) that is leaf
           
      j = ihru
      idp = pcom(j)%plcur(ipl)%idplt
      
      !! update plant mass for daily biomass/c increase and n and p uptake
      pl_mass(j)%tot(ipl) = pl_mass(j)%tot(ipl) + pl_mass_up
      
      !! partition root and above ground biomass
      pl_mass(j)%tot(ipl)%m = Max(pl_mass(j)%tot(ipl)%m, 0.)
      root_frac = pcom(j)%plg(ipl)%root_frac
      ab_gr_frac = 1. - root_frac
      pl_mass(j)%ab_gr(ipl)%m = ab_gr_frac * pl_mass(j)%tot(ipl)%m
      pl_mass(j)%root(ipl)%m = root_frac * pl_mass(j)%tot(ipl)%m

      !! partition leaf and stem (stalk) and seed (grain) mass
      if (pldb(idp)%typ == "perennial") then
        leaf_frac_veg = 0.03    !forest
      else
        leaf_frac_veg = 0.30    !should be plant parm
      end if
      seed_mass_frac = pcom(j)%plg(ipl)%hi_adj
      leaf_mass_frac_veg = leaf_frac_veg * pcom(j)%plg(ipl)%lai / pldb(idp)%blai
      leaf_mass_frac = leaf_mass_frac_veg * (1. - seed_mass_frac)
      stem_mass_frac = (1. - leaf_mass_frac_veg) * (1. - seed_mass_frac)
      
      pl_mass(j)%leaf(ipl)%m = leaf_mass_frac * pl_mass(j)%ab_gr(ipl)%m
      pl_mass(j)%seed(ipl)%m = seed_mass_frac * pl_mass(j)%ab_gr(ipl)%m
      pl_mass(j)%stem(ipl)%m = stem_mass_frac * pl_mass(j)%ab_gr(ipl)%m
          
      !! partition carbon with constant fractions
      pl_mass(j)%leaf(ipl)%c = c_frac%leaf * pl_mass(j)%leaf(ipl)%m
      pl_mass(j)%stem(ipl)%c = c_frac%stem * pl_mass(j)%stem(ipl)%m
      pl_mass(j)%seed(ipl)%c = c_frac%seed * pl_mass(j)%seed(ipl)%m
      pl_mass(j)%root(ipl)%c = c_frac%root * pl_mass(j)%root(ipl)%m
      pl_mass(j)%ab_gr(ipl)%c = pl_mass(j)%leaf(ipl)%c + pl_mass(j)%stem(ipl)%c + pl_mass(j)%seed(ipl)%c
      pl_mass(j)%tot(ipl)%c = pl_mass(j)%ab_gr(ipl)%c + pl_mass(j)%root(ipl)%c
          
      !! partition n and p
      if (pldb(idp)%typ == "perennial") then
        !! partition leaves and seed (stem is woody biomass)
        pl_mass(j)%seed(ipl)%n = pldb(idp)%cnyld * pl_mass(j)%seed(ipl)%m
        n_left = pl_mass(j)%tot(ipl)%n - pl_mass(j)%seed(ipl)%n
        m_left = pl_mass(j)%leaf(ipl)%m + pl_mass(j)%stem(ipl)%m + pl_mass(j)%root(ipl)%m
        if (m_left > 1.e-9) then
          !! partition n_left between remaining masses - assume equal concentrations
          pl_mass(j)%leaf(ipl)%n = n_left * pl_mass(j)%leaf(ipl)%m / m_left
          pl_mass(j)%stem(ipl)%n = n_left * pl_mass(j)%stem(ipl)%m / m_left
          pl_mass(j)%root(ipl)%n = n_left * pl_mass(j)%root(ipl)%m / m_left
          pl_mass(j)%ab_gr(ipl)%n = pl_mass(j)%seed(ipl)%n + pl_mass(j)%leaf(ipl)%n + pl_mass(j)%stem(ipl)%n
        
          pl_mass(j)%seed(ipl)%p = pldb(idp)%cpyld * pl_mass(j)%seed(ipl)%m
          p_left = pl_mass(j)%tot(ipl)%p - pl_mass(j)%seed(ipl)%p
          !! partition p_left between remaining masses - assume equal concentrations
          pl_mass(j)%leaf(ipl)%p = p_left * pl_mass(j)%leaf(ipl)%m / m_left
          pl_mass(j)%stem(ipl)%p = p_left * pl_mass(j)%stem(ipl)%m / m_left
          pl_mass(j)%root(ipl)%p = p_left * pl_mass(j)%root(ipl)%m / m_left
          pl_mass(j)%ab_gr(ipl)%p = pl_mass(j)%seed(ipl)%p + pl_mass(j)%leaf(ipl)%p + pl_mass(j)%stem(ipl)%p
        end if
      else
        !! annual or grass (stem is stalk) - partition seed (grain)
        pl_mass(j)%seed(ipl)%n = pldb(idp)%cnyld * pl_mass(j)%seed(ipl)%m
        !! assume same concentration in rest of plant
        n_left = pl_mass(j)%tot(ipl)%n - pl_mass(j)%seed(ipl)%n
        if (n_left < 0.) then
          pl_mass(j)%seed(ipl)%n = 0.9 * pl_mass(j)%tot(ipl)%n
          n_left = 0.1 * pl_mass(j)%tot(ipl)%n
        end if
        if (pl_mass(j)%tot(ipl)%m - pl_mass(j)%seed(ipl)%m > 1.e-6) then
          n_frac = n_left / (pl_mass(j)%tot(ipl)%m - pl_mass(j)%seed(ipl)%m)
        else
          n_frac = 0.
        end if
        pl_mass(j)%leaf(ipl)%n = n_frac * pl_mass(j)%leaf(ipl)%m
        pl_mass(j)%stem(ipl)%n = n_frac * pl_mass(j)%stem(ipl)%m
        pl_mass(j)%root(ipl)%n = n_frac * pl_mass(j)%root(ipl)%m
        pl_mass(j)%ab_gr(ipl)%n = pl_mass(j)%seed(ipl)%n + pl_mass(j)%leaf(ipl)%n + pl_mass(j)%stem(ipl)%n

        pl_mass(j)%seed(ipl)%p = pldb(idp)%cpyld * pl_mass(j)%seed(ipl)%m
        !! assume same concentration in rest of plant
        p_left = pl_mass(j)%tot(ipl)%p - pl_mass(j)%seed(ipl)%p
        if (p_left < 0.) then
          pl_mass(j)%seed(ipl)%p = 0.9 * pl_mass(j)%tot(ipl)%p
          p_left = 0.1 * pl_mass(j)%tot(ipl)%p
        end if
        if (pl_mass(j)%tot(ipl)%m - pl_mass(j)%seed(ipl)%m > 1.e-6) then
          p_frac = p_left / (pl_mass(j)%tot(ipl)%m - pl_mass(j)%seed(ipl)%m)
        else
          p_frac = 0.
        end if
        pl_mass(j)%leaf(ipl)%p = p_frac * pl_mass(j)%leaf(ipl)%m
        pl_mass(j)%stem(ipl)%p = p_frac * pl_mass(j)%stem(ipl)%m
        pl_mass(j)%root(ipl)%p = p_frac * pl_mass(j)%root(ipl)%m
        pl_mass(j)%ab_gr(ipl)%p = pl_mass(j)%seed(ipl)%p + pl_mass(j)%leaf(ipl)%p + pl_mass(j)%stem(ipl)%p
      end if
           
      return
      end subroutine pl_partition