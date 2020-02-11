      subroutine allocate_parms
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine allocates array sizes

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    mhyd        |none          |max number of hydrographs
!!    nstep       |none          |max number of time steps per day
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      use hru_module      
      use time_module
      use hydrograph_module
      use constituent_mass_module
      
!! initialize variables    
      mhyd = 1  !!added for jaehak vars
      mhru = sp_ob%hru
      mch = sp_ob%chan

!!    drains
      allocate (wnan(10))
      allocate (ranrns_hru(mhru))
      
      !dimension plant arrays used each day and not saved
       mpc = 10
       allocate (uno3d(mpc))
       allocate (uapd(mpc))
       allocate (un2(mpc))
       allocate (up2(mpc))
       allocate (translt(mpc))
       allocate (par(mpc))
       allocate (htfac(mpc))
       allocate (epmax(mpc))
       epmax = 0.

!!    arrays for plant communities
      allocate (cvm_com(mhru))
      allocate (blai_com(mhru))
      allocate (rsdco_plcom(mhru))
      allocate (percn(mhru))

!! septic changes added 1/28/09 gsm
      allocate (percp(mhru))
      allocate (i_sep(mhru))
      allocate (sep_tsincefail(mhru))
      allocate (qstemm(mhru))
      allocate (bio_bod(mhru))
      allocate (biom(mhru))
      allocate (rbiom(mhru))
      allocate (fcoli(mhru))
      allocate (bz_perc(mhru))
      allocate (plqm(mhru))
      allocate (itb(mhru))
      
      if (time%step > 0) allocate (hhqday(mhru,time%step))
      allocate (uh(mhru,time%step+1))

 !!  added per JGA for Srini by gsm 9/8/2011
 !! arrays for mangement output (output.mgt)  
      allocate (sol_sumno3(mhru))
      allocate (sol_sumsolp(mhru))

      allocate (iseptic(mhru))

!!    arrays which contain data related to years of rotation,
!!    grazings per year, and HRUs
      allocate (grz_days(mhru))

!!    arrays which contain data related to HRUs
      allocate (brt(mhru))
      allocate (canstor(mhru))
      allocate (cbodu(mhru))
      allocate (chl_a(mhru))
      allocate (cklsp(mhru))
      allocate (cn2(mhru))
      allocate (cnday(mhru))
!    Drainmod tile equations  01/2006 
	  allocate (cumei(mhru))
	  allocate (cumeira(mhru))
	  allocate (cumrt(mhru))
	  allocate (cumrai(mhru))
!    Drainmod tile equations  01/2006
      allocate (dormhr(mhru))
      allocate (doxq(mhru))
      allocate (filterw(mhru))
      allocate (hru_ra(mhru))
      allocate (hru_rmx(mhru))
      allocate (igrz(mhru))
      allocate (yr_skip(mhru))
      allocate (isweep(mhru))
      allocate (phusw(mhru))
      allocate (lai_yrmx(mhru))
      allocate (latno3(mhru))
      allocate (latq(mhru))
      allocate (ndeat(mhru))
      allocate (nplnt(mhru))
      allocate (orgn_con(mhru))
      allocate (orgp_con(mhru))
      allocate (ovrlnd(mhru))
      allocate (phubase(mhru))

      allocate (pplnt(mhru))
      allocate (qdr(mhru))
      allocate (rhd(mhru))

!    Drainmod tile equations  01/2006 
	  allocate (sstmaxd(mhru))	  
!    Drainmod tile equations  01/2006 
      allocate (sedminpa(mhru))
      allocate (sedminps(mhru))
      allocate (sedorgn(mhru))
      allocate (sedorgp(mhru))
      allocate (sedyld(mhru))

      allocate (sanyld(mhru))
      allocate (silyld(mhru))
      allocate (clayld(mhru))
      allocate (sagyld(mhru))
      allocate (lagyld(mhru))
      allocate (grayld(mhru))
      allocate (sed_con(mhru))
      allocate (sepbtm(mhru))
      allocate (smx(mhru))
      allocate (snotmp(mhru))
      allocate (soln_con(mhru))
      allocate (solp_con(mhru))
!!    Drainmod tile equations  01/2006 
	  allocate (stmaxd(mhru))
      allocate (itill(mhru))
      allocate (surfq(mhru))
      allocate (surqno3(mhru))
      allocate (surqsolp(mhru))
      allocate (swtrg(mhru))
      allocate (t_ov(mhru))
      allocate (tconc(mhru))
      allocate (tc_gwat(mhru))
      allocate (tileno3(mhru))
      allocate (tmn(mhru))
      allocate (tmpav(mhru))
      allocate (tmx(mhru))
      allocate (twash(mhru))
      allocate (u10(mhru))
      allocate (usle_cfac(mhru))
      allocate (usle_eifac(mhru))
      allocate (wfsh(mhru))

      allocate (bss(4,mhru))
      allocate (wrt(2,mhru))
      allocate (surf_bs(17,mhru))  

!! sj aug 09 end
	  allocate (hhsurf_bs(2,mhru,time%step))  !! nstep changed to nstep  OCt. 18,2007
      allocate (ubnrunoff(time%step),ubntss(time%step))

!! Arrays for subdaily erosion modeling by Jaehak Jeong
	  allocate (hhsedy(mhru,time%step),ovrlnd_dt(mhru,time%step))  
	  allocate (init_abstrc(mhru),hhsurfq(mhru,time%step))

       !Tillage factor on SOM decomposition
       allocate(tillage_switch(mhru))
       allocate(tillage_depth(mhru))
       allocate(tillage_days(mhru))
       allocate(tillage_factor(mhru))
       
       tillage_switch = 0
       tillage_depth = 0.
       tillage_days = 0
       tillage_factor = 0.
       
      !! By Zhang for C/N cycling
      !! ============================
      	  
      call zero0
      call zero1
      call zero2
      call zeroini

!!    zero reservoir module
      return
      end