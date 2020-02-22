      subroutine sq_greenampt

!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    Predicts daily runoff given breakpoint precipitation and snow melt
!!    using the Green & Ampt technique

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    nstep       |none          |max number of time steps per day
!!    nstep       |none          |number of rainfall time steps for day
!!    wst(:)%weat%ts(:)  |mm H2O        |precipitation for the time step during day
!!    swtrg(:)    |none          |rainfall event flag:
!!                               |  0: no rainfall event over midnight
!!                               |  1: rainfall event over midnight
!!    wfsh(:)     |mm            |average capillary suction at wetting front
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    hhqday(:)   |mm H2O        |surface runoff generated each hour of day
!!                               |in HRU
!!    surfq(:)    |mm H2O        |surface runoff for the day in HRU
!!    swtrg(:)    |none          |rainfall event flag:
!!                               |  0: no rainfall event over midnight
!!                               |  1: rainfall event over midnight
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ SUBROUTINES/FUNCTIONS CALLED ~ ~ ~
!!    Intrinsic: Sum, Exp, Real, Mod

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~

      use urban_data_module
      use climate_module
      use basin_module
      use hydrograph_module
      use hru_module, only : hru, swtrg, hhqday, ubnrunoff, hhsurfq, surfq, cnday, wfsh, ihru
      use soil_module
      use time_module
      
      implicit none
      
      integer :: j                               !none          |HRU number
      integer :: k                               !none          |counter
      integer :: kk                              !hour          |hour of day in which runoff is generated
      real :: adj_hc                             !mm/hr         |adjusted hydraulic conductivity
      real :: dthet                              !mm/mm         |initial moisture deficit
      real :: soilw                              !mm H2O        |amount of water in soil profile
      real :: psidt                              !mm            |suction at wetting front*initial moisture 
                                                 !              |deficit
      real :: tst                                !mm H2O        |test value for cumulative infiltration
      real :: f1                                 !mm H2O        |test value for cumulative infiltration
      real :: ulu
      real :: abstinit
      real, dimension (time%step+1) :: cuminf    !mm H2O        |cumulative infiltration for day
      real, dimension (time%step+1) :: cumr      !mm H2O        |cumulative rainfall for day
      real, dimension (time%step+1) :: excum     !mm H2O        |cumulative runoff for day
      real, dimension (time%step+1) :: exinc     !mm H2O        |runoff for time step
      real, dimension (time%step+1) :: rateinf   !mm/hr         |infiltration rate for time step
      real, dimension (time%step+1) :: rintns    !mm/hr         |rainfall intensity

      !! array location #1 is for last time step of prev day

       j = ihru
       ulu = hru(j)%luse%urb_lu
       
       !! reset values for day
       cumr = 0.
       cuminf = 0.
       excum = 0.
       exinc = 0.
       rateinf = 0.
       rintns = 0.

       !! calculate effective hydraulic conductivity
       adj_hc = (56.82 * soil(j)%phys(1)%k ** 0.286)                 &         
                       / (1. + 0.051 * Exp(0.062 * cnday(j))) - 2.
       if (adj_hc <= 0.) adj_hc = 0.001

       dthet = 0.
       if (swtrg(j) == 1) then
         swtrg(j) = 0
         dthet = 0.001 * soil(j)%phys(1)%por * 0.95
       else
         if (soil(j)%sw >= soil(j)%sumfc) then
           soilw = 0.999 * soil(j)%sumfc
         else
           soilw = soil(j)%sw
         end if
         dthet = (1. - soilw / soil(j)%sumfc) *                  & 
                                      soil(j)%phys(1)%por * 0.95
         rateinf(1) = 2000.
       end if

       psidt = 0.
       psidt = dthet * wfsh(j)

       k = 1
       rintns(1) = 60. * wst(iwst)%weat%ts(2) / Real(time%dtm)  !! urban 60./idt  NK Feb 4,08

       do k = 2, time%step+1
         !! calculate total amount of rainfall during day for time step
         cumr(k) = cumr(k-1) + wst(iwst)%weat%ts(k)
         !! and rainfall intensity for time step
         rintns(k) = 60. * wst(iwst)%weat%ts(k) / Real(time%dtm) !!urban 60./idt NK Feb 4,08 

         !! if rainfall intensity is less than infiltration rate
         !! everything will infiltrate
         if (rateinf(k-1) >= rintns(k-1)) then
           cuminf(k) = cuminf(k-1) + rintns(k-1) *                       & 
              Real(time%dtm) / 60. !!urban 60./idt NK Feb 4,08
           if (excum(k-1) > 0.) then
             excum(k) = excum(k-1)
             exinc(k) = 0.
           else
             excum(k) = 0.
             exinc(k) = 0.
           end if
          else
          !! if rainfall intensity is greater than infiltration rate
          !! find cumulative infiltration for time step by successive
          !! substitution
           tst = adj_hc * Real(time%dtm) / 60.  !!urban 60./idt NK Feb 4,08
           do
             f1 = 0.
             f1 = cuminf(k-1) + adj_hc * Real(time%dtm) / 60. +        &
                   psidt * Log((tst + psidt)/(cuminf(k-1) + psidt))
             if (Abs(f1 - tst) <= 0.001) then
               cuminf(k) = f1
               excum(k) = cumr(k) - cuminf(k)
               exinc(k) = excum(k) - excum(k-1)
               if (exinc(k) < 0.) exinc(k) = 0.
               hhqday(j,k-1) = exinc(k)
               exit
             else
               tst = 0.
               tst = f1
             end if
           end do
         end if  

	   !! Urban Impervious cover 
	   if (hru(j)%luse%urb_lu > 0) then
	      !runoff from pervious area
	      hhqday(j,k-1) = hhqday(j,k-1) * (1. - urbdb(ulu)%fcimp) 
           
           !runoff from impervious area with initial abstraction
            ubnrunoff(k-1) = (wst(iwst)%weat%ts(k) - abstinit) * urbdb(ulu)%fcimp
            if ( ubnrunoff(k-1)<0)  ubnrunoff(k-1) = 0.
         end if

	   !! daily total runoff
	   hhsurfq(j,k-1) = hhqday(j,k-1) + ubnrunoff(k-1)
       surfq(j) = surfq(j) + hhsurfq(j,k-1) 

         !! calculate new rate of infiltration
         rateinf(k) = adj_hc * (psidt / (cuminf(k) + 1.e-6) + 1.)
        
      end do
       
      if (Sum(wst(iwst)%weat%ts) > 12.) then
        swtrg(j) = 1
      end if

      return
 5000 format(//,"Excess rainfall calculation for day ",i3," of year ",   &  
              i4," for sub-basin",i4,".",/)
 5001 format(t2,"Time",t9,"Incremental",t22,"Cumulative",t35,"Rainfall",   &
             t45,"Infiltration",t59,"Cumulative",t71,"Cumulative",t82,     &  
             "Incremental",/,t2,"Step",t10,"Rainfall",t23,"Rainfall",      &   
             t35,"Intensity",t49,"Rate",t58,"Infiltration",t73,"Runoff",   & 
             t84,"Runoff",/,t12,"(mm)",t25,"(mm)",t36,"(mm/h)",t48,        &    
             "(mm/h)",t62,"(mm)",t74,"(mm)",t85,"(mm)",/)
 5002 format(i5,t12,f5.2,t24,f6.2,t36,f6.2,t47,f7.2,t61,f6.2,t73,f6.2,     &  
             t84,f6.2)
      end subroutine sq_greenampt