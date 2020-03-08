C     This module is written to work around yet another problem in
C     that embarrassing piece of software known as OpenFortranParser.  

      MODULE DATA_STMT_HACK

      USE ModuleDefs
      
      IMPLICIT NONE

      LOGICAL RunList_FIRST, MulchWarn
      CHARACTER*6, DIMENSION(EvaluateNum) :: OLAB, OLAP

      DATA RunList_FIRST /.TRUE./
      DATA MulchWarn /.FALSE./

!     Define headings for observed data file (FILEA)
      DATA OLAB / !Pred.          Obs.   Definition
                  !------------   -----  -----------
     & 'ADAT  ', ! 1 DNR1           DFLR   Anthesis date
     & 'PD1T  ', ! 2 DNR3           DFPD   First Pod        
     & 'PDFT  ', ! 3 DNR5           DFSD   First Seed       
     & 'MDAT  ', ! 4 DNR7           DMAT   Physiological Maturity
     & 'HWAM  ', ! 5 NINT(SDWT*10)  XGWT   Seed Yield (kg/ha;dry)
     & 'PWAM  ', ! 6 NINT(PODWT*10) XPDW   Pod Yield (kg/ha;dry) 
     & 'H#AM  ', ! 7 NINT(SEEDNO)   XNOGR  Seed Number (Seed/m2)
     & 'HWUM  ', ! 8 PSDWT          XGWU   Weight Per Seed (g;dry)
     & 'H#UM  ', ! 9 PSPP           XNOGU  Seeds/Pod
     & 'CWAM  ', !10 NINT(TOPWT*10) XCWT   Biomass (kg/ha) Harvest Mat.

!     & 'BWAH', !11 (STMWT)*10    XSWT   Stem weight (kg/ha) at Mat.
! KJB, LAH, CHP 12/16/2004  change BWAH to BWAM
!     & 'BWAM', !11 (TOPWT-SDWT)*10 XSWT  Tops - seed (kg/ha) at Mat.
! CHP/GH 08/11/2005 Change BWAM = TOPWT - PODWT
     & 'BWAM  ', !11 (TOPWT-PODWT)*10 XSWT Tops - seed (kg/ha) at Mat.

     & 'LAIX  ', !12 LAIMX          XLAM   Maximum LAI (m2/m2)
     & 'HIAM  ', !13 HI             XHIN   Harvest Index (kg/kg)
     & 'THAM  ', !14 THRES          XTHR   Shelling Percentage (%)
     & 'GNAM  ', !15 NINT(WTNSD*10) XNGR   Seed N (kg N/ha)
     & 'CNAM  ', !16 NINT(WTNCAN*10)XNTP   Biomass N (kg N/ha)
     & 'SNAM  ', !17 NINT(WTNST*10) XNST   Stalk N (kg N/ha)
     & 'GN%M  ', !18 PCNSD          XNPS   Seed N (%)
     & 'CWAA  ', !19 NINT(CANWAA*10)XCWAA  Biomass (kg/ha) at Anthesis
     & 'CNAA  ', !20                XCNAA 
     & 'L#SM  ', !21 VSTAGE         XLFNO  Final Leaf Number (Main Stem)
     & 'GL%M  ', !22 PCLSD          XLPS   Seed Lipid (%)
     & 'CHTA  ', !23 CANHT          XCNHT  Canopy Height (m)
     & 'R8AT  ', !24 DNR8           DHRV   Harvest Maturity (dap)
     & 'EDAT  ', !25 EDAT                  Emergence date
     & 15*'      '/


      END  MODULE DATA_STMT_HACK
