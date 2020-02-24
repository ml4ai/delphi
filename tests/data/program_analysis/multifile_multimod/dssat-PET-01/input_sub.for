C=======================================================================
C  COPYRIGHT 1998-2010 The University of Georgia, Griffin, Georgia
C                      University of Florida, Gainesville, Florida
C                      International Center for Soil Fertility and 
C                       Agricultural Development, Muscle Shoals, Alabama
C                     
C  ALL RIGHTS RESERVED
C=======================================================================
C=======================================================================
C  INPUT, Subroutine
C
C  INPUT MODULE FOR DSSAT MODELS,  DSSAT v4.5
C
C  October 2007      Gerrit Hoogenboom, Cheryl Porter and Jim Jones
C
C
C  Reads FileX, includes sensitivity analysis and writes a
C  temporary output file for input by the crop models
C-----------------------------------------------------------------------
C  Revision history
C
C 04/26/1993 GH  Written
C 05/28/1993 PWW Header revision and minor changes
C 01/12/1993 WTB Modified for soil P model                     .
C 06/12/1994 GH  Set to 1994 version (MINPT940.EXE)
C 02/15/1995 GH  Set to 1995 version (MINPT950.EXE)
C 03/25/1996 GH  Modified to add OILCROP Sunflower model
C 15/01/1996 GH  Add DSSAT v3.1 file structure for fileX
C 12/31/1996 GH  Modified to add chickpea and pigeonpea
C 01/03/1997 GH  Modified to add ALOHA Pineapple model
C 01/19/1997 GH  Modified to add pepper
C 09/29/1997 GH  Modified to add cotton and CSCOT model
C 05/06/1998 GH  Changed to May 15, 1998, DSSAT v3.5
C 05/07/1998 GH  Modified to add velvetbean
C 09/11/1998 GH  Modified to add cowpea
C 05/23/1999 GH  Changed to May 31, 1999, DSSAT v3.51
C 08/17/1999 GH  Modified to add cabbage
C 07/03/2000 GH  Modified for CVF compiler and modular CROPGRO
C 09/20/2000 GH  Modifed to add Brachiaria decumbens
C 09/20/2000 GH  Changed to September 20, 2000, DSSAT v3.7
C 11/04/2001 GH  Added CASUPRO model
C 12/13/2001 GH  Modified to fit with the CSM model
C 01/30/2002 GH  Modified for the new wheat model
C 04/15/2002 GH  Modified for sequence analysis
C 04/20/2002 GH  Modified temporary output file
C 06/06/2002 GH  Modified for Y2K output
C 12/25/2002 GH  Changed to December, 2002. CSM v3.9
C 03/31/2004 GH  Official release DSSAT v4.0, CSM040
C 08/31/2005 GH  Official release DSSAT Version 4.0.2.0, CSM040
C 02/21/2006 GH  Read Crop Module from DSSATPRO
! 05/18/2006 CHP Made into subroutine called by CSM
! 01/12/2007 CHP Treatment number (TRTNUM) and rotation number (ROTNUM)
!                added to argument string.
C 02/01/2007 GH  RNMODE=T option for Gencalc Batch files
C 02/07/2007 GH  Include path for FileX and rotation number to command
C                line
C-----------------------------------------------------------------------
C  INPUT  : None
C
C  LOCAL  : WMODI,WMODB,CROP,PRCROP,VARNO,VARTY,ERRKEY,ECOTYP,ECONO,
C           MODEL,FILEIO,ECONAM,VRNAME,TITLER,PATHMO,NLOOP,FROP,FTYPEN,RUN
C           LNSA,LNIC,LUNIO,NYRS,ERRNUM,ENDSIM,SENSMS,NSENS,YRIC,SEQNO
C           IVRTEM,IVRGRP,IPLT,ISIM,BEXIST,WRESR,WRESND,TSOC,PM06
C           PM09,SWINIT(20),INO3(20),INH4(20),TOTN(20),EFINOC,EFNFIX
C           AINO3,AINH4,TNMIN,ANO3,ANH4,TSWINI,ESW(20),SW(20),TLL,TSW,TDUL
C           TSAT,TPESW,CUMDEP,PESW,CO2,CLDVAR,THVAR,SDPRO,TRIFOL,SIZELF
C           THRESH,LNGSH,RHGHT,RWIDTH
C
C  OUTPUT :
C-----------------------------------------------------------------------
C  Called :
C
C  Calls  : ERROR CLEAR INTRO IPEXP IPSOIL IPVAR IPECO IPSLIN IPSLAN
C           SENS INSOIL WEATHR 
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  HDLAY  :
C=======================================================================
!      PROGRAM INPUT_PROGRAM
      SUBROUTINE INPUT_SUB(
     &    FILECTL, FILEIO, FILEX, MODELARG, PATHEX,       !Input
     &    RNMODE, ROTNUM, RUN, TRTNUM,                    !Input
     &    ISWITCH, CONTROL)                               !Output

      USE ModuleDefs
      IMPLICIT NONE
      SAVE

      INCLUDE 'COMSOI.blk'
      INCLUDE 'COMIBS.blk'
      INCLUDE 'COMSWI.blk'

      CHARACTER*  1 WMODI, RNMODE
      CHARACTER*  2 CROP,PRCROP
!      CHARACTER*  6 TRNARG
      CHARACTER*  6 VARNO,VARTY,ERRKEY,ECOTYP,ECONO
      CHARACTER*  8 MODEL, MODELARG
      CHARACTER* 12 INPUT, FILEX
      CHARACTER* 16 ECONAM,VRNAME
!      CHARACTER* 18 RUNARG
      CHARACTER* 25 TITLET
      CHARACTER* 30 FILEIO
      CHARACTER* 42 CHEXTR(NAPPL)
      CHARACTER* 80 PATHEX
      CHARACTER* 92 FILEX_P
      CHARACTER*120 INPUTX
      CHARACTER*120 WTHSTR, FILECTL
      CHARACTER*1000 ATLINE

      INTEGER       NLOOP,FROP,FTYPEN,RUN,IIRV(NAPPL)
      INTEGER       LUNIO,NYRS,ERRNUM,NSENS,YRIC
      INTEGER       IVRGRP,IPLT,ISIM,EXPP,EXPN,TRTN,TRTALL
      INTEGER       NFORC,NDOF,PMTYPE,ISENS, TRTNUM, ROTNUM
      INTEGER       LNSIM,LNCU,LNHAR,LNENV,LNTIL,LNCHE
      INTEGER       LNFLD,LNSA,LNIC,LNPLT,LNIR,LNFER,LNRES

!      INTEGER       IP,IPX
      INTEGER       IPX
C-SUN INTEGER       LNBLNK

      LOGICAL       FEXIST,INITIAL, UseSimCtr

      REAL          WRESR,WRESND,TSOC,SWINIT(NL),CO2
      REAL          INO3(NL),INH4(NL),EFINOC,EFNFIX
      REAL          AINO3,AINH4,TNMIN,ANO3,ANH4,TSWINI
      REAL          ESW(NL),SW(NL),TLL,TSW,TDUL,TSAT,TPESW,CUMDEP,PESW
      REAL          PLTFOR

      TYPE (ControlType) CONTROL
      TYPE (SwitchType)  ISWITCH

      PARAMETER (ERRKEY = 'INPUT ')
      PARAMETER (LUNIO  = 21)

C-----------------------------------------------------------------------
C     Get argument from runtime module to determine path and run mode
C-----------------------------------------------------------------------
C   Fortran Compaq Visual Fortran
C-----------------------------------------------------------------------
      CALL GETARG (0,INPUTX)
!      call path_adj(inputx)
      IPX = LEN_TRIM(INPUTX)
D     INPUTX = STDPATH // 'DSCSM047.EXE'
      CALL PATHD  (DSSATP,INPUTX,IPX)
      CONTROL % DSSATP = DSSATP

      write (*,*) 'fileio = ', FILEIO
      write (*,*) 'filex = ', FILEX
      WRITE (*,*) 'ma = ', MODELARG
      write (*,*) 'pathex = ', PATHEX
      write (*,*) 'rnmode = ', RNMODE
      write (*,*) 'rotnum = ', ROTNUM
      write (*,*) 'run = ', RUN
      write (*,*) 'trtnum = ', TRTNUM

C === code below commented out for initial development -- SKD, 02/2020 ===
C C-----------------------------------------------------------------------
C C
C C-----------------------------------------------------------------------
C !      INPUT = INPUTX((index(inputx,slash,back=.true.)+1):IPX)
C 
C C-----------------------------------------------------------------------
C C    Initialize and delete previous copy of FILEIO
C C-----------------------------------------------------------------------
C       INQUIRE (FILE = FILEIO,EXIST = FEXIST)
C       IF (FEXIST) THEN
C           OPEN (LUNIO, FILE = FILEIO,STATUS = 'UNKNOWN',IOSTAT=ERRNUM)
C           READ (LUNIO,40) EXPP,TRTN,TRTALL
C           READ (LUNIO,70,IOSTAT=ERRNUM) IOX,IDETO,IDETS,FROP,IDETG,
C      &            IDETC,IDETW,IDETN,IDETP,IDETD,IDETL,IDETH,IDETR
C           CLOSE (LUNIO,STATUS = 'DELETE')
C       ENDIF
C 
C C-----------------------------------------------------------------------
C C     BEGINNING of READING INPUT files
C C-----------------------------------------------------------------------
C       IF (RNMODE .EQ. 'I' .AND. RUN .EQ. 1) THEN
C         CALL CLEAR
C         CALL INTRO
C       ENDIF
C       NSENS  = 0
C       ISENS  = 0
C       TITLER(1:5) = '     '
C       
C       FILEX_P = TRIM(PATHEX)//FILEX
C       CALL Join_Trim(PATHEX, FILEX, FILEX_P)
C C-----------------------------------------------------------------------
C C     Call IPEXP
C C-----------------------------------------------------------------------
C        CALL IPEXP (MODEL, RUN, RNMODE, FILEX,PATHEX,FILEX_P, FILECTL,
C      &     SLNO,NYRS,VARNO,CROP,WMODI,
C      &     FROP,TRTN,EXPP,EXPN,TITLET,TRTALL,TRTNUM,ROTNUM, 
C      &     IIRV,FTYPEN,CHEXTR,NFORC,PLTFOR,NDOF,PMTYPE,
C      &     LNSIM,LNCU,LNHAR,LNENV,LNTIL,LNCHE,
C      &     LNFLD,LNSA,LNIC,LNPLT,LNIR,LNFER,LNRES, 
C      &     CONTROL, ISWITCH, UseSimCtr, MODELARG)
C 
C C-----------------------------------------------------------------------
C C     Call IPSOIL
C C-----------------------------------------------------------------------
C ! ** DEFAULT MESOL = 2 ** 3/26/2007
C !  MESOL = '1' Original soil layer distribution. Calls LYRSET.
C !  MESOL = '2' New soil layer distribution. Calls LYRSET2.
C !  MESOL = '3' User specified soil layer distribution. Calls LYRSET3.
C !     Skip soils field and soils input for sequence mode
C       IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN == 1) THEN
C         CALL IPSOIL_Inp (RNMODE,FILES,PATHSL,NSENS,ISWITCH)
C       ENDIF
C C-----------------------------------------------------------------------
C C     Call IPVAR 
C C-----------------------------------------------------------------------
C       IF (CROP .NE. 'FA') THEN
C         CALL IPVAR (FILEG,NSENS,RNMODE,VARNO,VARTY,VRNAME,PATHGE,
C      &              ECONO, MODEL, ATLINE, CROP)
C       ENDIF
C 
C C-----------------------------------------------------------------------
C C     Call IPSLIN to read initial soil conditions
C C-----------------------------------------------------------------------
C !      IF (ISWWAT .NE. 'N' .AND. MESIC .EQ. 'M') THEN
C       IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN == 1) THEN
C          CALL IPSLIN (FILEX,FILEX_P,LNIC,NLAYR,DUL,YRIC,PRCROP,WRESR,
C      &        WRESND,EFINOC,EFNFIX,PEDON,SLNO,DS,SWINIT,INH4,INO3,
C      &        ISWITCH,ICWD,ICRES,ICREN,ICREP,ICRIP,ICRID,YRSIM) 
C          IF (ISIMI .EQ. 'I') THEN
C            IF (YRIC .LT. YRSIM .AND. YRIC .GT. 0) THEN
C              YRSIM = YRIC
C              CALL YR_DOY (YRSIM,YEAR,ISIM)
C              IF (MEWTH .EQ. 'M' .OR. MEWTH .EQ. 'G') THEN
C                 WRITE (FILEW(5:6),'(I2)') YEAR
C              ENDIF
C            ENDIF
C          ENDIF
C 
C C-----------------------------------------------------------------------
C C        Call IPSLAN to read soil analysis information
C C-----------------------------------------------------------------------
C          IF (ISWNIT .EQ. 'Y') THEN
C             CALL IPSLAN (FILEX, FILEX_P,LNSA, BD, DS, EXK, EXTP, OC,
C      &            PEDON, PH, PHKCL, SLNO, SMHB, SMKE, SMPX, TOTN, 
C      &            SASC, NLAYR)
C          ENDIF
C !      ENDIF
C       ENDIF
C C-----------------------------------------------------------------------
C C        Sensitivity Analysis Section
C C-----------------------------------------------------------------------
C       IF (INDEX('IE',RNMODE) .GT. 0 .AND. NYRS .EQ. 1) THEN
C          IF (INDEX('I',RNMODE) .GT. 0) THEN
C            NLOOP = 0
C   300      CONTINUE
C            NLOOP = NLOOP + 1
C            IF (NLOOP .GT. 25) CALL ERROR (ERRKEY,1,' ',0)
C            CALL CLEAR
C            WRITE (*,400)
C            READ (5,'(I2)',ERR = 300) NSENS
C          ELSE
C            NSENS = 1
C          ENDIF
C          IF (NSENS .EQ. 1) THEN
C             INITIAL = (ISWWAT .EQ.'N')
C C            CALL SENS (NSENS,VARNO,VARTY,VRNAME,FTYPEN,LNIC,LNSA,
C C     &        WRESR,WRESND,ISIM,NYRS,IPLT,WMODI,ECONO,ECONAM,ECOTYP,
C C     &        PRCROP,SWINIT,INO3,INH4,RUN,FROP,YRIC,EFINOC,EFNFIX,
C C     &        CROP,IVRGRP,ISENS,MODEL, RNMODE, FILEX,FILEX_P, 
C C     &        ISWITCH,CONTROL)
C             IF (INITIAL) THEN
C                IF ((ISWNIT .EQ. 'Y') .OR. (ISWWAT .NE.'N')) THEN
C                   NSENS = 0
C                   CALL IPSOIL_Inp(RNMODE,FILES,PATHSL,NSENS,ISWITCH)
C                   CALL IPSLIN (FILEX,FILEX_P,LNIC,NLAYR,DUL,YRIC,
C      &                 PRCROP,WRESR,WRESND,EFINOC,EFNFIX,PEDON,SLNO,DS,
C      &                 SWINIT,INH4,INO3,ISWITCH,
C      &                 ICWD,ICRES,ICREN,ICREP,ICRIP,ICRID,YRSIM) 
C                   CALL IPSLAN (FILEX, FILEX_P,LNSA, BD, DS, EXK, EXTP, 
C      &            OC, PEDON, PH, PHKCL, SLNO, SMHB, SMKE, SMPX, TOTN, 
C      &            SASC, NLAYR)
C                   NSENS = 1
C                ENDIF
C             ENDIF
C          ENDIF
C          WRITE (*,1000) RUN
C          READ (5,'(A25)') TITLER
C          IF (TITLER .EQ. '                         ') THEN
C             TITLER = TITLET
C          ENDIF
C        ELSE
C          TITLER = TITLET
C       ENDIF
C       
C C     Regenarate short headers now that Run Title is known.
C       CALL OPHEAD (RUNINIT,99,0.0,0.0,"                ",0.0,0.0, 
C      &     "      ",RUN,MODEL,TITLER,WTHSTR, RNMODE,
C      &     CONTROL, ISWITCH, UseSimCtr, PATHEX)
C 
C C-----------------------------------------------------------------------
C C     Call INSOIL to calculate initial conditions for each soil layer
C C-----------------------------------------------------------------------
C !     Skip soils field and soils input for sequence mode
C       IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN == 1) THEN
C 
C         CALL INSOIL (ISWWAT,ISWNIT,AINO3,ANO3,AINH4,ANH4,TNMIN,
C      &  SWINIT,TSWINI,NLAYR,DUL,LL,ESW,DLAYR,SAT,SW,TLL,TDUL,
C      &  TSAT,TPESW,CUMDEP,PESW,TSW,BD,INO3,INH4,TSOC,OC,PH,
C      &  RESN,RESP,RESIDUE,RINP,DEPRES,ICRES,ICREN,ICREP,ICRIP,
C      &  ICRID,NARES,YRSIM,RESAMT,RESDAY,SLTX,SLTXS,TOTN)
C       ENDIF
C       
C C-----------------------------------------------------------------------
C C     Call WEATHR to set CO2 conditions and weather parameter modifications
C C-----------------------------------------------------------------------
C 
C       CALL WEATHR_Inp (CO2ADJ,CO2FAC,DAYADJ,DAYFAC,DPTADJ,DPTFAC,PRCADJ,
C      &     PRCFAC,RADADJ,RADFAC,TMADJ,TMFAC,TXADJ,TXFAC,WMODI,WNDADJ,
C      &     WNDFAC,WTHADJ,CO2,WTHSTR,NEV)
C       
C C-----------------------------------------------------------------------
C C     Write temporary output files for runtime modules
C C-----------------------------------------------------------------------
C C     Write DSSAT Format Version 4 Output file for input by Version 4
C C     
C C-----------------------------------------------------------------------
C       
C         CALL OPTEMPY2K(RNMODE,FILEX,PATHEX,
C      &            YRIC,PRCROP,WRESR,WRESND,EFINOC,EFNFIX,
C      &            SWINIT,INH4,INO3,NYRS,VARNO,VRNAME,CROP,MODEL,
C      &            RUN,FILEIO,EXPN,ECONO,FROP,TRTALL,TRTN,
C      &            CHEXTR,NFORC,PLTFOR,NDOF,PMTYPE,ISENS)
C       
C         CALL OPTEMPXY2K (YRIC,PRCROP,WRESR,WRESND,EFINOC,EFNFIX,
C      &           SWINIT,INH4,INO3,NYRS,VARNO,VRNAME,CROP,
C      &           FILEIO,FROP,ECONO,ATLINE,
C      &           LNSIM,LNCU,LNHAR,LNENV,LNTIL,LNCHE,
C      &           LNFLD,LNSA,LNIC,LNPLT,LNIR,LNFER,LNRES,
C      &           NFORC,PLTFOR,PMTYPE,NDOF,CHEXTR, MODEL, PATHEX)
C 
C C-----------------------------------------------------------------------
C C     Write DSSAT Format Version 4 Output files
C C-----------------------------------------------------------------------
C       
C       CALL OPGEN (CUMDEP,TPESW,VRNAME,AINO3,AINH4,TLL,TDUL,TSAT,
C      &     TSWINI,RUN,MODEL,CROP,CROPD,TITLET,ECONO,VARTY,
C      &     ESW,SWINIT,INO3,INH4,TSOC,WTHSTR,NYRS, RNMODE, 
C      &     CONTROL, ISWITCH, UseSimCtr, ATLINE, PATHEX)
C 
C C-----------------------------------------------------------------------
C C     FORMAT Strings
C C-----------------------------------------------------------------------
C 
C   40  FORMAT (36X,3(1X,I5))
C   70  FORMAT (17(/),14X,3(5X,A1),4X,I2,9(5X,A1))
C  400  FORMAT (/////,5X,'What Would You Like To Do ?',
C      &            //,1X,' 0. Run Simulation.',
C      &             /,1X,' 1. Select Sensitivity Analysis Options.',
C      &            //,1X,'    CHOICE ?   [ Default = 0 ] ===> ',$)
C  1000 FORMAT (/,5X,'Please enter Run',I3,' name : ===> ',$)

      END SUBROUTINE INPUT_SUB
