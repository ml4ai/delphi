!=======================================================================
!  MODULE OSDefinitions
!  08/08/2017 WP Written
!=======================================================================

      MODULE OSDefinitions
!     Contains defintion for Linux Platform which are used throughout 
!     the model.

      SAVE

!=======================================================================

      CHARACTER(LEN=1), PARAMETER   :: SLASH    = '/'
      CHARACTER(LEN=12), PARAMETER  :: DSSATPRO = 'DSSATPRO.L47'
      CHARACTER(LEN=255)            :: STDPATH  = '/DSSAT47/'

!======================================================================
      END MODULE OSDefinitions
!======================================================================


!=======================================================================
C  MODULE ModuleDefs
C  11/01/2001 CHP Written
C  06/15/2002 CHP Added flood-related data constructs 
C  03/12/2003 CHP Added residue data construct
C  05/08/2003 CHP Added version information
C  09/03/2004 CHP Modified GetPut_Control routine to store entire
C                   CONTROL variable. 
C             CHP Added GETPUT_ISWITCH routine to store ISWITCH.
C             CHP Added TRTNUM to CONTROL variable.
!  06/14/2005 CHP Added FILEX to CONTROL variable.
!  10/24/2005 CHP Put weather variables in constructed variable. 
!             CHP Added PlantStres environmental stress variable
!  11/07/2005 CHP Added KG2PPM conversion variable to SoilType
!  03/03/2006 CHP Tillage variables added to SOILPROP
!                 Added N_ELEMS to CONTROL variable.
!  03/06/2006 CHP Added mulch variable
!  07/13/2006 CHP Add P variables to SwitchType and SoilType TYPES
!  07/14/2006 CHP Added Fertilizer type, Organic Matter Application type
!  07/24/2006 CHP Added mulch/soil albedo (MSALB) and canopy/mulch/soil
!                   albedo (CMSALB) to SOILPROP variable
!  01/12/2007 CHP Changed TRTNO to TRTNUM to avoid conflict with same
!                 variable name (but different meaning) in input module.
!  01/24/2007 CHP Changed GET & PUT routines to more extensible format.
!  01/22/2008 CHP Moved data exchange (GET & PUT routines) to ModuleData
!  04/28/2008 CHP Added option to read CO2 from file 
!  08/08/2008 CHP Use OPSYS to define variables dependant on operating system
!  08/08/2008 CHP Compiler directives for system call library
!  08/21/2008 CHP Add ROTNUM to CONTROL variable
!  11/25/2008 CHP Mauna Loa CO2 is default
!  12/09/2008 CHP Remove METMP
!  11/19/2010 CHP Added "branch" to version to keep track of non-release branches
!  08/08/2017 WP  Version identification moved to CSMVersion.for
!  08/08/2017 WP  Definitions related with OS platform moved to OSDefinitions.for
!=======================================================================

      MODULE ModuleDefs
!     Contains defintion of derived data types and constants which are 
!     used throughout the model.

!=======================================================================
      USE OSDefinitions
      SAVE
!=======================================================================

!     Global CSM Version Number
      TYPE VersionType
        INTEGER :: Major = 4
        INTEGER :: Minor = 7
        INTEGER :: Model = 2
        INTEGER :: Build = 0
      END TYPE VersionType
      TYPE (VersionType) Version
!     CHARACTER(len=10) :: VBranch = '-develop  '
      CHARACTER(len=10) :: VBranch = '-release  '

!     Version history:  
!       4.7.2.0  chp 05/07/2018 v4.7.2 Release 2018 Workshop
!       4.7.1.0  chp 10/27/2017 v4.7.1 Release
!       4.7.0.0  chp 08/09/2017 v4.7.0 Release
!       4.6.5.1  chp 05/10/2017 v4.6.5 Release 2017 Workshop  
!       4.6.0.1  chp 06/28/2011 v4.6.0 Release
!       4.5.1.0  chp 10/10/2010 V4.5.1 Release
!       4.0.2.0  chp 08/11/2005 v4.0.2 Release
!       4.0.1.0  chp 01/28/2004 v4.0.1 Release 

!=======================================================================

!     Global constants
      INTEGER, PARAMETER :: 
     &    NL       = 20,  !Maximum number of soil layers 
     &    TS       = 24,  !Number of hourly time steps per day
     &    NAPPL    = 9000,!Maximum number of applications or operations
     &    NCOHORTS = 300, !Maximum number of cohorts
     &    NELEM    = 3,   !Number of elements modeled (currently N & P)
!            Note: set NELEM to 3 for now so Century arrays will match
     &    NumOfDays = 1000, !Maximum days in sugarcane run (FSR)
     &    NumOfStalks = 42, !Maximum stalks per sugarcane stubble (FSR)
     &    EvaluateNum = 40, !Number of evaluation variables
     &    MaxFiles = 500,   !Maximum number of output files
     &    MaxPest = 500    !Maximum number of pest operations

      REAL, PARAMETER :: 
     &    PI = 3.14159265,
     &    RAD=PI/180.0

      INTEGER, PARAMETER :: 
         !Dynamic variable values
     &    RUNINIT  = 1, 
     &    INIT     = 2,  !Will take the place of RUNINIT & SEASINIT
                         !     (not fully implemented)
     &    SEASINIT = 2, 
     &    RATE     = 3,
     &    EMERG    = 3,  !Used for some plant processes.  
     &    INTEGR   = 4,  
     &    OUTPUT   = 5,  
     &    SEASEND  = 6,
     &    ENDRUN   = 7 

      INTEGER, PARAMETER :: 
         !Nutrient array positions:
     &    N = 1,          !Nitrogen
     &    P = 2,          !Phosphorus
     &    Kel = 3         !Potassium

      CHARACTER(LEN=3)  ModelVerTxt
      CHARACTER(LEN=6)  LIBRARY    !library required for system calls

      CHARACTER*3 MonthTxt(12)
      DATA MonthTxt /'JAN','FEB','MAR','APR','MAY','JUN',
     &               'JUL','AUG','SEP','OCT','NOV','DEC'/

!=======================================================================
!     Data construct for control variables
      TYPE ControlType
        CHARACTER (len=1)  MESIC, RNMODE
        CHARACTER (len=2)  CROP
        CHARACTER (len=8)  MODEL, ENAME
        CHARACTER (len=12) FILEX
        CHARACTER (len=30) FILEIO
        CHARACTER (len=102)DSSATP
        CHARACTER (len=120) :: SimControl = 
     &  "                                                            "//
     &  "                                                            "
        INTEGER   DAS, DYNAMIC, FROP, ErrCode, LUNIO, MULTI, N_ELEMS
        INTEGER   NYRS, REPNO, ROTNUM, RUN, TRTNUM
        INTEGER   YRDIF, YRDOY, YRSIM
      END TYPE ControlType

!=======================================================================
!     Data construct for control switches
      TYPE SwitchType
        CHARACTER (len=1) FNAME
        CHARACTER (len=1) IDETC, IDETD, IDETG, IDETH, IDETL, IDETN
        CHARACTER (len=1) IDETO, IDETP, IDETR, IDETS, IDETW
        CHARACTER (len=1) IHARI, IPLTI, IIRRI, ISIMI
        CHARACTER (len=1) ISWCHE, ISWDIS, ISWNIT
        CHARACTER (len=1) ISWPHO, ISWPOT, ISWSYM, ISWTIL, ISWWAT
        CHARACTER (len=1) MEEVP, MEGHG, MEHYD, MEINF, MELI, MEPHO
        CHARACTER (len=1) MESOM, MESOL, MESEV, MEWTH
        CHARACTER (len=1) METMP !Temperature, EPIC
        CHARACTER (len=1) IFERI, IRESI, ICO2, FMOPT
        INTEGER NSWI
      END TYPE SwitchType

!Other switches and methods used by model:
! MELI, IOX - not used
! IDETH - only used in MgmtOps
! MEWTH - only used in WEATHR

!=======================================================================
!     Data construct for weather variables
      TYPE WeatherType
        SEQUENCE

!       Weather station information
        REAL REFHT, WINDHT, XLAT

!       Daily weather data.
        REAL CLOUDS, CO2, DAYL, DCO2, PAR, RAIN, RHUM, SNDN, SNUP, 
     &    SRAD, TAMP, TA, TAV, TAVG, TDAY, TDEW, TGROAV, TGRODY,      
     &    TMAX, TMIN, TWILEN, VAPR, WINDSP, VPDF, VPD_TRANSP

!       Hourly weather data
        REAL, DIMENSION(TS) :: AMTRH, AZZON, BETA, FRDIFP, FRDIFR, PARHR
        REAL, DIMENSION(TS) :: RADHR, RHUMHR, TAIRHR, TGRO, WINDHR

      END TYPE WeatherType

!=======================================================================
!     Data construct for soil variables
      TYPE SoilType
        INTEGER NLAYR
        CHARACTER (len=5) SMPX
        CHARACTER (len=10) SLNO
        CHARACTER (len=12) TEXTURE(NL)
        CHARACTER (len=17) SOILLAYERTYPE(NL)
        CHARACTER*50 SLDESC, TAXON
        
        LOGICAL COARSE(NL)
        
        REAL ALES, DMOD, SLPF         !DMOD was SLNF
        REAL CMSALB, MSALB, SWALB, SALB      !Albedo 
        REAL, DIMENSION(NL) :: BD, CEC, CLAY, DLAYR, DS, DUL
        REAL, DIMENSION(NL) :: KG2PPM, LL, OC, PH, PHKCL, POROS
        REAL, DIMENSION(NL) :: SAND, SAT, SILT, STONES, SWCN
        
      !Residual water content
        REAL, DIMENSION(NL) :: WCR

      !vanGenuchten parameters
        REAL, DIMENSION(NL) :: alphaVG, mVG, nVG

      !Second tier soils data:
        REAL, DIMENSION(NL) :: CACO3, EXTP, ORGP, PTERMA, PTERMB
        REAL, DIMENSION(NL) :: TOTP, TOTBAS, EXCA, EXK, EXNA

      !Soil analysis data 
        REAL, DIMENSION(NL) :: SASC   !stable organic C

      !Variables added with new soil format:
        REAL ETDR, PONDMAX, SLDN, SLOPE
!       REAL, DIMENSION(NL) :: RCLPF, RGIMPF

      !Variables deleted with new soil format:
      !Still needed for Ritchie hydrology
        REAL CN, SWCON, U
        REAL, DIMENSION(NL) :: ADCOEF, TOTN, TotOrgN, WR

      !Text describing soil layer depth data
      !1-9 describe depths for layers 1-9
      !10 depths for layers 10 thru NLAYR (if NLAYR > 9)
      !11 depths for layers 5 thru NLAYR (if NLAYR > 4)
        CHARACTER*8 LayerText(11)

      !These variables could be made available if needed elsewhere.
      !  They are currently read by SOILDYN module.
      !  CHARACTER*5 SLTXS
      !  CHARACTER*11 SLSOUR
      !  CHARACTER*50 SLDESC, TAXON

      !Second tier soils data that could be used:
!        REAL, DIMENSION(NL) :: EXTAL, EXTFE, EXTMN, 
!        REAL, DIMENSION(NL) :: EXMG, EXTS, SLEC

      END TYPE SoilType

!=======================================================================
!     Data construct for mulch layer
      TYPE MulchType
        REAL MULCHMASS    !Mass of surface mulch layer (kg[dry mat.]/ha)
        REAL MULCHALB     !Albedo of mulch layer
        REAL MULCHCOVER   !Coverage of mulch layer (frac. of surface)
        REAL MULCHTHICK   !Thickness of mulch layer (mm)
        REAL MULCHWAT     !Water content of mulch (mm3/mm3)
        REAL MULCHEVAP    !Evaporation from mulch layer (mm/d)
        REAL MULCHSAT     !Saturation water content of mulch (mm3/mm3)
        REAL MULCHN       !N content of mulch layer (kg[N]/ha)
        REAL MULCHP       !P content of mulch layer (kg[P]/ha)
        REAL NEWMULCH     !Mass of new surface mulch (kg[dry mat.]/ha)
        REAL NEWMULCHWAT  !Water content of new mulch ((mm3/mm3)
        REAL MULCH_AM     !Area covered / dry weight of residue (ha/kg)
        REAL MUL_EXTFAC   !Light extinction coef. for mulch layer
        REAL MUL_WATFAC   !Saturation water content (mm[water] ha kg-1)
      END TYPE MulchType

!=======================================================================
!     Data construct for tillage operations
      TYPE TillType
        INTEGER NTIL      !Total number of tillage events in FILEX
        INTEGER TILDATE   !Date of current tillage event

!       Maximum values for multiple events in a single day
        REAL TILDEP, TILMIX, TILRESINC

!       Irrigation amount which affects tilled soil properties 
!          expressed in units of equivalent rainfall depth
        REAL TIL_IRR   

!       Allows multiple tillage passes in a day
        INTEGER NTil_today !number of tillage events today (max 3)
        INTEGER, DIMENSION(NAPPL) :: NLYRS
        REAL, DIMENSION(NAPPL) :: CNP, TDEP
        REAL, DIMENSION(NAPPL,NL) :: BDP, DEP, SWCNP
      END TYPE TillType

!=======================================================================
!     Data construct for oxidation layer
      TYPE OxLayerType
        INTEGER IBP
        REAL    OXU, OXH4, OXN3   
        REAL    OXLT, OXMIN4, OXMIN3
        REAL    DLTOXU, DLTOXH4, DLTOXN3
        REAL    ALGACT
        LOGICAL DailyCalc
      END TYPE OxLayerType

!======================================================================
!     Fertilizer application data
      TYPE FertType
        CHARACTER*7 AppType != 'UNIFORM', 'BANDED ' or 'HILL   '
        INTEGER FERTDAY, FERTYPE
        INTEGER, DIMENSION(NELEM) :: NAPFER
        REAL FERDEPTH, FERMIXPERC
        REAL ADDFNH4, ADDFNO3, ADDFUREA
        REAL ADDOXU, ADDOXH4, ADDOXN3
        REAL, DIMENSION(NELEM) :: AMTFER
        REAL, DIMENSION(NL) :: ADDSNH4, ADDSNO3, ADDUREA
        REAL, DIMENSION(NL) :: ADDSPi
        REAL, DIMENSION(NL) :: ADDSKi
        LOGICAL UNINCO
      END TYPE FertType

!=======================================================================
!   Data construct for residue (harvest residue, senesced matter, etc.)
      TYPE ResidueType
        REAL, DIMENSION(0:NL) :: ResWt        !kg[dry matter]/ha/d
        REAL, DIMENSION(0:NL) :: ResLig       !kg[lignin]/ha/d
        REAL, DIMENSION(0:NL,NELEM) :: ResE   !kg[E]/ha/d (E=N,P,K,..)
        REAL  CumResWt                        !cumul. kg[dry matter]/ha
        REAL, DIMENSION(NELEM) :: CumResE     !cumulative kg[E]/ha
      END TYPE ResidueType

!======================================================================
!     Organic Matter Application data
      TYPE OrgMatAppType
        INTEGER NAPRes, ResDat, ResDepth
        CHARACTER (len=5) RESTYPE
        REAL ResMixPerc   !Percent mixing rate for SOM module

        REAL, DIMENSION(0:NL) :: ResWt        !kg[dry matter]/ha/d
        REAL, DIMENSION(0:NL) :: ResLig       !kg[lignin]/ha/d
        REAL, DIMENSION(0:NL,NELEM) :: ResE   !kg[E]/ha/d (E=N, P, ..)
        REAL  CumResWt                        !cumul. kg[dry matter]/ha
        REAL, DIMENSION(NELEM) :: CumResE     !cumulative kg[E]/ha
      END TYPE OrgMatAppType

!======================================================================
!     Plant stresses for environmental stress summary
      Type PlStresType
        INTEGER NSTAGES   !# of stages (max 5)
        CHARACTER(len=23) StageName(0:5)
        REAL W_grow, W_phot, N_grow, N_phot
        REAL P_grow, P_phot
        LOGICAL ACTIVE(0:5)
      End Type PlStresType

!======================================================================
!     Array of output files, aliases, unit numbers, etc.
      Type OutputType
        INTEGER NumFiles
        CHARACTER*16, DIMENSION(MaxFiles) :: FileName
        CHARACTER*2,  DIMENSION(MaxFiles) :: OPCODE
        CHARACTER*50, DIMENSION(MaxFiles) :: Description
        CHARACTER*10, DIMENSION(MaxFiles) :: ALIAS
        INTEGER, DIMENSION(MaxFiles) :: LUN
      End Type


!======================================================================
!      CONTAINS
!
!!----------------------------------------------------------------------
!      SUBROUTINE SETOP ()
!      IMPLICIT NONE
!
!      WRITE(ModelVerTxt,'(I2.2,I1)') Version%Major, Version%Minor
!
!      END SUBROUTINE SETOP

!======================================================================
      END MODULE ModuleDefs
!======================================================================


!=======================================================================
!  MODULE ModuleData
!  01/22/2008 CHP Written
!=======================================================================

      MODULE ModuleData
!     Data storage and retrieval module.
!     Defines data structures that hold information that can be 
!       stored or accessed by query.  

!     A call to the GET routine will return the value of variable 
!       requested.  The procedure is "overloaded", i.e., the procedure 
!       executed will depend on the type of variable(s) in the argument 
!       list.  A request for a "real" data type will invoke the GET_Real
!       procedure, for example.  

!     Similarly, a call to the PUT routine will store the data sent.
!       It is also an overloaded procedure including several different
!       types of data which can be stored.

!     The SAVE_data variable is used to store all information.

!     To add a variable for storage and retrieval: 
!     1.  Add the variable to one of the Type constructs based on the 
!         module that "owns" the variable, for example SPAMType, Planttype 
!         or MgmtType.
!     2.  For a real data type, add a line of code in both the GET_Real and
!         PUT_Real subroutines.  
!     3.  For an integer data type, add a line of code in both the 
!         GET_Integer and PUT_Integer subroutines.  
!     4.  All routines accessing GET or PUT procedures must include a 
!         "USE ModuleData" statement.
!     5.  A call to the PUT routine must be used to store data prior to
!         a call to the GET routine to retrive the data.

      USE ModuleDefs
      SAVE

!======================================================================
!     Data transferred from hourly energy balance 
      Type SPAMType
        REAL AGEFAC, PG                   !photosynthese
        REAL CEF, CEM, CEO, CEP, CES, CET !Cumulative ET - mm
        REAL  EF,  EM,  EO,  EP,  ES,  ET !Daily ET - mm/d
        REAL  EOP, EVAP                   !Daily mm/d
        REAL, DIMENSION(NL) :: UH2O       !Root water uptake
      End Type SPAMType

!     Data transferred from CROPGRO routine 
      TYPE PlantType
        REAL CANHT, CANWH, DXR57, EXCESS,
     &    PLTPOP, RNITP, SLAAD, XPOD
        REAL BIOMAS
        INTEGER NR5, iSTAGE, iSTGDOY
        CHARACTER*10 iSTNAME
      END TYPE PlantType

!     Data transferred from management routine 
      Type MgmtType
        REAL DEPIR, EFFIRR, FERNIT, IRRAMT, TOTIR, TOTEFFIRR
        REAL V_AVWAT(20)    ! Create vectors to save growth stage based irrigation
        REAL V_IMDEP(20)
        REAL V_ITHRL(20)
        REAL V_ITHRU(20)
        INTEGER V_IRON(20)
        REAL V_IRAMT(20)
        REAL V_IREFF(20)
        INTEGER V_IFREQ(20)
        INTEGER GSIRRIG
        CHARACTER*5 V_IRONC(20)
      End Type MgmtType

!     Data transferred from Soil water routine
      Type WatType
        REAL DRAIN, RUNOFF, SNOW
      End Type WatType

!     Data transferred from Soil Inorganic Nitrogen routine
      Type NiType
        REAL TNOXD, TLeachD    !, TN2OD     ! added N2O PG
      End Type NiType

!     Data transferred from Organic C routines
      Type OrgCType
        REAL TOMINFOM, TOMINSOM, TOMINSOM1, TOMINSOM2
        REAL TOMINSOM3, TNIMBSOM
        REAL MULCHMASS
      End Type OrgCType

!     Data from weather
      Type WeathType
        Character*8 WSTAT
      End Type WeathType

      TYPE PDLABETATYPE
        REAL PDLA
        REAL BETALS
      END TYPE

!     Data which can be transferred between modules
      Type TransferType
        Type (ControlType) CONTROL
        Type (SwitchType)  ISWITCH
        Type (OutputType)  OUTPUT
        Type (PlantType)   PLANT
        Type (MgmtType)    MGMT
        Type (NiType)      NITR
        Type (OrgCType)    ORGC
        Type (SoilType)    SOILPROP
        Type (SPAMType)    SPAM
        Type (WatType)     WATER
        Type (WeathType)   WEATHER
        TYPE (PDLABETATYPE) PDLABETA
      End Type TransferType

!     The variable SAVE_data contains all of the components to be 
!     stored and retrieved.
      Type (TransferType) SAVE_data

!======================================================================
!     GET and PUT routines are differentiated by argument type.  All of 
!       these procedures can be accessed with a CALL GET(...)
      INTERFACE GET
         MODULE PROCEDURE GET_Control
     &                  , GET_ISWITCH 
     &                  , GET_Output 
     &                  , GET_SOILPROP
!     &                  , GET_Weather
     &                  , GET_Real 
     &                  , GET_Real_Array_NL
     &                  , GET_Integer
     &                  , GET_Char
      END INTERFACE

      INTERFACE PUT
         MODULE PROCEDURE PUT_Control
     &                  , PUT_ISWITCH 
     &                  , PUT_Output 
     &                  , PUT_SOILPROP
!     &                  , PUT_Weather
     &                  , PUT_Real 
     &                  , PUT_Real_Array_NL
     &                  , PUT_Integer
     &                  , PUT_Char
      END INTERFACE

      CONTAINS

!----------------------------------------------------------------------
      Subroutine Get_CONTROL (CONTROL_arg)
!     Retrieves CONTROL variable
      IMPLICIT NONE
      Type (ControlType) Control_arg
      Control_arg = SAVE_data % Control
      Return
      End Subroutine Get_CONTROL

!----------------------------------------------------------------------
      Subroutine Put_CONTROL (CONTROL_arg)
!     Stores CONTROL variable
      IMPLICIT NONE
      Type (ControlType) Control_arg
      SAVE_data % Control = Control_arg
      Return
      End Subroutine Put_CONTROL

!----------------------------------------------------------------------
      Subroutine Get_ISWITCH (ISWITCH_arg)
!     Retrieves ISWITCH variable
      IMPLICIT NONE
      Type (SwitchType) ISWITCH_arg
      ISWITCH_arg = SAVE_data % ISWITCH
      Return
      End Subroutine Get_ISWITCH

!----------------------------------------------------------------------
      Subroutine Put_ISWITCH (ISWITCH_arg)
!     Stores ISWITCH variable
      IMPLICIT NONE
      Type (SwitchType) ISWITCH_arg
      SAVE_data % ISWITCH = ISWITCH_arg
      Return
      End Subroutine Put_ISWITCH

!----------------------------------------------------------------------
      SUBROUTINE GET_OUTPUT(OUTPUT_ARG)
!     Retrieves OUTPUT variable as needed
      IMPLICIT NONE
      TYPE (OutputType) OUTPUT_ARG
      OUTPUT_ARG = SAVE_data % OUTPUT
      RETURN
      END SUBROUTINE GET_OUTPUT

!----------------------------------------------------------------------
      SUBROUTINE PUT_OUTPUT(OUTPUT_ARG)
!     Stores OUTPUT variable 
      IMPLICIT NONE
      TYPE (OutputType) OUTPUT_ARG
      SAVE_data % OUTPUT = OUTPUT_ARG
      RETURN
      END SUBROUTINE PUT_OUTPUT

!----------------------------------------------------------------------
      SUBROUTINE GET_SOILPROP(SOIL_ARG)
!     Retrieves SOILPROP variable as needed
      IMPLICIT NONE
      TYPE (SoilType) SOIL_ARG
      SOIL_ARG = SAVE_data % SOILPROP
      RETURN
      END SUBROUTINE GET_SOILPROP

!----------------------------------------------------------------------
      SUBROUTINE PUT_SOILPROP(SOIL_ARG)
!     Stores SOILPROP variable 
      IMPLICIT NONE
      TYPE (SoilType) SOIL_ARG
      SAVE_data % SOILPROP = SOIL_ARG
      RETURN
      END SUBROUTINE PUT_SOILPROP

!!----------------------------------------------------------------------
!      SUBROUTINE GET_WEATHER(WEATHER_ARG)
!!     Retrieves WEATHER variable as needed
!      IMPLICIT NONE
!      TYPE (WeathType) WEATHER_ARG
!      WEATHER_ARG = SAVE_data % WEATHER
!      RETURN
!      END SUBROUTINE GET_WEATHER
!
!!----------------------------------------------------------------------
!      SUBROUTINE PUT_WEATHER(WEATHER_ARG)
!!     Stores WEATHER variable 
!      IMPLICIT NONE
!      TYPE (WeathType) WEATHER_ARG
!      SAVE_data % WEATHER = WEATHER_ARG
!      RETURN
!      END SUBROUTINE PUT_WEATHER

!----------------------------------------------------------------------
      Subroutine GET_Real(ModuleName, VarName, Value)
!     Retrieves real variable from SAVE_data.  Variable must be
!         included as a component of SAVE_data. 
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78 MSG(2)
      Real Value
      Logical ERR

      Value = 0.0
      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('SPAM')
        SELECT CASE (VarName)
        Case ('AGEFAC'); Value = SAVE_data % SPAM % AGEFAC
        Case ('PG');     Value = SAVE_data % SPAM % PG
        Case ('CEF');    Value = SAVE_data % SPAM % CEF
        Case ('CEM');    Value = SAVE_data % SPAM % CEM
        Case ('CEO');    Value = SAVE_data % SPAM % CEO
        Case ('CEP');    Value = SAVE_data % SPAM % CEP
        Case ('CES');    Value = SAVE_data % SPAM % CES
        Case ('CET');    Value = SAVE_data % SPAM % CET
        Case ('EF');     Value = SAVE_data % SPAM % EF
        Case ('EM');     Value = SAVE_data % SPAM % EM
        Case ('EO');     Value = SAVE_data % SPAM % EO
        Case ('EP');     Value = SAVE_data % SPAM % EP
        Case ('ES');     Value = SAVE_data % SPAM % ES
        Case ('ET');     Value = SAVE_data % SPAM % ET
        Case ('EOP');    Value = SAVE_data % SPAM % EOP
        Case ('EVAP');   Value = SAVE_data % SPAM % EVAP
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('BIOMAS'); Value = SAVE_data % PLANT % BIOMAS
        Case ('CANHT') ; Value = SAVE_data % PLANT % CANHT
        Case ('CANWH') ; Value = SAVE_data % PLANT % CANWH
        Case ('DXR57') ; Value = SAVE_data % PLANT % DXR57
        Case ('EXCESS'); Value = SAVE_data % PLANT % EXCESS
        Case ('PLTPOP'); Value = SAVE_data % PLANT % PLTPOP
        Case ('RNITP') ; Value = SAVE_data % PLANT % RNITP
        Case ('SLAAD') ; Value = SAVE_data % PLANT % SLAAD
        Case ('XPOD')  ; Value = SAVE_data % PLANT % XPOD
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('MGMT')
        SELECT CASE (VarName)
        Case ('EFFIRR'); Value = SAVE_data % MGMT % EFFIRR
        Case ('TOTIR');  Value = SAVE_data % MGMT % TOTIR
        Case ('TOTEFFIRR');Value=SAVE_data % MGMT % TOTEFFIRR
        Case ('DEPIR');  Value = SAVE_data % MGMT % DEPIR
        Case ('IRRAMT'); Value = SAVE_data % MGMT % IRRAMT
        Case ('FERNIT'); Value = SAVE_data % MGMT % FERNIT
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('WATER')
        SELECT CASE (VarName)
        Case ('DRAIN'); Value = SAVE_data % WATER % DRAIN
        Case ('RUNOFF');Value = SAVE_data % WATER % RUNOFF
        Case ('SNOW');  Value = SAVE_data % WATER % SNOW
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('NITR')
        SELECT CASE (VarName)
        Case ('TNOXD'); Value = SAVE_data % NITR % TNOXD
       Case ('TLCHD'); Value = SAVE_data % NITR % TLeachD
!       Case ('TN2OD'); Value = SAVE_data % NITR % TN2OD
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('ORGC')
        SELECT CASE (VarName)
        Case ('MULCHMASS');Value = SAVE_data % ORGC % MULCHMASS
        Case ('TOMINFOM'); Value = SAVE_data % ORGC % TOMINFOM
        Case ('TOMINSOM'); Value = SAVE_data % ORGC % TOMINSOM
        Case ('TOMINSOM1');Value = SAVE_data % ORGC % TOMINSOM1
        Case ('TOMINSOM2');Value = SAVE_data % ORGC % TOMINSOM2
        Case ('TOMINSOM3');Value = SAVE_data % ORGC % TOMINSOM3
        Case ('TNIMBSOM'); Value = SAVE_data % ORGC % TNIMBSOM
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('SOIL')
        SELECT CASE (VarName)
        Case ('TOMINFOM'); Value = SAVE_data % ORGC % TOMINFOM
        Case ('TOMINSOM'); Value = SAVE_data % ORGC % TOMINSOM
        Case ('TOMINSOM1');Value = SAVE_data % ORGC % TOMINSOM1
        Case ('TOMINSOM2');Value = SAVE_data % ORGC % TOMINSOM2
        Case ('TOMINSOM3');Value = SAVE_data % ORGC % TOMINSOM3
        Case ('TNIMBSOM'); Value = SAVE_data % ORGC % TNIMBSOM
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      CASE ('PDLABETA')
        SELECT CASE(VarName)
        CASE('PDLA'); Value = SAVE_data % PDLABETA % PDLA
        CASE('BETA'); Value = SAVE_data % PDLABETA % BETALS
        CASE DEFAULT; ERR = .TRUE.
        END SELECT
            
      Case DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, " in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value set to zero.'
        CALL WARNING(2,'GET_REAL',MSG)
      ENDIF

      RETURN
      END SUBROUTINE GET_Real

!----------------------------------------------------------------------
      SUBROUTINE PUT_Real(ModuleName, VarName, Value)
!     Stores real variable SAVE_data.  
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78 MSG(2)
      Real Value
      Logical ERR

      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('SPAM')
        SELECT CASE (VarName)
        Case ('AGEFAC'); SAVE_data % SPAM % AGEFAC = Value
        Case ('PG');     SAVE_data % SPAM % PG     = Value
        Case ('CEF');    SAVE_data % SPAM % CEF    = Value
        Case ('CEM');    SAVE_data % SPAM % CEM    = Value
        Case ('CEO');    SAVE_data % SPAM % CEO    = Value
        Case ('CEP');    SAVE_data % SPAM % CEP    = Value
        Case ('CES');    SAVE_data % SPAM % CES    = Value
        Case ('CET');    SAVE_data % SPAM % CET    = Value
        Case ('EF');     SAVE_data % SPAM % EF     = Value
        Case ('EM');     SAVE_data % SPAM % EM     = Value
        Case ('EO');     SAVE_data % SPAM % EO     = Value
        Case ('EP');     SAVE_data % SPAM % EP     = Value
        Case ('ES');     SAVE_data % SPAM % ES     = Value
        Case ('ET');     SAVE_data % SPAM % ET     = Value
        Case ('EOP');    SAVE_data % SPAM % EOP    = Value
        Case ('EVAP');   SAVE_data % SPAM % EVAP   = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('BIOMAS'); SAVE_data % PLANT % BIOMAS = Value
        Case ('CANHT');  SAVE_data % PLANT % CANHT  = Value
        Case ('CANWH');  SAVE_data % PLANT % CANWH  = Value
        Case ('DXR57');  SAVE_data % PLANT % DXR57  = Value
        Case ('EXCESS'); SAVE_data % PLANT % EXCESS = Value
        Case ('PLTPOP'); SAVE_data % PLANT % PLTPOP = Value
        Case ('RNITP');  SAVE_data % PLANT % RNITP  = Value
        Case ('SLAAD');  SAVE_data % PLANT % SLAAD  = Value
        Case ('XPOD');   SAVE_data % PLANT % XPOD   = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('MGMT')
        SELECT CASE (VarName)
        Case ('EFFIRR'); SAVE_data % MGMT % EFFIRR = Value
        Case ('TOTIR');  SAVE_data % MGMT % TOTIR  = Value
        Case ('TOTEFFIRR');SAVE_data%MGMT % TOTEFFIRR=Value
        Case ('DEPIR');  SAVE_data % MGMT % DEPIR  = Value
        Case ('IRRAMT'); SAVE_data % MGMT % IRRAMT = Value
        Case ('FERNIT'); SAVE_data % MGMT % FERNIT = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('WATER')
        SELECT CASE (VarName)
        Case ('DRAIN'); SAVE_data % WATER % DRAIN  = Value
        Case ('RUNOFF');SAVE_data % WATER % RUNOFF = Value
        Case ('SNOW');  SAVE_data % WATER % SNOW   = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('NITR')
        SELECT CASE (VarName)
        Case ('TNOXD'); SAVE_data % NITR % TNOXD = Value
        Case ('TLCHD'); SAVE_data % NITR % TLeachD = Value
!       Case ('TN2OD'); SAVE_data % NITR % TN2OD = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('ORGC')
        SELECT CASE (VarName)
        Case ('MULCHMASS');SAVE_data % ORGC % MULCHMASS = Value
        Case ('TOMINFOM'); SAVE_data % ORGC % TOMINFOM  = Value
        Case ('TOMINSOM'); SAVE_data % ORGC % TOMINSOM  = Value
        Case ('TOMINSOM1');SAVE_data % ORGC % TOMINSOM1 = Value
        Case ('TOMINSOM2');SAVE_data % ORGC % TOMINSOM2 = Value
        Case ('TOMINSOM3');SAVE_data % ORGC % TOMINSOM3 = Value
        Case ('TNIMBSOM'); SAVE_data % ORGC % TNIMBSOM  = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      CASE ('PDLABETA')
        SELECT CASE(VarName)
        CASE('PDLA'); SAVE_data % PDLABETA % PDLA = Value
        CASE('BETA'); SAVE_data % PDLABETA % BETALS = Value
        CASE DEFAULT; ERR = .TRUE.
        END SELECT
            
      Case DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value not saved! Errors may result.'
        CALL WARNING(2,'PUT_REAL',MSG)
      ENDIF

      RETURN
      END SUBROUTINE PUT_Real

!----------------------------------------------------------------------
      SUBROUTINE GET_Real_Array_NL(ModuleName, VarName, Value)
!     Retrieves array of dimension(NL) 
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78 MSG(2)
      REAL, DIMENSION(NL) :: Value
      Logical ERR

      Value = 0.0
      ERR = .FALSE.

      SELECT CASE (ModuleName)

      CASE ('SPAM')
        SELECT CASE (VarName)
          CASE ('UH2O'); ; Value = SAVE_data % SPAM % UH2O
          CASE DEFAULT; ERR = .TRUE.
        END SELECT

        CASE DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value set to zero.'
        CALL WARNING(2,'GET_Real_Array_NL',MSG)
      ENDIF

      RETURN
      END SUBROUTINE GET_Real_Array_NL

!----------------------------------------------------------------------
      SUBROUTINE PUT_Real_Array_NL(ModuleName, VarName, Value)
!     Stores array of dimension NL
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78 MSG(2)
      REAL, DIMENSION(NL) :: Value
      Logical ERR

      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('SPAM')
        SELECT CASE (VarName)
        Case ('UH2O'); SAVE_data % SPAM % UH2O = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value not saved! Errors may result.'
        CALL WARNING(2,'PUT_Real_Array_NL',MSG)
      ENDIF

      RETURN
      END SUBROUTINE PUT_Real_Array_NL

!----------------------------------------------------------------------
      Subroutine GET_Integer(ModuleName, VarName, Value)
!     Retrieves Integer variable as needed
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78  MSG(2)
      Integer Value
      Logical ERR

      Value = 0
      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('NR5');  Value = SAVE_data % PLANT % NR5
        Case ('iSTAGE');  Value = SAVE_data % PLANT % iSTAGE
        Case ('iSTGDOY'); Value = SAVE_data % PLANT % iSTGDOY
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case Default; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value set to zero.'
        CALL WARNING(2,'GET_INTEGER',MSG)
      ENDIF

      RETURN
      END SUBROUTINE GET_Integer

!----------------------------------------------------------------------
      SUBROUTINE PUT_Integer(ModuleName, VarName, Value)
!     Stores Integer variable
      IMPLICIT NONE
      Character*(*) ModuleName, VarName
      Character*78 MSG(2)
      Integer Value
      Logical ERR

      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('NR5');  SAVE_data % PLANT % NR5  = Value
        Case ('iSTAGE');  SAVE_data % PLANT % iSTAGE  = Value
        Case ('iSTGDOY'); SAVE_data % PLANT % iSTGDOY = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value not saved! Errors may result.'
        CALL WARNING(2,'PUT_Integer',MSG)
      ENDIF

      RETURN
      END SUBROUTINE PUT_Integer

!----------------------------------------------------------------------
      Subroutine GET_Char(ModuleName, VarName, Value)
!     Retrieves Integer variable as needed
      IMPLICIT NONE
      Character*(*) ModuleName, VarName, Value
      Character*78  MSG(2)
      Logical ERR

      Value = ' '
      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('WEATHER')
        SELECT CASE (VarName)
        Case ('WSTA');  Value = SAVE_data % WEATHER % WSTAT
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('iSTNAME');  Value = SAVE_data % PLANT % iSTNAME
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case Default; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value set to zero.'
        CALL WARNING(2,'GET_INTEGER',MSG)
      ENDIF

      RETURN
      END SUBROUTINE GET_Char

!----------------------------------------------------------------------
      SUBROUTINE PUT_Char(ModuleName, VarName, Value)
!     Stores Character variable
      IMPLICIT NONE
      Character*(*) ModuleName, VarName, Value
      Character*78 MSG(2)
      Logical ERR

      ERR = .FALSE.

      SELECT CASE (ModuleName)
      Case ('WEATHER')
        SELECT CASE (VarName)
        Case ('WSTA');  SAVE_data % WEATHER % WSTAT  = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case ('PLANT')
        SELECT CASE (VarName)
        Case ('iSTNAME');  SAVE_data % PLANT % iSTNAME = Value
        Case DEFAULT; ERR = .TRUE.
        END SELECT

      Case DEFAULT; ERR = .TRUE.
      END SELECT

      IF (ERR) THEN
        WRITE(MSG(1),'("Error transferring variable: ",A, "in ",A)') 
     &      Trim(VarName), Trim(ModuleName)
        MSG(2) = 'Value not saved! Errors may result.'
        CALL WARNING(2,'PUT_Integer',MSG)
      ENDIF

      RETURN
      END SUBROUTINE PUT_Char

!======================================================================
      END MODULE ModuleData
!======================================================================


!=======================================================================
! OPHEAD.for
! Includes:
! Module HeaderMod contains header data
! OPHEAD generates general header for OVERVIEW.OUT & daily output files
! OPSOIL generates soil and genetics header for OVERVIEW.OUT
! HEADER writes headers for OVERVIEW.OUT and daily output files 
!-----------------------------------------------------------------------
!  Revision history
!  09/25/2007 CHP Created HeaderMod 
!                 Moved OPSOIL to OPHEAD.for
!                 Moved HEADER to OPHEAD.for
!                 Subroutines rewritten to use array of header data, 
!                   rather than saved ASCII file
!=======================================================================
!     Module to generate header info for output files.
      MODULE HeaderMod
      Use ModuleDefs
      Use ModuleData

        TYPE HeaderType
          INTEGER ICOUNT, ShortCount, RUN
          CHARACTER*80 Header(100)   !header for Overview
        END TYPE
        TYPE (HeaderType) Headers

!       ICOUNT = number of lines in long (OVERVIEW) header
!       ShortCount = number of lines in short (DAILY) header
!       RUN = run number associated with this header
!       Header = array of text lines for short and long headers

        CONTAINS

        SUBROUTINE MULTIRUN(RUN, YRPLT)
!       Updates header for multi-year runs
        IMPLICIT NONE
        CHARACTER*3 RMS
        CHARACTER*8 WSTAT
        CHARACTER*11 TEXT
        CHARACTER*80 HEADER2
        INTEGER I, IDYS, IPLT, IPYRS, ISIM, RUN, YRPLT
        INTEGER LenString
        TYPE (ControlType) CONTROL
        CALL GET(CONTROL)

        DO I = 2, Headers%ShortCount
          IF (HEADERS % Header(I)(1:4) .EQ. '*RUN') THEN
!           Update run number
            HEADER2 = HEADERS % Header(I)
            WRITE(HEADERS % Header(I),'(A5,I3,A72)')
     &        HEADER2(1:5), MOD(RUN,1000), HEADER2(9:80)
            HEADERS % RUN = RUN
            EXIT
          ENDIF
        ENDDO

        DO I = 2, Headers%ICOUNT
          IF (HEADERS % Header(I)(1:6) .EQ. ' START') THEN
!           Update simulation start date
            CALL YR_DOY (CONTROL%YRSIM,IPYRS,ISIM)
            CALL NAILUJ (ISIM,IPYRS,RMS,IDYS)
            WRITE(HEADERS%Header(I),400) RMS,IDYS,IPYRS
  400       FORMAT (1X,'STARTING DATE  :',1X,A3,1X,I2,1X,I4)
            EXIT
          ENDIF
        ENDDO

        IF (YRPLT > 0) THEN
          DO I = 2, Headers%ICOUNT
            IF (HEADERS % Header(I)(1:6) .EQ. ' PLANT') THEN
!             Update planting date
              CALL YR_DOY (YRPLT,IPYRS,IPLT)
              CALL NAILUJ (IPLT,IPYRS,RMS,IDYS)
              WRITE(TEXT,'(A3,1X,I2,1X,I4)') RMS,IDYS,IPYRS
              HEADERS%Header(I)(19:29) = TEXT
              HEADERS%Header(I)(30:36) = '       '
              EXIT
            ENDIF
          ENDDO
        ENDIF

        DO I = 2, Headers%ICOUNT
          IF (HEADERS % Header(I)(1:6) .EQ. ' WEATH') THEN
!           Update WEATHER file
            CALL GET("WEATHER", "WSTA", WSTAT)
            IF (LenString(WSTAT) > 0) THEN
             WRITE(HEADERS%HEADER(I),"(1X,'WEATHER',8X,':',1X,A8)")WSTAT
            ENDIF
            EXIT
          ENDIF
        ENDDO

        RETURN
        END SUBROUTINE MULTIRUN

      END MODULE HeaderMod
!=======================================================================


!=======================================================================
C  UTILS, File, G. Hoogenboom, P.W. Wilkens and B. Baer
C  General utility functions
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  01/30/1998 GH  Combine UTILS based on UPCASE, VERIFY, TABEX, and CLEAR
C  07/01/2000 GH  Added SWAP
C=======================================================================

C=======================================================================
C  UPCASE, Function
C
C  Function to return the upper case of a lower case letter.  Otherwise
C  returns the same character
C-----------------------------------------------------------------------
C  Revision history
C
C  05/15/1992 BDB Written
C  05/28/1993 PWW Header revision and minor changes   
C-----------------------------------------------------------------------
C  INPUT  : INCHAR
C
C  LOCAL  :
C
C  OUTPUT : INCHAR
C-----------------------------------------------------------------------
C  Called : INPUT READS IPEXP JULIAN
C
C  Calls  : None
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  INCHAR :
C  CHAVAL :
C=======================================================================

      CHARACTER*1 FUNCTION UPCASE (INCHAR)

      IMPLICIT  NONE

      CHARACTER INCHAR*1
      INTEGER   CHAVAL

      CHAVAL = ICHAR(INCHAR)

      IF ((CHAVAL .LE. 122) .AND. (CHAVAL .GE. 97)) THEN
         UPCASE = CHAR(CHAVAL-32)
       ELSE
         UPCASE = INCHAR
      ENDIF

      END FUNCTION UPCASE

C=======================================================================
C  VERIFY, Subroutine
C
C  I/O utility routine
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2  Modified by
C  3. Header revision and minor changes             P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : LINE VALUE FLAG
C
C  LOCAL  :
C
C  OUTPUT :
C-----------------------------------------------------------------------
C  Called : IPECO IPSOIL IPVAR SECLI  SECROP SEFERT SEFREQ SEHARV SEINIT
C           SEIRR SEPLT  SERES SESOIL SEVAR  SEWTH  IPEXP  SETIME
C
C  Calls  : None
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C=======================================================================

      SUBROUTINE VERIFY (LINE,VALUE,FLAG)

      IMPLICIT    NONE

      CHARACTER*1 DIGITS(13),LINE(80)
      INTEGER     JKNEW,JSIGNR,JTIME,JVALUE,NC,NDECML,I,J,JSIGN
      REAL        TFRAC,VALUE,FLAG

      DATA DIGITS/'0','1','2','3','4','5','6','7','8','9',
     &            '+','-','.'/

      FLAG   = -1.0
      JKNEW  =    0
      JSIGNR =    1
      JTIME  =    0
      JVALUE =    0
      NC     =    0
      NDECML =   -1
      TFRAC  =  0.0
      VALUE  =  0.0
C
C     Check for CR/LF
C
      DO I = 1, 80
         IF (LINE(I) .NE. ' ') GO TO 100
      END DO
      !
      ! Nothing entered .. spaces or ÄÄÙ .. set FLAG to 1.0 and exit
      !
      FLAG = 1.0
      GO TO 1300

  100 NC     = NC + 1
      IF (NC .GT. 80) GO TO 1300
      IF (LINE(NC) .EQ. ' ') GO TO 100
      DO I = 1, 13
         J = I
         IF (LINE(NC) .EQ. DIGITS(I)) GO TO 300
      END DO
      GO TO 1200
  300 IF (J .LE. 10) GO TO 600
      IF (J .EQ. 13) GO TO 500
      JSIGN  = 1
      IF (J .EQ. 12) JSIGN = -1

C-----------------------------------------------------------------------
C***    IF SIGN IS REPEATED
C-----------------------------------------------------------------------

      IF (JKNEW .GT. 0) GO TO 1200
      JKNEW = 1
C-----------------------------------------------------------------------
C
C***    SIGN APPEARS AFTER THE DECIMAL POINT
C
C-----------------------------------------------------------------------
      IF (NDECML) 400,1200,900
  400 JSIGNR = JSIGN
      GO TO 900
  500 JKNEW  = 1
C-----------------------------------------------------------------------
C***    DECIMAL REPEATED
C-----------------------------------------------------------------------

      IF (NDECML .GE. 0) GO TO 1200
      NDECML = 0
      GO TO 900
  600 J = J - 1
      JKNEW = 1
      IF (NDECML) 700,800,900
  700 JVALUE = JVALUE*10 + J
      GO TO 900
  800 JTIME = JTIME + 1
      TFRAC = TFRAC + FLOAT(J)/(10.**JTIME)
  900 VALUE = FLOAT(JSIGNR*JVALUE)+FLOAT(JSIGNR)*TFRAC
 1000 NC    = NC + 1
      IF (NC .GT. 80) GO TO 1300
      IF (LINE(NC) .EQ. ' ') GO TO 1000
      DO I = 1, 13
         J = I
         IF (LINE(NC) .EQ. DIGITS(I)) GO TO 300
      END DO
 1200 FLAG = 1.0

 1300 DO I = 1, 80
         LINE(I) = ' '
      END DO

      END SUBROUTINE VERIFY

C=======================================================================
C  TABEX, Function
C
C  Look up utility routine
C-----------------------------------------------------------------------
      FUNCTION TABEX(VAL,ARG,DUMMY,K)

      IMPLICIT NONE

      INTEGER K,J

      REAL VAL(K),ARG(K),DUMMY,TABEX

           DO 100  J = 2,K
           IF (DUMMY .GT. ARG(J)) GO TO 100
           GO TO 200
  100      CONTINUE
      J = K
  200 TABEX = (DUMMY-ARG(J-1))*(VAL(J)-VAL(J-1))/(ARG(J)-ARG(J-1))+VAL
     &     (J-1)

      END FUNCTION TABEX

C=======================================================================
C  CLEAR, Subroutine
C
C  Clears the screen using ANSI codes, sets color to white on blue
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2. Header revision and minor changes           P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : None
C
C  LOCAL  : ESC
C
C  OUTPUT : None
C-----------------------------------------------------------------------
C  Called : SESOIL SESIM SERES SEPLT SENSDM SENS SEFREQ SEWTH SEFERT SECLI
C           SEVAR SETIME IPVAR IPSOIL WTHMDI OPHARV OPDAY IPEXP SEIRR SEINIT
C           SEHARV SECROP INVAR INPUT
C
C  Calls  : None
C-----------------------------------------------------------------------

C                         DEFINITIONS
C
C  ESC    : ESC key codes
C=======================================================================

      SUBROUTINE CLEAR

      IMPLICIT  NONE

      !CHARACTER ESC
      !ESC = CHAR(27)
C      WRITE  (*,100) ESC
      WRITE  (*,100) 
C 100   FORMAT (1X,A1,'[2J',//////)
100   FORMAT (20/)

      END SUBROUTINE CLEAR

C=======================================================================
C  HOME, Subroutine
C
C  Moves cursor to 0,0 ANSI and clears the screen
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written                                     P.W.W.      2- 9-93
C  2. Header revision and minor changes           P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : None
C
C  LOCAL  : ESC
C
C  OUTPUT : None
C-----------------------------------------------------------------------
C  Called : ERROR AUTPLT AUTHAR INPUT
C
C  Calls  : CLEAR
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  ESC    : ESC key codes
C=======================================================================

      SUBROUTINE HOME

      IMPLICIT  NONE

      CHARACTER ESC

      ESC = CHAR(27)

      WRITE (*,100) ESC
      WRITE (*,200) ESC

100   FORMAT (1X,A1,'[2J')
200   FORMAT (1X,A1,'[0;0H')

      END SUBROUTINE HOME

C=======================================================================
C  CURPOS, Subroutine
C
C  Moves cursor to specified row
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written                                     P.W.W.      2- 9-93
C  2. Header revision and minor changes           P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : None
C
C  LOCAL  : ESC
C
C  OUTPUT : None
C-----------------------------------------------------------------------
C  Called : INTRO
C
C  Calls  : NONE
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  ESC    : ESC key codes
C=======================================================================

      SUBROUTINE CURPOS (LINE)

      IMPLICIT  NONE

      CHARACTER ESC,LINE*2

      ESC = CHAR(27)

      WRITE (*,200) ESC,LINE

200   FORMAT (1X,A1,'[',A2,';0H')

      END SUBROUTINE CURPOS

C=======================================================================
C  CURV, Function
C
C  Function to interpolate between four data points.
C-----------------------------------------------------------------------
C  Revision history
C
C  01/01/1990 JWJ Written
C  01/12/1999 GH  Added to UTILS routine
C  10/08/2004 CHP Changed criteria for XM from .GT. to .GE. 
C-----------------------------------------------------------------------
C  INPUT  :
C
C  LOCAL  :
C
C  OUTPUT : INCHAR
C-----------------------------------------------------------------------
C  Called :
C
C  Calls  : None
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  CTYPE  :
C  CURV   :
C  XB     :
C  XM     :
C=======================================================================
      FUNCTION CURV(CTYPE,XB,X1,X2,XM,X)

      IMPLICIT NONE

      CHARACTER*3 CTYPE
      REAL CURV,XB,X1,X2,XM,X

      CURV = 1.0
      IF (CTYPE .EQ. 'NON' .OR. CTYPE .EQ. 'non') RETURN

C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'LIN' .OR. CTYPE .EQ. 'lin') THEN
        CURV = 0.
        IF(X .GT. XB .AND. X .LT. X1)CURV = (X-XB)/(X1-XB)
        IF(X .GE. X1 .AND. X .LE. X2)CURV = 1.
        IF(X .GT. X2 .AND. X .LT. XM)CURV = 1.0 - (X-X2)/(XM-X2)
        CURV = MAX(CURV,0.0)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------

C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'QDR' .OR. CTYPE .EQ. 'qdr') THEN
        CURV = 0.
        IF(X .GT. XB .AND. X .LT. X1)CURV = 1. -((X1-X)/(X1-XB))**2
        IF(X .GE. X1 .AND. X .LE. X2)CURV = 1.
        IF(X .GT. X2 .AND. X .LT. XM)CURV = 1. - ((X-X2)/(XM-X2))**2
        CURV = MAX(CURV,0.0)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------

C-------------------------------------------------------------------------------
C     Curve type INL is the inverse linear with a minimum for use in photoperiod
C     In this case, XM is the lowest relative rate, X1 and X2 are critical dayl
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'INL' .OR. CTYPE .EQ. 'inl') THEN
        CURV = 1.0
        IF(X .GT. X1 .AND. X .LT. X2)CURV = 1.-(1.-XM)*((X-X1)/(X2-X1))
!        IF(X .GT. X2) CURV = XM
        IF(X .GE. X2) CURV = XM       !CHP per Stu Rymph 10/8/2004
        CURV = MAX(CURV,XM)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------
C     Curve type SHO for use with short day plants.
C     The curve is the inverse linear with a minimum for use in photoperiod
C     In this case, XM is the lowest relative rate, X1 and X2 are critical dayl
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'SHO' .OR. CTYPE .EQ. 'sho') THEN
        IF (X .LE. X1) THEN
           CURV = 1.0
        ELSE IF ((X .GT. X1) .AND. (X .LT. X2)) THEN
           CURV = 1.-(1.-XM)*((X-X1)/(X2-X1))
        ELSE IF (X .GE. X2) THEN
          CURV = XM
        ENDIF
        CURV = MAX(CURV,XM)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------
C     Curve type LON for use with long day plants.
C     The curve is the inverse linear with a minimum for use in photoperiod
C     In this case, XM is the lowest relative rate, X1 and X2 are critical dayl
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'LON' .OR. CTYPE .EQ. 'lon') THEN
        IF (X .LT. X2) THEN
           CURV = XM
        ELSE IF ((X .GE. X2) .AND. (X .LT. X1)) THEN
           CURV = 1.-(1.-XM)*((X1-X)/(X1-X2))
        ELSE
           CURV = 1.0
        ENDIF
        CURV = MAX(CURV,XM)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------
C
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'SIN' .OR. CTYPE .EQ. 'sin') THEN
        CURV = 0.
        IF(X .GT. XB .AND. X .LT. X1)
     &   CURV = 0.5*(1.+COS(2.*22./7.*(X-X1)/(2.*(X1-XB))))
        IF(X .GE. X1 .AND. X .LE. X2)CURV = 1.
        IF(X .GT. X2 .AND. X .LT. XM)
     &   CURV = 0.5*(1.+COS(2.*22./7.*(X2-X)/(2.*(XM-X2))))
        CURV = MAX(CURV,0.0)
        CURV = MIN(CURV,1.0)
      ENDIF
C-------------------------------------------------------------------------------
C-------------------------------------------------------------------------------
C-------------------------------------------------------------------------------
C     Curve type REV - Reversible process - used for cold hardening
C	Rate of cold hardening increases as TMIN decreases from X1 to XB
C	Cold hardening reverses at an increasing rate as TMIN increases from X1 to X2
C     Process at maximum rate at or below XB
C	Rate decreases linearly to 0 at X1
C	Process reverses at a linear rate from X1 to X2
C	XM is the maximum absolute rate
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'REV' .OR. CTYPE .EQ. 'rev') THEN
        CURV = 1.
        IF(X .GT. XB .AND. X .LT. X1)CURV = 1.0-((X-XB)/(X1-XB))
        IF(X .GE. X1 .AND. X .LE. X2)CURV = 0.0-((X-X1)/(X2-X1))
        IF(X .GT. X2 )CURV = -1.0 
        CURV = MAX(CURV,-1.0)
        CURV = MIN(CURV,1.0)
	  CURV = CURV * XM
      ENDIF
C-------------------------------------------------------------------------------
C     Curve type DHD - used for cold dehardening in spring
C	No cold dehardening below XB (rate=0)
C	Rate of cold dehardening increases as TMIN increases from XB to X1
C     Process at maximum rate at or above X1
C	X2 is not used
C	XM is the maximum absolute rate
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'DHD' .OR. CTYPE .EQ. 'dhd') THEN
        CURV = 0.
        IF(X .GT. XB .AND. X .LT. X1)CURV = (X-XB)/(X1-XB)
        IF(X .GE. X1 .AND. X .LE. X2)CURV = 1
        IF(X .GT. X2 )CURV = 1
        CURV = MAX(CURV,0.0)
        CURV = MIN(CURV,1.0)
	  CURV = CURV * XM
      ENDIF

C-------------------------------------------------------------------------------
C     Curve type DRD - used for reducing rates of processes as dormancy advances
C	Multiply rates by this factor to reduce them on short days, 
C	no effect on long days
C	XM is the maximum reduction factor at full dormancy (daylength=XB)
C	Less reduction as daylength gets longer
C     Process at maximum rate at or above X1
C	X2 is not used
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'DRD' .OR. CTYPE .EQ. 'drd') THEN
        CURV = X2
        IF(X .GT. XB .AND. X .LT. X1)
     &	  CURV = X2+(XM-X2)*(X-XB)/(X1-XB)
        IF(X .GE. X1 )CURV = XM
        CURV = MAX(CURV,X2)
        CURV = MIN(CURV,XM)
      ENDIF

C-------------------------------------------------------------------------------
C    Curve type CDD - used for reducing rates of processes as dormancy advances
C	Multiply rates by this factor to reduce them on short days, 
C	Long day effect depends on value of XM
C	X2 is the maximum reduction factor at full dormancy (daylength=XB)
C	Less reduction as daylength gets longer
C    Process at maximum rate at or above X1
C	Curvilinear version of DRD
C-------------------------------------------------------------------------------

      IF(CTYPE .EQ. 'CDD' .OR. CTYPE .EQ. 'cdd') THEN
        CURV = X2
        IF(X .GT. XB .AND. X .LT. X1)
     &	  CURV = XM-((XM-X2)*((X1-X)/(X1-XB))**2)
        IF(X .GE. X1)CURV = XM
        CURV = MAX(CURV,X2)
        CURV = MIN(CURV,XM)
      ENDIF

C-------------------------------------------------------------------------------
C	Curve type EXK - generic exponential function with "k"
C	XB sets the amplitude of the curve (max Y value)
C	X1/XM sets the amount of curvature (k) and shape of the curve (+ or -)
C	X2 shifts the curve left (- X2) or right (+X2) on the X axis 
C	If X1/XM is positive, X2 is the X-intercept 
C-------------------------------------------------------------------------------

      IF(CTYPE .EQ. 'EXK' .OR. CTYPE .EQ. 'exk') THEN

        CURV = XB - EXP(X1*(X-X2)/XM)
      ENDIF
C-------------------------------------------------------------------------------
C-------------------------------------------------------------------------------
C     Curve type VOP - Variable Order Polynomial
C	Polynomial order (quadratic, cubic, etc.) is continuously variable 
C	(not discete steps)
C	XB=T0, the lower temperature where the function scales to 0
C	XM=T0', the upper temperature where the function scales to 0
C	X1=Tref, reference temperature at which the functio scales to 1.0
C	X2=qft, variable that sets the order of the polynomial
C	Set X2=1 the function is quadratic, X2=2 cubic, X2=3 quartic, etc. 
C     X2 does not have to be an integer
C	Function scales to 0 below XB and above XM
C	Minimum CURV value =0.0, maximum can exceed 1.0
C	Can use mft, a multiplier, to set the scale of the function
C	Read mft in from file and apply in main section of code (ex. mft*CURV)
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'VOP' .OR. CTYPE .EQ. 'vop') THEN
        CURV=0.0
	  IF(X .GT. XB .AND. X .LT. XM)
     &	  CURV = (((X-XB)**X2)*(XM-X))/(((X1-XB)**X2)*(XM-X1))
        IF(X .GE. XM ) CURV = 0.0
        CURV = MAX(CURV,0.0)
      ENDIF

C-------------------------------------------------------------------------------
C-------------------------------------------------------------------------------
C     Curve type Q10 - basic Q10 function
C	XB=Tref, reference temperature
C	X1=k, te response at Tref
C	X2= Q10 increase in the response for every 10°K increase in temperature
C	XM is not used
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'Q10' .OR. CTYPE .EQ. 'q10') THEN
	  CURV=X1*(X2**((X-XB)/10))
      ENDIF

C-------------------------------------------------------------------------------
C-------------------------------------------------------------------------------
C     Curve type PWR - basic function raising X to some power with scaling
C	XB=multiplier for main function
C	X1=power to raise X to
C	X2= scaling multiplier to scale reults to a given range
C	XM is not used
C	Added condition for negative values of X - was generating NaN with
C	negative values of X and fractional vlaues of X1 
C	(ex. Temp=-1.8C and X1=1.5905).  Now uses value for 0.0 when X<0.0
C-------------------------------------------------------------------------------
      IF(CTYPE .EQ. 'PWR' .OR. CTYPE .EQ. 'pwr') THEN
		IF (X .LT. 0.0) THEN
		CURV=X2*XB*(0**X1)
		ELSE
		CURV=X2*XB*(X**X1)
		ENDIF
      ENDIF

C-------------------------------------------------------------------------------

      END FUNCTION CURV

C=======================================================================
C  SWAP
C
C  Subroutine to swap values among different variables
C-----------------------------------------------------------------------
C  Revision history
C
C  01/01/90 JWH Written
C  07/01/00 GH  Added to UTILS routine
C-----------------------------------------------------------------------

      SUBROUTINE SWAP(A, B, C, D, E, F)

      IMPLICIT NONE

      INTEGER A, B, C, D, E, F, I

      I = A
      A = B
      B = I
      I = C
      C = D
      D = I
      I = E
      E = F
      F = I
      RETURN
      END SUBROUTINE SWAP
C-----------------------------------------------------------------------


C=======================================================================
C  GETLUN, Subroutine, C. H. Porter
C-----------------------------------------------------------------------
C  Assigns unique output file unit numbers to input and output files
C     based on file variable name.  If valid file variable name is not
C     specified, unit numbers are assigned incrementally starting with
C     unit 500.
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  10/17/2001 CHP Written.
C-----------------------------------------------------------------------
! Called by: IRRIG, OPWBAL, OPGROW, . . . 
! Calls: None
C========================================================================

      SUBROUTINE GETLUN(FileVarName, LUN)

!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      IMPLICIT NONE
      SAVE

      CHARACTER*(*), INTENT(IN) :: FileVarName
      INTEGER, INTENT(OUT) :: LUN

      LOGICAL FEXIST, FIRST, FPRINT(2*MaxFiles)
      INTEGER Counter, Len2, Length, I, OUTLUN, StartLun
      INTEGER Len1, LenString
      CHARACTER*10 ALIAS 
      CHARACTER*16 SaveName(2*MaxFiles), FileName

      TYPE (OutputType) FileData

      DATA OUTLUN /30/    !LUN.LST - list of unit assignments
      DATA FIRST /.TRUE./

      INTEGER       DATE_TIME(8)
!      date_time(1)  The 4-digit year  
!      date_time(2)  The month of the year  
!      date_time(3)  The day of the month  
!      date_time(4)  The time difference with respect to Coordinated Universal Time (UTC) in minutes  
!      date_time(5)  The hour of the day (range 0 to 23) - local time  
!      date_time(6)  The minutes of the hour (range 0 to 59) - local time  
!      date_time(7)  The seconds of the minute (range 0 to 59) - local time  
!      date_time(8)  The milliseconds of the second (range 0 to 999) - local time  

!      CHARACTER*3   MON(12)
!      DATA MonthTxt /'Jan','Feb','Mar','Apr','May','Jun','Jul'
!     &         ,'Aug','Sep','Oct','Nov','Dec'/

!-----------------------------------------------------------------------
!     Get list of output files
      IF (FIRST) THEN
        CALL GET(FileData)
        FPRINT = .FALSE.
        StartLun = MaxFiles + 1
        Counter = 0
        FIRST = .FALSE.

!       On first call to subroutine, open new file to record
!       input and output file information.
        INQUIRE (FILE = 'LUN.LST', EXIST = FEXIST)
        IF (FEXIST) THEN
          OPEN (UNIT = OUTLUN, FILE = 'LUN.LST', STATUS = 'REPLACE',
     &      POSITION = 'APPEND')
        ELSE
          OPEN (UNIT = OUTLUN, FILE = 'LUN.LST', STATUS = 'NEW')
          WRITE(OUTLUN,10)
   10     FORMAT('*Summary of files opened during simulation',
     &        //,'Unit  File',/'Num.  Variable Name')
        ENDIF

        CALL DATE_AND_TIME (VALUES=DATE_TIME)
      
!       Version information stored in ModuleDefs.for
        WRITE (OUTLUN,100) Version, VBranch, MonthTxt(DATE_TIME(2)),
     &    DATE_TIME(3), DATE_TIME(1), DATE_TIME(5), 
     &    DATE_TIME(6), DATE_TIME(7)
  100   FORMAT ("*DSSAT Cropping System Model Ver. ",I1,".",I1,".",I1,
     &    ".",I3.3,1X,A10,4X,
     &    A3," ",I2.2,", ",I4,"; ",I2.2,":",I2.2,":",I2.2,
     &        //," Unit  Filename")
      ENDIF

!-----------------------------------------------------------------------
      LUN = 0
      Length = LenString(FileVarName)

!     Assign input file names:
      SELECT CASE (FileVarName(1:Length))

!     Input Files (Units 8 through 29):
      CASE ('FILEA');   LUN = 8   !observed time series data
      CASE ('FILEC', 'FILEE', 'FINPUT');  LUN = 10          
                  !*.spe, *.eco, miscellaneous input files

      CASE ('FILEW');   LUN = 11  !*.wth - weather files
      CASE ('FILEP');   LUN = 12  !*.pst - pest files
      CASE ('FILESS');  LUN = 13  !RESCH???.SDA (was SOILN980.SOL)
      CASE ('BATCH');   LUN = 14  !Batch run input file
      CASE ('ERRORX');  LUN = 15  !Model.err
      CASE ('FILETL');  LUN = 16  !TILOP???.SDA
      CASE ('PFILE');   LUN = 17  !Phosphorus input files
      CASE ('FILEIO');  LUN = 21  !temp. input file; dssat45.inp
      CASE ('DTACDE');  LUN = 22  !DATA.CDE
      CASE ('FILETMP'); LUN = 23  !Tony Hunt temp file
      CASE ('SIMCNTL'); LUN = 24  !Simulation Control file
      CASE ('DSPRO');   LUN = 25  !DSSATPRO file

!     RESERVE FOR ORYZA
      CASE ('ORYZA1');  LUN = 26  !ORYZA
      CASE ('ORYZA1A'); LUN = 27  !Prevent ORYZA1+1 from being used

!     Currently 30 is the highest number for reserved logical units
!     Change value in subroutine OUTFILES if necessary

!     Reserve unit #30 for OUTLUN
      CASE ('LUN.LST'); LUN = OUTLUN  !List of output files
      CASE DEFAULT;     LUN = 0
      END SELECT

      IF (LUN == 0) THEN
!       Assign output file names:
        DO I = 1, FileData % NumFiles
          FileName = FileData % FileName(I)
          Len1 = LenString(FileName)
          ALIAS = FileData % ALIAS(I)
          Len2 = LenString(ALIAS)
          IF (FileVarName(1:Length) == FileName(1:Len1) .OR.
     &        FileVarName(1:Length) == ALIAS(1:Len2)) THEN
            LUN = FileData % LUN(I)
            EXIT
          ENDIF
        ENDDO
      ENDIF

!     Files not covered above will be assigned numbers
!     incrementally starting with unit number 110.
      IF (LUN == 0) THEN
        !First check to see if a unit number has already been
        !assigned to this FileVarName.  If so, assign same LUN.
        DO I = StartLun, StartLun + Counter
          FileName = SaveName(I)
          Len1 = LenString(FileName)
          IF (FileVarName(1:Length) == FileName(1:Len1)) THEN
            LUN = I
            EXIT 
          ENDIF
        ENDDO

        !Assign a unique unit number to this FileVarName
        IF (I .GT. StartLun + Counter) THEN
          LUN = StartLun + Counter
          Counter = Counter + 1
          SaveName(LUN) = FileVarName
        ENDIF
      ENDIF

!     Print to 'LUN.LST' file each file assigned a unit number
!     (only print the first time a unit is assigned)
!       OUTPUT.LST - ICASA format headers, etc.
!     Save FileVarName in case it is used again.
      IF (.NOT. FPRINT(LUN)) THEN
        INQUIRE (UNIT = OUTLUN, OPENED = FEXIST)
        IF (.NOT. FEXIST) THEN
          OPEN (UNIT = OUTLUN, FILE = 'LUN.LST', STATUS = 'OLD',
     &      POSITION = 'APPEND')
        ENDIF
        WRITE(OUTLUN,'(I5,2X,A)') LUN, FileVarName
        FPRINT(LUN) = .TRUE.
        SaveName(LUN) = FileVarName
      ENDIF

!      CLOSE(OUTLUN)

      RETURN
      END SUBROUTINE GETLUN
C=======================================================================


!=======================================================================
!  HEADER, Subroutine, C. H. Porter
!-----------------------------------------------------------------------
!  Writes simulation header info to file.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!========================================================================
!  09/25/2007 CHP  Moved HEADER to OPHEAD.for
!=======================================================================


C=======================================================================
C  GET_CROPD, Subroutine, C. H. Porter
C-----------------------------------------------------------------------
C  Assigns text description of crop based on 2 letter crop code.

C-----------------------------------------------------------------------
C  REVISION HISTORY
C  02/19/2002 CHP Written.
C  08/22/2003 CHP revised to read crop descriptions from DETAIL.CDE
C  11/23/2004 CHP Increased length of PATHX (path for executable) to 120.
C========================================================================

      SUBROUTINE GET_CROPD(CROP, CROPD)
C-----------------------------------------------------------------------
      IMPLICIT NONE
      CHARACTER*2  CROP
      CHARACTER*6  ERRKEY
!      CHARACTER*10 FILECDE
      CHARACTER*16 CROPD
      CHARACTER*78 MSG(3)
      PARAMETER (ERRKEY = 'CROPD')

C-----------------------------------------------------------------------
      CROPD = '          '

      CALL READ_DETAIL(2, 16, CROP, "*Crop and Weed Species", CROPD)

      IF (CROPD .EQ. '          ') THEN
        WRITE(MSG(1),11) CROP , "DETAIL.CDE"
   11   FORMAT('Crop code ',A2, ' could not be found in file: ',A)
        CALL WARNING(1, ERRKEY, MSG)
      ENDIF

!Previous code:
!      SELECT CASE (CROP)
!      CASE ('BA'); CROPD = 'BARLEY    '
!      CASE ('BN'); CROPD = 'DRY BEAN  '
!      CASE ('BR'); CROPD = 'BRACHIARIA'
!      CASE ('C3'); CROPD = 'C3-CROPS  '
!      CASE ('C4'); CROPD = 'C4-CROPS  '
!      CASE ('CB'); CROPD = 'CABBAGE   '
!      CASE ('CS'); CROPD = 'CASSAVA   '
!      CASE ('CH'); CROPD = 'CHICKPEA  '
!      CASE ('CO'); CROPD = 'COTTON    '
!      CASE ('CP'); CROPD = 'COWPEA    '
!      CASE ('CT'); CROPD = 'CITRUS    '
!      CASE ('FA'); CROPD = 'FALLOW    '
!      CASE ('FB'); CROPD = 'FABA BEAN '
!      CASE ('G0'); CROPD = 'BAHIA     '
!      CASE ('G1'); CROPD = 'GRASS-1   '
!      CASE ('G2'); CROPD = 'GRASS-2   '
!      CASE ('G3'); CROPD = 'GRASS-3   '
!      CASE ('G4'); CROPD = 'GRASS-4   '
!      CASE ('G5'); CROPD = 'GRASS-5   '
!      CASE ('G6'); CROPD = 'GRASS-6   '
!      CASE ('G7'); CROPD = 'GRASS-7   '
!      CASE ('G8'); CROPD = 'GRASS-8   '
!      CASE ('MZ'); CROPD = 'MAIZE     '
!      CASE ('ML'); CROPD = 'MILLET    '
!      CASE ('PE'); CROPD = 'PEA       '
!      CASE ('PI'); CROPD = 'PINEAPPLE '
!      CASE ('PN'); CROPD = 'PEANUT    '
!      CASE ('PP'); CROPD = 'PIGEONPEA '
!      CASE ('PR'); CROPD = 'PEPPER    '
!      CASE ('PT'); CROPD = 'POTATO    '
!      CASE ('RI'); CROPD = 'RICE      '
!      CASE ('SB'); CROPD = 'SOYBEAN   '
!      CASE ('SC'); CROPD = 'SUGARCANE '
!      CASE ('SG'); CROPD = 'SORGHUM   '
!      CASE ('SU'); CROPD = 'SUNFLOWER '
!      CASE ('TM'); CROPD = 'TOMATO    '
!      CASE ('TN'); CROPD = 'TANIER    '
!      CASE ('TR'); CROPD = 'TARO      '
!      CASE ('VB'); CROPD = 'VELVETBEAN'
!      CASE ('WH'); CROPD = 'WHEAT     '
!      END SELECT

      RETURN
      END SUBROUTINE GET_CROPD
C=======================================================================



C=======================================================================
C  INTERPOLATE, Subroutine
C
C  Interpolates and coagulates across/between layers
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2  Modified by
C  3. Header revision and minor changes             P. Wilkens  2-8-93
C-----------------------------------------------------------------------
C  INPUT  : NOUTDM,DAP,YRDOY,PCINPD,PG,GROWTH,MAINR,GRWRES,CADLF,CADST,
C           CMINEA,THUMC,CUMRES,TOTWT,RHOL,RHOS,PGNOON,PCINPN
C
C  LOCAL  :
C
C  OUTPUT :
C-----------------------------------------------------------------------
C  Called : OPDAY
C
C  Calls  :
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  HDLAY  :
C=======================================================================

      REAL FUNCTION INTERPOLATE (ARRVAR, DEPTH, DS)
      USE MODULEDEFS
      IMPLICIT  NONE

      REAL      ARRVAR(NL),X(NL),DS(NL), DEPTH,TOTALVAL,TOTDEPTH,DIFF
      INTEGER   I

   !   DATA X   /5.,10.,15.,15.,15.,30.,30.,30.,30.,30./
      X = DS

      IF (DEPTH .LE. X(1)) THEN
         !
         ! Depth is <= to 5 cm
         !
         INTERPOLATE = ARRVAR(1)
         RETURN
      ENDIF

      TOTDEPTH = 0.0
      TOTALVAL = 0.0

      DO I = 1, 10
         IF (TOTDEPTH + X(I) .LE. DEPTH) THEN
           !
           ! We have not yet reached the queried depth
           !
           TOTDEPTH = TOTDEPTH + X(I)
           TOTALVAL = TOTALVAL + ARRVAR(I)*X(I)
!           IF (TOTDEPTH .EQ. DEPTH) THEN
           IF (ABS(TOTDEPTH - DEPTH) .LT. 0.0005) THEN
              INTERPOLATE = TOTALVAL / TOTDEPTH
              EXIT
           ENDIF
         ELSE
           !
           ! Ended up in the middle of a layer .. mix it
           !
           DIFF     = DEPTH - TOTDEPTH
           TOTDEPTH = DEPTH
           TOTALVAL = TOTALVAL + ARRVAR(I)*DIFF
           INTERPOLATE = TOTALVAL / TOTDEPTH
           EXIT
         ENDIF
      END DO

      END FUNCTION INTERPOLATE
C=======================================================================

C=======================================================================
C  OPCLEAR, Subroutine C.H. Porter
C  Delete *.OUT files 
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  07-22-2002 CHP Written.
C  03/07/2005 CHP Read previous output file names from OUTPUT.LST 
C                 rather than delete all *.OUT files.
!  11/02/2006 CHP If OUTPUT.CDE not available, save *.OUT files in temp
!                 directory.
C=======================================================================
      SUBROUTINE OPCLEAR
      USE ModuleDefs
!cDEC$ IF (COMPILER == 0) 
!        USE DFPORT
!cDEC$ ELSE
C        USE IFPORT
!cDEC$ ENDIF
      IMPLICIT NONE

      Type (OutputType) FileData
      CHARACTER*16, DIMENSION(MaxFiles) :: FileName
      CHARACTER*78 MSG(2)
      INTEGER ERR, I, LUN, LUNTMP, SYS, SYSTEM
      LOGICAL FEXIST

!     Read in list of possible output file names
      CALL OUTFILES(FileData)
      FileName = FileData % FileName

      IF (FileData % NumFiles > 0) THEN
!       File data was read from OUTPUT.CDE
!       Find available logical unit for temporary file open commands.
        CALL GETLUN("OPCLR", LUN)

!       Loop thru files, if it exists, delete it.
        DO I = 1, FileData % NumFiles
          OPEN (FILE=trim(FileName(I)),UNIT=LUN,STATUS='OLD',IOSTAT=ERR)
          IF (ERR == 0) CLOSE(LUN,STATUS='DELETE',ERR=50)
  50    CONTINUE
        ENDDO

      ELSE
!       File data was not succesfully read from OUTPUT.CDE
!       Copy *.OUT files to *.BAK files, then delete

        INQUIRE (FILE="HEADER.OUT", EXIST=FEXIST)
        IF (FEXIST) THEN

!         Open temporary batch file
          CALL GETLUN("OUTBAT", LUNTMP)
          OPEN (LUNTMP, FILE="TEMP.BAT", STATUS = 'REPLACE')
          WRITE(LUNTMP,'(A)') "ECHO OFF"
          WRITE(LUNTMP,'(A)') "COPY /Y *.OUT *.BAK"
          WRITE(LUNTMP,'(A)') "ERASE *.OUT"
          CLOSE (LUNTMP)

          SYS = SYSTEM("TEMP.BAT >TEMP.BAK")
          
!         Delete TEMP.BAT file
          OPEN (LUNTMP, FILE = "TEMP.BAT", STATUS = 'UNKNOWN')
          CLOSE (LUNTMP, STATUS = 'DELETE')
  
!         Delete TEMP.BAK file
          OPEN (LUNTMP, FILE = "TEMP.BAK", STATUS = 'UNKNOWN')
          CLOSE (LUNTMP, STATUS = 'DELETE')

!         Output.CDE file is missing
          WRITE(MSG(1),'(A,A)') "OUTPUT.CDE file not found."
          MSG(2)=
     &          "Previous *.OUT files will be saved as *.BAK files."
          CALL WARNING(2,'CSM',MSG)
        ENDIF
      ENDIF

      RETURN
      END SUBROUTINE OPCLEAR
C=======================================================================


C=======================================================================
C  OPNAMES, Subroutine C.H. Porter
C  Use alternate output filenames if FNAME is set.
C  File commands are written to a DOS batch file so the screen output 
C    can be controlled.
C  To expand the list of files: Increase MaxFiles, Add filename and 
C    character code to list.
C  Write names of output files to OUTPUT.LST file
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  07/22/2002 CHP Written.
!  01/11/2005 CHP Added a few more output files to the list
!  07/27/2006 CHP Get filenames from FileData, rather than hardwire
C=======================================================================
      SUBROUTINE OPNAMES(FNAME)
      USE ModuleDefs
      USE ModuleData
!!!!cDEC$ IF (COMPILER == 0) 
!        USE DFPORT
!!!!cDEC$ ELSE
C        USE IFPORT
!!!!cDEC$ ENDIF
      IMPLICIT NONE

      SAVE
      INTEGER i, COUNT, LUNLST, LUNTMP, SYS, SYSTEM
      INTEGER FNUM

      CHARACTER*8  FNAME
      CHARACTER*12  TempName
      CHARACTER*16, DIMENSION(MaxFiles) :: FileName
      CHARACTER*2,  DIMENSION(MaxFiles) :: OPCODE
      CHARACTER*50, DIMENSION(MaxFiles) :: Comment
      CHARACTER*80 BatchCommand

      LOGICAL FOPEN, FEXIST

      TYPE (SwitchType) ISWITCH

      Type (OutputType) FileData

!-----------------------------------------------------------------------
!     If list of names is absent, can't rename files...
      CALL GET(FileData)
      FileName = FileData % FileName
      OPCODE   = FileData % OPCODE
      Comment  = FileData % Description

      IF (FileData % NumFiles == 0) RETURN

!-----------------------------------------------------------------------
!     If FNAME = 'OVERVIEW', then keep default names
!     If FNAME not equal to 'OVERVIEW': 
!       Assign output file names based on FNAME.

!     The following files will be copied to a file of the form 
!     FNAME.Occ, 
!     where FNAME is read from FILEIO and sent from main program, 
!           cc  = Alternate filename character code (see below).

!-----------------------------------------------------------------------
!     Open OUTPUT.LST file - list of new output files
      CALL GETLUN("OUTO", LUNLST)
      OPEN (LUNLST, FILE="OUTPUT.LST", STATUS = 'REPLACE')

      WRITE(LUNLST,10)
   10 FORMAT('*Output file list',
     &    //,'@FILENAME          DESCRIPTION')

      IF (FNAME .EQ. 'OVERVIEW') THEN
!       Keep filenames, just write to OUTPUT.LST file
        DO i = 1, FileData % NumFiles
          !Check if FileName(i) exists, if not, go on to the next file
          INQUIRE (FILE = Trim(FileName(i)), EXIST = FEXIST)
          IF (.NOT. FEXIST) CYCLE   
          WRITE(LUNLST,'(A16,3X,A50)') FileName(i), Comment(i)
        Enddo

      ELSE
!       Re-name output files based on FNAME
!       Open temporary batch file
        CALL GETLUN("OUTBAT", LUNTMP)
        OPEN (LUNTMP, FILE="TEMP.BAT", STATUS = 'REPLACE')
        COUNT = 0

        DO i = 1, FileData % NumFiles
          !Check if FileName(i) exists, if not, go on to the next file
          INQUIRE (FILE = Trim(FileName(i)), EXIST = FEXIST)
          IF (.NOT. FEXIST) CYCLE   

          !Determine new file name and store as TempName
          IF (OPCODE(i) /= '  ') THEN
            TempName = FNAME // '.' // 'O' // OPCODE(i) 
            WRITE(LUNLST,'(A12,5X,A50)') TempName, Comment(i)

            !Check if TempName exists, if so, delete it.
            INQUIRE (FILE = TempName, EXIST = FEXIST)
            IF (FEXIST) THEN   
              BatchCommand = 'ERASE ' // TempName
              WRITE(LUNTMP, '(A50)') BatchCommand
              COUNT = COUNT + 1
            ENDIF

            !Copy from default filename into new filename 
            BatchCommand = 'COPY ' // FileName(i) // ' '  
     &                             // TempName // ' /Y'
            WRITE(LUNTMP,'(A50)') BatchCommand

            !Delete old file
            BatchCommand = 'ERASE ' // FileName(i)
            WRITE(LUNTMP,'(A50)') BatchCommand
            WRITE(LUNTMP, '(" ")')
            COUNT = COUNT + 2

            !If file was left open, close it now.
            INQUIRE(FILE=FILENAME(I), OPENED=FOPEN)
            IF (FOPEN) THEN
              INQUIRE(FILE=FILENAME(I), NUMBER=FNUM)
              CLOSE(FNUM)
            ENDIF

          ELSE
!           Don't rename
            !Check if FileName(i) exists, if not, go on to the next file
            INQUIRE (FILE = Trim(FileName(i)), EXIST = FEXIST)
            IF (FEXIST) THEN
              WRITE(LUNLST,'(A16,3X,A50)') FileName(i), Comment(i)
            ENDIF
          ENDIF 
        Enddo

        CLOSE (LUNTMP)

        IF (COUNT > 0) THEN
!         Run batch file - direct output to TEMP.BAK file
          BatchCommand = "TEMP.BAT >TEMP.BAK"
          SYS = SYSTEM(BatchCommand)
  
!         Delete TEMP.BAT file
          OPEN (LUNTMP, FILE = "TEMP.BAT", STATUS = 'UNKNOWN')
          CLOSE (LUNTMP, STATUS = 'DELETE')
  
!         Delete TEMP.BAK file
          OPEN (LUNTMP, FILE = "TEMP.BAK", STATUS = 'UNKNOWN')
          CLOSE (LUNTMP, STATUS = 'DELETE')
        ELSE
!         Close empty batch file
          CLOSE (LUNTMP, STATUS = 'DELETE')
        ENDIF

      ENDIF

      CALL GET (ISWITCH)
      IF (INDEX('0N',ISWITCH % IDETL) < 1) THEN
        CLOSE (LUNLST)
      ELSE
        CLOSE (LUNLST, STATUS = 'DELETE')
      ENDIF

      RETURN
      END SUBROUTINE OPNAMES
C=======================================================================

C=======================================================================
C  ALIN, Function
C
C  Linear intepolation routine
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2. Header revision and minor changes           P.W.W.      8-7-93
C-----------------------------------------------------------------------
C  INPUT  : TABX,TABY,N
C
C  LOCAL  : I
C
C  OUTPUT : XVAL
C-----------------------------------------------------------------------
C  Called : NFACTO
C
C  Calls  : None
C=======================================================================

      REAL FUNCTION ALIN (TABX,TABY,N,XVAL)

      IMPLICIT  NONE

      INTEGER   N,I
      REAL      TABX,TABY,XVAL

      DIMENSION TABX(N),TABY(N)

      IF (XVAL .LE. TABX(1)) THEN
         ALIN = TABY(1)
         RETURN
      ENDIF
      IF (XVAL .GE. TABX(N)) THEN
         ALIN = TABY(N)
         RETURN
      ENDIF

      DO I = 2, N
        IF (XVAL .LE. TABX(I)) EXIT
      END DO

      ALIN = (XVAL-TABX(I-1))*(TABY(I)-TABY(I-1))/
     &            (TABX(I)-TABX(I-1))+TABY(I-1)

      RETURN    
      END FUNCTION ALIN 
C=======================================================================


!=======================================================================
!  OUTFILES, Subroutine C.H. Porter
!  Reads available output files from OUTPUT.CDE
!  08/16/2005 CHP These alternate names must be coordinated with the 
!                 DSSAT shell. Do not change arbitrarily.
!  07/27/2006 CHP Read list of files from OUTPUT.CDE rather than
!                 hardwire.  Assign unit numbers here.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  03/09/2005 CHP Written.
!  01/11/2007 CHP Changed GETPUT calls to GET and PUT
!=======================================================================
      SUBROUTINE OUTFILES(FileData)
      USE ModuleDefs 
      USE ModuleData
      IMPLICIT NONE
      SAVE

      CHARACTER*10 FILECDE
      CHARACTER*80 CHARTEST
      CHARACTER*120 DATAX, PATHX

      INTEGER ERR, I, ISECT, LNUM, LUN
      LOGICAL FEXIST      !EOF, 
      TYPE (OutputType) FileData

      DATA FILECDE /'OUTPUT.CDE'/

!-----------------------------------------------------------------------
!     Initialize
      FileData % FileName    = ''
      FileData % OpCode      = ''
      FileData % Description = ''
      FileData % ALIAS       = ''
      FileData % LUN         = 0
      FileData % NumFiles    = 0

!     Does file exist in data directory?
      DATAX = FILECDE
      INQUIRE (FILE = DATAX, EXIST = FEXIST)

      IF (.NOT. FEXIST) THEN
!       File does not exist in data directory, check directory
!         with executable.
        CALL GETARG(0,PATHX)
!        call path_adj(pathx)
        call get_dir(pathx,datax)
        datax = trim(datax)//filecde
!        IPX = LEN_TRIM(PATHX)
!        DATAX = PATHX(1:(IPX-12)) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF        

      IF (.NOT. FEXIST) THEN
!       Last, check for file in C:\DSSAT45 directory
        DATAX = trim(STDPATH) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF

      IF (FEXIST) THEN
        LUN = 22
        OPEN (LUN, FILE=DATAX, STATUS = 'OLD', IOSTAT=ERR)
        IF (ERR /= 0) THEN
          FEXIST = .FALSE.
        ELSE
          I = 0
          DO WHILE (I <= MaxFiles)   !.NOT. EOF(LUN) .AND. 
            CALL IGNORE(LUN,LNUM,ISECT,CHARTEST)
            IF (ISECT == 0) EXIT
            IF (ISECT /= 1) CYCLE
            I = I + 1
            READ (CHARTEST,'(A16,A2,1X,A50,1X,A10)',IOSTAT=ERR)
     &        FileData % FileName(I), 
     &        FileData % OpCode(I), 
     &        FileData % Description(I), 
     &        FileData % ALIAS(I)
            FileData % LUN(I) = I + 30
          ENDDO
          FileData % NumFiles = I
          CLOSE(LUN)
        ENDIF
      ENDIF

      CALL PUT(FileData)

      RETURN
      END SUBROUTINE OUTFILES
C=======================================================================

!=======================================================================
!  LenString, Function
!
!  Function to return the length of a character string, excluding 
!     trailing blanks.  This is the same as the fortran function
!     LEN.  When the string is read from an ASCII file, the LEN
!     function does not recognize the blanks as blanks.  Same for 
!     functions ADJUSTL, ADJUSTR, LEN_TRIM.
!-----------------------------------------------------------------------
!  Revision history
!
!  10/24/2005 CHP Written
!=======================================================================

      INTEGER FUNCTION LenString (STRING)

      IMPLICIT  NONE

      CHARACTER(len=*) STRING
      CHARACTER(len=1) CHAR
      INTEGER   I, Length, CHARVAL
      
      LenString = 0

      Length = LEN(STRING)
      DO I = Length, 1, -1
        CHAR = STRING(I:I)
        CHARVAL = ICHAR(CHAR)
        IF (CHARVAL < 33 .OR. CHARVAL > 126) THEN
          CYCLE
        ELSE
          LenString = I
          EXIT
        ENDIF
      ENDDO

      RETURN
      END FUNCTION LenString

!=======================================================================


!=======================================================================
!  StartString, Function
!
!  Function to return the location of the first non-blank character
!     in a string.  Works with strings read from ASCII file, where 
!     blanks may not be recognized.
!-----------------------------------------------------------------------
!  Revision history
!
!  11/30/2007 CHP Written
!=======================================================================

      INTEGER FUNCTION StartString (STRING)

      IMPLICIT  NONE

      CHARACTER(len=*) STRING
      CHARACTER(len=1) CHAR
      INTEGER   CHARVAL, I, LenString
      
      StartString = 0
      DO I = 1, LenString (STRING)
        CHAR = STRING(I:I)
        CHARVAL = ICHAR(CHAR)
        IF (CHARVAL < 33 .OR. CHARVAL > 126) THEN
          CYCLE
        ELSE
          StartString = I
          EXIT
        ENDIF
      ENDDO
      RETURN
      END FUNCTION StartString

!=======================================================================

!=======================================================================
!  Join_Trim, Subroutine
!
!  Subroutine concatenates two strings and ignores leading or trailing 
!     blanks.
!-----------------------------------------------------------------------
!  Revision history
!
!  11/30/2007 CHP Written
!=======================================================================

      Subroutine Join_Trim (STRING1, STRING2, JoinString)

      IMPLICIT NONE

      CHARACTER(len=*) STRING1, STRING2, JoinString
      INTEGER EndLen1, EndLen2, LenString
      INTEGER StartLen1, StartLen2, StartString
      
      EndLen1 = LenString(STRING1)
      IF (EndLen1 > 0) THEN
        StartLen1 = StartString(STRING1)
      ENDIF
      
      EndLen2 = LenString(STRING2)
      IF (EndLen2 > 0) THEN
        StartLen2 = StartString(STRING2)
      ENDIF

      IF (EndLen1 * EndLen2 > 0) THEN
        JoinString = STRING1(StartLen1:EndLen1) // 
     &               STRING2(StartLen2:EndLen2)
      ELSEIF (EndLen1 > 0) THEN
        JoinString = STRING1(StartLen1:EndLen1)
      ELSEIF (EndLen2 > 0) THEN
        JoinString = STRING2(StartLen2:EndLen2)
      ELSE
        JoinString = ""
      ENDIF

      RETURN
      END Subroutine Join_Trim

!=======================================================================



!!=======================================================================
!!  path_adj, Subroutine
!!
!!  Subroutine adjusts path for indications of relative path such as up one 
!!    directory '..'
!!-----------------------------------------------------------------------
!!  Revision history
!!
!!  3/26/2014 PDA Written
!!=======================================================================
!
!      subroutine path_adj(path)
!
!        use moduledefs
!
!        implicit none
!
!        character(len=*),intent(inout) :: path
!
!        integer           :: p1,p2,p3
!
!          p1=index(path,'.')
!          p2=index(path,'..')
!
!          if(p1>1)then
!             p3=len_trim(path)
!             p1=index(path(1:p1),slash,back=.true.)
!             if(p2>0)then
!                p2=p2+3
!                p1=index(path(1:(p1-1)),slash,back=.true.)
!             else
!                p2 = p1 + 2
!             end if
!                path = path(1:p1)//path(p2:p3)
!          end if
!
!      end subroutine path_adj
!
!  3/26/2014 PDA Written
!=======================================================================

      subroutine path_adj(path)

        use moduledefs

        implicit none

        character(len=*),intent(inout) :: path

        integer           :: p1,p2,p3
        
           p1 = index(path,slash//'..'//slash)

           do while(p1>1)
              p2 = p1 + 4
              p1=index(path(1:(p1-1)),slash,back=.true.)
              p3=len_trim(path)
              if(p3<p1+4)then
                 path = path(1:p1)
              else
                 path = path(1:p1)//path(p2:p3)
              end if
              p1 = index(path,slash//'..'//slash)
           end do

      end subroutine path_adj

!=======================================================================
!  skipspc, Subroutine
!
!  Subroutine to skip spaces in a string
!-----------------------------------------------------------------------
!  Revision history
!
!  3/26/2014 PDA Added to CSM
!=======================================================================

      SUBROUTINE SKIPSPC(STRING,POS,DIRECTION)
    
        IMPLICIT NONE

        INTEGER         :: POS

        CHARACTER(LEN=*):: STRING
        CHARACTER(LEN=*):: DIRECTION
        
      
        SELECT CASE(DIRECTION)
            CASE('R','r')
                DO WHILE(STRING(POS:POS)==' ')
                    IF (POS < LEN(STRING)) THEN
                        POS = POS + 1
                    ELSE
                        EXIT
                    ENDIF
                ENDDO
            CASE('L','l')
                DO WHILE(STRING(POS:POS)==' ')
                    IF (POS > 1 ) THEN
                        POS = POS - 1
                    ELSE
                        EXIT
                    ENDIF
                ENDDO
        END SELECT
        
      END SUBROUTINE SKIPSPC


!=======================================================================
!  get_next_string, Subroutine
!
!  Subroutine to get next non-space substring from a larger string
!-----------------------------------------------------------------------
!  Revision history
!
!  3/26/2014 PDA Added to CSM
!=======================================================================

      subroutine get_next_string(full_string,start,next_string)

        implicit none

        character(len=*),intent(in)  :: full_string
        character(len=*),intent(out) :: next_string
        integer,intent(in)           :: start
        integer                      :: pos

        pos = start + index(full_string(start:len(full_string)),' ') - 1

        call skipspc(full_string,pos,'R')

        read(full_string(pos:len(full_string)),'(a)') next_string

      end subroutine get_next_string

!=======================================================================
!  get_dir, Subroutine
!
!  Subroutine to strip file name from full path and return only the directory
!-----------------------------------------------------------------------
!  Revision history
!
!  3/26/2014 PDA Written
!=======================================================================

      subroutine get_dir(full_path,only_dir)

        use moduledefs

        implicit none

        character(len=*),intent(in)  :: full_path
        character(len=*),intent(out) :: only_dir
        integer pos

        pos = index(full_path,slash,back=.true.)

        only_dir = full_path(1:pos)        

      end subroutine get_dir



C=======================================================================
C  FIND, Subroutine, J.W.Jones, 01/03/91
C  Finds appropriate SECTION in a file of logical unit number LUNUM by
C  searching for a 6-character NAME at beginning of each line.
C-----------------------------------------------------------------------
C  INPUT : LUNUM  - logical unit number of the file to read
C          NAME  - 6-character variable name of section to find
C  OUTPUT: LNUM  - Line number of the file currently being read
C          FOUND - Indicator of completion of find routine
C                    0 - End-of-file encountered i.e. name not found
C                    1 - NAME was found
C  LOCAL :
C  IFILE : LUNUM
C  NOTES : Modified N.B. Pickering, 08/27/91
C=======================================================================

      SUBROUTINE FIND(LUNUM,NAME,LNUM,FOUND)

      IMPLICIT NONE
      INTEGER FOUND,I,LNUM,LUNUM
      CHARACTER SECTION*6,NAME*6,UPCASE*1
C
C     Initialization.
C
      FOUND = 0
      LNUM  = 1
      DO I = 1, LEN(NAME)
         NAME(I:I) = UPCASE(NAME(I:I))
      END DO
C
C     Loop to read through data file.
C
   10 IF (.TRUE.) THEN
         READ(LUNUM,'(A)',END=20) SECTION
         DO I = 1,LEN(SECTION)
            SECTION(I:I) = UPCASE(SECTION(I:I))
         END DO
C
C        String found, set FOUND to 1, and exit loop.
C
         IF (NAME .EQ. SECTION) then
            FOUND = 1
            GOTO 20
C
C           String not found, set FOUND to 0.
C
          ELSE
            FOUND = 0
         ENDIF

         LNUM = LNUM + 1
         GOTO 10
      ENDIF

   20 RETURN
      END

C=======================================================================
C  FIND2, Subroutine, J.W.Jones, 01/03/91
C  Finds appropriate SECTION in a file of logical unit number LUNUM by
C  searching for a *-character NAME at beginning of each line.
!  Same as FIND, but no UPCASE function and NAME can be up 
!    to 50 characters long
C-----------------------------------------------------------------------
C  INPUT : LUNUM  - logical unit number of the file to read
C          NAME  - *-character variable name of section to find
C  OUTPUT: LNUM  - Line number of the file currently being read
C          FOUND - Indicator of completion of find routine
C                    0 - End-of-file encountered i.e. name not found
C                    1 - NAME was found
C  LOCAL :
C  IFILE : LUNUM
C  NOTES : Modified N.B. Pickering, 08/27/91
C=======================================================================

      SUBROUTINE FIND2(LUNUM,NAME,LNUM,FOUND)

      IMPLICIT NONE
      INTEGER FOUND,LNUM,LUNUM, LENGTH
      CHARACTER SECTION*50
      CHARACTER NAME*(*)
C
C     Initialization.
C
      FOUND = 0
      LNUM  = 1
      LENGTH = LEN(NAME)
C
C     Loop to read through data file.
C
   10 IF (.TRUE.) THEN
         READ(LUNUM,'(A)',END=20) SECTION
C
C        String found, set FOUND to 1, and exit loop.
C
         IF (NAME .EQ. SECTION(1:LENGTH)) THEN
            FOUND = 1
            GOTO 20
C
C           String not found, set FOUND to 0.
C
          ELSE
            FOUND = 0
         ENDIF

         LNUM = LNUM + 1
         GOTO 10
      ENDIF

   20 RETURN
      END

!=======================================================================
!  FIND_IN_FILE, Function, M.Jones, 2006
!  Finds NEEDLE in a file of logical unit number HAYSTACK by
!  searching first * characters in each LINE.
!  Same as FIND, but no UPCASE function and NEEDLE is variable length
!-----------------------------------------------------------------------
!  INPUT : HAYSTACK  - logical unit number of the file to read
!          NEEDLE    - 6-character variable name of section to find
!  OUTPUT: FIND_IN_FILE - Indicator of completion of find routine
!             0 - End-of-file encountered i.e. NEEDLE not found
!             1 - NEEDLE was found
!=======================================================================

      INTEGER FUNCTION FIND_IN_FILE(NEEDLE, HAYSTACK)
          IMPLICIT NONE

!         Function returns success (1) or failure (0) code
c         File unit number of file in which to search
          INTEGER HAYSTACK
c         Text to search for:
          CHARACTER*(*) NEEDLE
c         length of chars to search for: 
          INTEGER STRLEN
c         Line to read from file:
          CHARACTER*1024 LINE

c         ====================================
          FIND_IN_FILE = 0
          STRLEN = LEN_TRIM(NEEDLE)

c                 Search file for line starting with needle
                  DO WHILE (.TRUE.)
c                     Read a line
                      READ(HAYSTACK, '(A)', END=50) LINE
c                     Check if the first chars match:
                      IF (LINE(1:STRLEN) .EQ. TRIM(NEEDLE)) THEN 
                          FIND_IN_FILE = 1        
                          RETURN
                      ENDIF
                  ENDDO

   50             RETURN

      END FUNCTION FIND_IN_FILE



C=======================================================================
C  IGNORE, Subroutine, J.W.Jones, 01/03/91
C----------------------------------------------------------------------------
C  PURPOSE: To read lines as an n-character variable and check it
C           for a blank line or for a comment line denoted by ! in col 1.
C  INPUTS:  LUN - Logical unit number of the file to be read
C           LINEXP - Starting line number at which this routine begins to
C                    read the file
C  OUTPUTS: LINEXP - Line number last read by the routine
C           ISECT - Indicator of completion of IGNORE routine
C                   0 - End of file encountered
C                   1 - Found a good line to read
C                   2 - End of Section in file encountered, denoted by *
C                       in column 1
C           CHARTEST - n-character variable containing the contents of
C                      the last line read by the IGNORE routine
C----------------------------------------------------------------------------
C
      SUBROUTINE IGNORE(LUN,LINEXP,ISECT,CHARTEST)

      CHARACTER BLANK*(80),CHARTEST*(*)
      INTEGER   LENGTH, LUN,LINEXP,ISECT
      DATA BLANK/'                                                    '/

      LENGTH = LEN(CHARTEST)

      ISECT = 1
 30   READ(LUN,'(A)',ERR=70, END=70)CHARTEST
      LINEXP = LINEXP + 1

!     CHP 5/1/08
      IF (CHARTEST(1:1) == CHAR(26)) THEN
        GO TO 70
      ENDIF

C     Check to see if all of this section has been read
      IF(CHARTEST(1:1) .EQ. '*'  .OR. CHARTEST(1:1) .EQ. '$') THEN
C        End of section encountered
         ISECT = 2
         RETURN
      ENDIF
C
C     Check for blank lines and comments (denoted by ! in column 1)
      IF(CHARTEST(1:1).NE.'!' .AND. CHARTEST(1:1).NE.'@') THEN
!         IF(CHARTEST(1:80).NE.BLANK)THEN
         IF(CHARTEST(1:LENGTH).NE.BLANK)THEN
C           FOUND A GOOD LINE TO READ
            RETURN
         ENDIF
      ENDIF

      GO TO 30
C     To read the next line

 70   ISECT = 0
      RETURN
      END SUBROUTINE IGNORE

C=======================================================================
C  IGNORE2, Subroutine, J.W.Jones, 01/03/91
C----------------------------------------------------------------------------
C  PURPOSE: To read lines as an n-character variable and check it
C           for a blank line or for a comment line denoted by ! in col 1.
C           Also check for second tier of data as notated by @ in the first
C           column.
C----------------------------------------------------------------------------
! Revision history
! 06/14/2005 CHP Return w/ ISECT=2, when '*' in column 1 found (previously 
!                looked for next data line.
C----------------------------------------------------------------------------
C INPUTS:  LUN - Logical unit number of the file to be read
C          LINEXP - Starting line number at which this routine begins to
C                   read the file
C OUTPUTS: LINEXP - Line number last read by the routine
C          ISECT - Indicator of completion of IGNORE2 routine
C                  0 - End of file encountered
C                  1 - Found a good line to read
C                  2 - End of Section in file encountered ("*") found
C                  3 - Second tier headers found ("@" found)
C          CHARTEST - 80-character variable containing the contents of
C                     the last line read by the IGNORE2 routine
C=======================================================================
      SUBROUTINE IGNORE2(LUN,LINEXP,ISECT,CHARTEST)
      CHARACTER CHARTEST*(*)
      INTEGER LUN,LINEXP,ISECT, Length
!     CHARACTER BLANK*80
!     DATA BLANK/'                                                    '/
C----------------------------------------------------------------------------
      ISECT = 1
 30   READ(LUN,'(A)',ERR=70,END=70)CHARTEST
      LINEXP = LINEXP + 1
C     CHECK TO SEE IF ALL OF THIS SECTION HAS BEEN READ

      IF(CHARTEST(1:1) .EQ. '*' )THEN
C       INTERMEDIATE HEADER FOUND.  
        ISECT = 2
!        GOTO 30
        RETURN
      ENDIF

      IF(CHARTEST(1:1) .EQ.'@') THEN
C       NEXT TIER ENCOUNTERED
        ISECT = 3
        RETURN
      ENDIF
C
C     CHECK FOR BLANK LINES AND COMMENTS (DENOTED BY ! IN COLUMN 1)
      IF(CHARTEST(1:1).NE.'!' .AND. CHARTEST(1:1).NE.'@') THEN
!       IF(CHARTEST.NE.BLANK)THEN
        Length = Len_Trim(CHARTEST)
        IF (Length > 0) THEN
C         FOUND A GOOD LINE TO READ
          RETURN
        ENDIF
      ENDIF

      GO TO 30
C       TO READ THE NEXT LINE
 70   ISECT = 0

      RETURN
      END SUBROUTINE IGNORE2

C=======================================================================
C  HFIND, Subroutine  GPF 7/95
C  Finds appropriate HEADER in a file of logical unit number LUNUM
C  by searching for a 5-character NAME following the '@' at the
C  beginning of a header line
C-----------------------------------------------------------------------
C  INPUT  : LUNUM  logical unit of file to read
C           NAME   variable name of header section to find (5-char)
C  OUTPUT : LNUM   line number of file currently read
C           ISECT  return status of find routine
C                  0  - EOF, name not found
C                  1  - NAME found
C                  2  - End of section encountered, denoted by *
C=======================================================================
      SUBROUTINE HFIND(LUNUM,NAME,LNUM,ISECT)

      IMPLICIT NONE
      INTEGER ISECT,I,LNUM,LUNUM
      CHARACTER HEADER*5,NAME*(*),UPCASE*1,LINE*128
C
C     Initialization, save initial line
C
      ISECT = 1
      DO I = 1, LEN(NAME)
         NAME(I:I) = UPCASE(NAME(I:I))
      END DO

C     Loop to read through data file.

   10  IF (.TRUE.) THEN
         READ(LUNUM,'(A)',ERR=20,END=20) LINE
         LNUM = LNUM + 1

C     End of section

         IF (LINE(1:1) .EQ. '*') THEN
            ISECT = 2
            RETURN
         ENDIF

C     Header line

         IF (LINE(1:1) .EQ. '@') THEN
            HEADER='     '
            DO I=2,LEN(LINE)
               IF (LINE(I:I) .NE. ' ') THEN
                  LINE(I:I) = UPCASE(LINE(I:I))
               ENDIF
            ENDDO
            DO I=2,(LEN(LINE)-LEN(NAME)+1)
               HEADER(1:LEN(NAME)) = LINE(I:(I+LEN(NAME)-1))
               IF (HEADER(1:LEN(NAME)) .EQ. NAME) THEN
                 ISECT = 1
                 RETURN
               ENDIF
            ENDDO
         ENDIF
         GOTO 10
      ENDIF
   20 ISECT = 0
      RETURN
      END
C=======================================================================


C=======================================================================
C  READ_DETAIL, Subroutine, C. H. Porter
C-----------------------------------------------------------------------
C  Reads DETAIL.CDE file, searches for CODE within SECTION, returns TEXT
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  10/15/2007 CHP Written.
C========================================================================

      SUBROUTINE READ_DETAIL(LENCDE, LENTXT, CODE, SECTION, TEXT)
C-----------------------------------------------------------------------
      USE ModuleDefs
      IMPLICIT NONE

      INTEGER,          INTENT(IN)  :: LENCDE, LENTXT
      CHARACTER*(LENCDE), INTENT(IN)  :: CODE
      CHARACTER*(*),    INTENT(IN)  :: SECTION
      CHARACTER*(LENTXT), INTENT(OUT) :: TEXT

      CHARACTER*(LENCDE) FCODE
      CHARACTER*(LENTXT) FTEXT
      CHARACTER*10 FILECDE
      CHARACTER*120 DATAX
      CHARACTER*120 PATHX

      CHARACTER*6, PARAMETER :: ERRKEY = 'DETAIL'
      INTEGER FOUND, FIND_IN_FILE, ERR, LNUM, LUN
!      INTEGER IPX

      LOGICAL FEXIST    !, EOF

      DATA FILECDE /'DETAIL.CDE'/

C-----------------------------------------------------------------------
      TEXT = ''
      LNUM = 0
      DATAX = FILECDE
      INQUIRE (FILE = DATAX, EXIST = FEXIST)

      IF (.NOT. FEXIST) THEN
!       File does not exist in data directory, check directory
!         with executable.
        CALL GETARG(0,PATHX)
!        call path_adj(pathx)
        call get_dir(pathx,datax)
        datax = trim(datax)//filecde
!        IPX = LEN_TRIM(PATHX)
!        DATAX = PATHX(1:(IPX-12)) // FILECDE
C       DATAX = STDPATH // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF        

      IF (.NOT. FEXIST) THEN
!       Last, check for file in C:\DSSAT45 directory
        DATAX = trim(STDPATH) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF

      IF (FEXIST) THEN
        CALL GETLUN('DTACDE',LUN)
        OPEN (LUN, FILE=DATAX, STATUS = 'OLD',IOSTAT=ERR)
        IF (ERR .NE. 0) CALL ERROR(ERRKEY, ERR, DATAX, 0)
        FOUND = FIND_IN_FILE(SECTION,LUN)
        IF (FOUND .EQ. 0) CALL ERROR(SECTION, 42, DATAX, LNUM)
        DO WHILE (.TRUE.)    !.NOT. EOF(LUN)
          READ(LUN,'(A,T10,A)',END=200,IOSTAT=ERR) FCODE, FTEXT
          LNUM = LNUM + 1
          IF (ERR .NE. 0) CALL ERROR(ERRKEY, ERR, DATAX, LNUM)
          IF (CODE .EQ. FCODE) THEN
            TEXT = FTEXT
            EXIT
          ENDIF
        ENDDO
        CLOSE(LUN)
      ELSE
!       Detail.CDE file is missing -- stop program with message.
        CALL ERROR(ERRKEY, 29, FILECDE, 0)
      ENDIF
  200 CONTINUE
      RETURN
      END SUBROUTINE READ_DETAIL
C=======================================================================

!=======================================================================
!  Subroutine READA
!   Reads measured development and final harvest data from FILEA 
!   and maps measured data to appropriate headers for output to 
!   OVERVEIW.OUT
!-----------------------------------------------------------------------
!  Revision history:
!  08/12/2005 CHP Modified to read "alias" headers for some variables
C  02/09/2007 GH  Add path for fileA
!=======================================================================
      SUBROUTINE READA(FILEA, PATHEX, OLAB, TRTNUM, YRSIM, X)

!-----------------------------------------------------------------------
!     READ DEVELOPMENT AND FINAL HARVEST DATA FROM  FILEA
!-----------------------------------------------------------------------
      USE ModuleDefs
      IMPLICIT NONE

      INTEGER TRTNUM,ERRNUM,LUNA,LINEXP,ISECT,NTR,I, J
      INTEGER YRSIM,YR,ISIM
      INTEGER COUNT
!     Headers with aliases -- save column
      INTEGER HWAM, HWAH, BWAM, BWAH, PDFT, R5AT  

      REAL TESTVAL

      CHARACTER*6   OLAB(EvaluateNum), HD
      CHARACTER*6   HEAD(EvaluateNum)
      CHARACTER*6   DAT(EvaluateNum), X(EvaluateNum)  !, ERRKEY
      CHARACTER*12  FILEA
      CHARACTER*78  MSG(10)
	CHARACTER*80  PATHEX
	CHARACTER*92  FILEA_P
      CHARACTER*255 C255

      LOGICAL FEXIST

      FILEA_P = TRIM(PATHEX)//FILEA

C-----------------------------------------------------------------------
C     Initialize measured values to -99 before reading values
C
      X = '   -99'

      CALL GETLUN('FILEA', LUNA)
      LINEXP = 0

      INQUIRE (FILE = FILEA_P, EXIST = FEXIST)

      IF (FEXIST) THEN
        OPEN (LUNA,FILE = FILEA_P,STATUS = 'OLD',IOSTAT=ERRNUM)
        IF (ERRNUM .NE. 0) GOTO 5000
        CALL YR_DOY(YRSIM,YR,ISIM)

C       FIND THE HEADER LINE, DESIGNATED BY @TRNO
        DO WHILE (.TRUE.)
          READ(LUNA,'(A)',END=5000) C255
          LINEXP = LINEXP + 1
          IF (C255(1:1) .EQ. '@') EXIT    
        ENDDO

C       FOUND HEADER LINE, SAVE IT IN HEAD AND SEARCH FOR TREATMENT
        DO I = 1,EvaluateNum
          READ(C255,'(1X,A5)') HEAD(I)
          IF (HEAD(I) .EQ. '     ') THEN
            COUNT = I - 1
            EXIT
          ENDIF
          C255 = C255(7:255)
        ENDDO

C       FIND THE RIGHT TREATMENT LINE OF DATA
        DO I = 1,1000
          CALL IGNORE(LUNA,LINEXP,ISECT,C255)
C
C    Return if no matching treatment is found in file FILA
C    No field measured data are necessary to be able to run the
C    model

          IF (ISECT .EQ. 0) GO TO 100
          READ(C255(1:6),'(2X,I4)',IOSTAT=ERRNUM) NTR
          IF (ERRNUM .NE. 0) GOTO 5000
          IF(NTR .EQ. TRTNUM) GO TO 60
        ENDDO
  
  60    CONTINUE
  
C       READ DATA LINE
        DO I = 1,COUNT
          READ(C255,'(A6)',IOSTAT=ERRNUM) DAT(I)
          IF (ERRNUM .NE. 0) GOTO 5000

          !Test for numeric value -- set non-numeric values to -99
          READ(C255,'(F6.0)',IOSTAT=ERRNUM) TESTVAL
          IF (ERRNUM .NE. 0 .AND. 
     &        TRIM(ADJUSTL(HEAD(I))) .NE. 'TNAM') THEN
            DAT(I) = '   -99'
          ENDIF 

          C255 = C255(7:255)
        ENDDO
  
!       Search for location within array of headers which can
!       contain the same data.  Store index for later use.
!       Pairs of headers:  
!       'HWAM' or 'HWAH' 
!       'BWAM' or 'BWAH' 
!       'PDFT' or 'R5AT' 
        HWAM = -99; HWAH = -99
        BWAM = -99; BWAH = -99
        PDFT = -99; R5AT = -99
        DO J = 1, EvaluateNum
          SELECT CASE (OLAB(J))
            CASE('HWAM  '); HWAM = J 
            CASE('HWAH  '); HWAH = J 
            CASE('BWAM  '); BWAM = J 
            CASE('BWAH  '); BWAH = J 
            CASE('PDFT  '); PDFT = J 
            CASE('R5AT  '); R5AT = J 
          END SELECT
        ENDDO

C       MATCH HEADER WITH DATA
        DO I = 2, COUNT   !Loop thru FILEA headers
          HD = ADJUSTL(HEAD(I))

!         For "alias" headers already know columns to store data
          SELECT CASE(HD)

          CASE ('HWAM')
            IF (HWAM > 0) THEN
              !store HWAM with HWAM header
              X(HWAM) = DAT(I)    
            ELSEIF (HWAH > 0) THEN
              !store HWAM with HWAH header 
              IF (X(HWAH) == '   -99') X(HWAH) = DAT(I)  
            ENDIF

          CASE ('HWAH')
            IF (HWAH > 0) THEN
              !store HWAH with HWAH header
              X(HWAH) = DAT(I)
            ELSEIF (HWAM > 0) THEN
              !store HWAH with HWAM header
              IF (X(HWAM) == '   -99') X(HWAM) = DAT(I) 
            ENDIF

          CASE ('BWAM')
            IF (BWAM > 0) THEN
              !store BWAM with BWAM header
              X(BWAM) = DAT(I)
            ELSEIF (BWAH > 0) THEN
              !store BWAM with BWAH header
              IF (X(BWAH) == '   -99') X(BWAH) = DAT(I)
            ENDIF

          CASE ('BWAH')
            IF (BWAH > 0) THEN
              !store BWAH with BWAH header
              X(BWAH) = DAT(I)
            ELSEIF (BWAM > 0) THEN
              !store BWAH with BWAM header
              IF (X(BWAM) == '   -99') X(BWAM) = DAT(I)
            ENDIF

          CASE ('PDFT')  
            IF (PDFT > 0) THEN
              !store PDFT with PDFT header
              X(PDFT) = DAT(I)
            ELSEIF (R5AT > 0) THEN
              !store PDFT with R5AT header
              IF (X(R5AT) == '   -99') X(R5AT) = DAT(I)
            ENDIF

          CASE ('R5AT')  
            IF (R5AT > 0) THEN
              !store R5AT with R5AT header
              X(R5AT) = DAT(I)
            ELSEIF (PDFT > 0) THEN
              !store R5AT with PDFT header
              IF (X(PDFT) == '   -99') X(PDFT) = DAT(I)
            ENDIF

!         No aliases for the rest
          CASE DEFAULT
            DO J = 1, EvaluateNum    !Loop thru crop-specific headers
              IF (OLAB(J) == HD) THEN
                X(J) = DAT(I)
                EXIT
              ENDIF
            ENDDO

          END SELECT
        ENDDO
  
 100    CONTINUE
        CLOSE(LUNA)
        RETURN

!       Error handling
 5000   CONTINUE
        X = '   -99'
        WRITE (MSG(1),'(" Error in FILEA - Measured data not used")')
        CALL INFO(1, "READA ", MSG)
      ENDIF

      CLOSE (LUNA)
      RETURN
      END SUBROUTINE READA

C=======================================================================
C  GETDESC, Subroutine C.H. Porter
C  Reads DATA.CDE to get descriptions for Measured and predicted data.
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  02/01/2002 CHP Written.
C  12/17/2004 CHP Increased length of PATHX (path for executable) to 120.
C=======================================================================

      SUBROUTINE GETDESC(COUNT, OLAB, DESCRIP)

C-----------------------------------------------------------------------
      USE ModuleDefs
      IMPLICIT NONE

      CHARACTER*6 CODE, OLAB(*)
      CHARACTER*6, PARAMETER :: ERRKEY = 'GETDSC'
      CHARACTER*8  FILECDE
      CHARACTER*50 DESCRIP(*), LONGTEXT
      CHARACTER*120 DATAX
      CHARACTER*78 MSG(3)
      CHARACTER*120 PATHX

      INTEGER COUNT, ERR, I, LUN, LNUM

      LOGICAL FEXIST    !, EOF

      DATA FILECDE /'DATA.CDE'/
C-----------------------------------------------------------------------

      DATAX = FILECDE
      INQUIRE (FILE = DATAX, EXIST = FEXIST)

      IF (.NOT. FEXIST) THEN
!       File does not exist in data directory, check directory
!         with executable.
        CALL GETARG(0,PATHX)
!        call path_adj(pathx)
        call get_dir(pathx,datax)
        datax = trim(datax)//filecde
!        IPX = LEN_TRIM(PATHX)
!        DATAX = PATHX(1:(IPX-12)) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF        

      IF (.NOT. FEXIST) THEN
!       Last, check for file in C:\DSSAT45 directory
        DATAX = trim(STDPATH) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF

      IF (FEXIST) THEN
        CALL GETLUN('DTACDE',LUN)
        OPEN (LUN, FILE=DATAX, STATUS = 'OLD', IOSTAT=ERR)
        IF (ERR /= 0) GOTO 100
        DO I = 1, COUNT
          DESCRIP(I) = ' '
          REWIND (LUN)
          LNUM = 0
          DO WHILE (.TRUE.)
            LNUM = LNUM + 1
            READ(LUN,'(A6,17X,A50)',END=20,ERR=20,IOSTAT=ERR)
     &          CODE,LONGTEXT
            IF (CODE .EQ. OLAB(I) .AND. ERR .EQ. 0) THEN
              DESCRIP(I) = LONGTEXT
              EXIT
            ENDIF
          ENDDO 
          !print *, i, " of ", count, olab(i), " ", descrip(i)
   20     IF (DESCRIP(I) .EQ. ' ') THEN
            DESCRIP(I) = OLAB(I)
            IF (ERR < 0) EXIT
          ENDIF
        ENDDO
        CLOSE(LUN)
        RETURN
      ELSE
!       Data.CDE file is missing -- stop program with message.
        CALL ERROR(ERRKEY, 29, FILECDE, 0)
      ENDIF

!     Data.cde file can not be found.  Just use OLAB (four character
!       code to fill description array.
  100   DO I = 1, COUNT
          DESCRIP(I) = OLAB(I)
        ENDDO

        WRITE(MSG(1),11) FILECDE
        WRITE(MSG(2),12) 
        WRITE(MSG(3),13) 
   11   FORMAT(' ',A8,' can not be found.')
   12   FORMAT(' Overview file will display variable labels ')
   13   FORMAT(' for simulated vs. measured data.')

        CALL INFO(3, ERRKEY, MSG)

C-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE GETDESC
C=======================================================================

C=======================================================================
C  CHANGE_DESC, Subroutine C.H. Porter
C  Change units from 'YrDoy' to 'DAP' in descriptions for which date
!     conversion has been done.
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  02/01/2002 CHP Written.
C=======================================================================

      SUBROUTINE CHANGE_DESC(Descrip)

C-----------------------------------------------------------------------
      IMPLICIT NONE

      INTEGER StartCol, I, LENGTH
      CHARACTER*1 CHAR, UPCASE
      CHARACTER(50) Descrip, NewDesc
C-----------------------------------------------------------------------
!     Convert string to uppercase - save as NewDesc
      LENGTH = LEN(TRIM(Descrip))
      NewDesc = ""
      DO I = 1, LENGTH
        CHAR = Descrip(I:I)
        CHAR = UPCASE(CHAR)
        NewDesc = NewDesc(1:I-1) // CHAR
      ENDDO

!     Find occurance of '(YRDOY)' in description string
      StartCol = INDEX(NewDesc,'(YRDOY)')
      IF (StartCol .GT. 0) THEN
        Descrip = 
     &    Descrip(1:StartCol-1) // '(dap)  ' // Descrip(StartCol+7:50)
      ENDIF

C-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE CHANGE_DESC
C=======================================================================

C=======================================================================
C  READA_Dates, Subroutine C.H. Porter
C  Convert dates from READA (text) to ouptut format (integer)
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  05-14-2002 CHP Written.
C=======================================================================
      SUBROUTINE READA_Dates(XDAT, YRSIM, IDAT)

      IMPLICIT NONE

      CHARACTER*6 XDAT
      REAL        RDAT
      INTEGER     ERRNUM, IDAT, ISIM, YR, YRSIM

      CALL YR_DOY(YRSIM, YR, ISIM)
      READ(XDAT(1:6),1000,IOSTAT=ERRNUM) RDAT
      IF (ERRNUM .NE. 0) CALL ERROR('READA ',2,'FILEA',0)
 1000 FORMAT(F6.0)
      IDAT = INT(RDAT)

      IF (IDAT .GT. 0 .AND. IDAT .LT. 1000) THEN
        IF (IDAT .GT. ISIM) THEN
          IDAT = YR*1000 + IDAT
        ELSE
          IDAT = (YR+1)*1000 + IDAT
        ENDIF

      ELSEIF (IDAT .GT. 0 .AND. IDAT .GE. 1000) THEN
        CALL Y2K_DOY(IDAT)
        !CALL FullYear (IDAT, YR, DOY)
        !IDAT = YR*1000 + DOY
      ENDIF

      RETURN
      END SUBROUTINE READA_Dates
C=======================================================================

C=======================================================================
C  PARSE_HEADERS, Subroutine C.H. Porter
C  Reads a line of headers and determines column widths for 
C     corresponding data.
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  11/05/2003 CHP Written.
C=======================================================================
      SUBROUTINE PARSE_HEADERS(CHAR, MAXCOL, HEADER, COUNT, COL)

      IMPLICIT NONE
      CHARACTER*(*) CHAR
      INTEGER MAXCOL
!     Up to MAXCOL headers per line, up to 10 characters long each
      CHARACTER*15  HEADER(MAXCOL), TEXT
      INTEGER COUNT, COL(MAXCOL,2), I, J, LENGTH
      LOGICAL SPACES

!     Initialize
      HEADER = '          '
      COUNT = 0
      COL = 0
      LENGTH = LEN(TRIM(CHAR))    !Don't use first character
      IF (LENGTH .LE. 1) RETURN   !No headers to parse, go back empty

      COUNT = 1
      COL(COUNT,1) = 1
      SPACES = .TRUE. !Allows for multiple spaces between headers

!     Look for spaces between headers
      DO I = 2, LENGTH       
!       A "!" signifies -- do not read the rest of the record 
        IF (CHAR(I:I) .EQ. '!') THEN
          LENGTH = I-1
          EXIT
        
        ELSEIF (CHAR(I:I) .EQ. ' ') THEN
          IF (SPACES) THEN
            CYCLE         !Treat multiple blanks as one
          ELSE
            COL(COUNT,2) = I - 1
            HEADER(COUNT) = ADJUSTL(CHAR(COL(COUNT,1):COL(COUNT,2)))
            COUNT = COUNT + 1
            COL(COUNT,1) = I + 1
            SPACES = .TRUE.
          ENDIF
        ELSE
          SPACES = .FALSE.
          CYCLE
        ENDIF
      ENDDO
      COL(COUNT,2) = LENGTH
      HEADER(COUNT) = ADJUSTL(CHAR(COL(COUNT,1):COL(COUNT,2)))
      HEADER(1) = ADJUSTL(CHAR(2:COL(1,2)))

!     Take out trailing periods from header names
      DO I = 1, COUNT
        TEXT = HEADER(I)
        LENGTH = LEN(TRIM(TEXT))
        DO J = LENGTH, 1, -1
          IF (TEXT(J:J) .EQ. '.' .OR. TEXT(J:J) .EQ. ' ') THEN
            LENGTH = LENGTH - 1
          ELSE
            EXIT
          ENDIF
        ENDDO
        HEADER(I) = TEXT(1:LENGTH)
      ENDDO
!--------------------------------------------------------------------
!     Need to adjust starting and ending columns for left-justified 
!         column headers
      DO I = 1, COUNT
        IF (TRIM(HEADER(I)) .EQ. 'SOIL_NAME') THEN
          !For SOIL_NAME, start in column 2 and read 50 characters
          COL(COUNT,1) = 2
          COL(COUNT,2) = 51
        ENDIF

        IF (TRIM(HEADER(I)) .EQ. 'SOIL_SCS') THEN
          !For SOIL_SCS, start 2 columns over from previous column 
          !  and read 50 characters
          IF (COUNT .GT. 1) THEN
            COL(COUNT,1) = COL(COUNT-1,2) + 2
            COL(COUNT,2) = COL(COUNT,1) + 49
          ELSE
            COL(COUNT,1) = 2
            COL(COUNT,2) = 51
          ENDIF
        ENDIF
      ENDDO

      RETURN
      END SUBROUTINE PARSE_HEADERS
C=======================================================================


C=======================================================================
C  ReadCropModels, Subroutine, C. H. Porter
C-----------------------------------------------------------------------
C  Reads SIMULATION.CDE file, reads Crop Models section
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  10/30/2007 CHP Written.
C========================================================================

      SUBROUTINE ReadCropModels(MaxNum, ModelName, CropName)
C-----------------------------------------------------------------------
      USE Moduledefs
      IMPLICIT NONE

      INTEGER MaxNum
      CHARACTER*2   CropName(MaxNum)
      CHARACTER*5   ModelName(MaxNum)
      CHARACTER*14  FILECDE
      CHARACTER*23  SECTION
      CHARACTER*80  CHAR
      CHARACTER*120 DATAX
      CHARACTER*120 PATHX

      CHARACTER*6, PARAMETER :: ERRKEY = 'SIMCDE'
      INTEGER FOUND, ERR, LNUM, LUN
      INTEGER I, ISECT, FIND_IN_FILE

      LOGICAL FEXIST    !, EOF

      DATA FILECDE /'SIMULATION.CDE'/

C-----------------------------------------------------------------------
      DATAX = FILECDE
      INQUIRE (FILE = DATAX, EXIST = FEXIST)

      IF (.NOT. FEXIST) THEN
!       File does not exist in data directory, check directory
!         with executable.
        CALL GETARG(0,PATHX)
!        call path_adj(pathx)
        call get_dir(pathx,datax)
        datax = trim(datax)//filecde
!        IPX = LEN_TRIM(PATHX)
!        DATAX = PATHX(1:(IPX-12)) // FILECDE
C       DATAX = STDPATH // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF        

      IF (.NOT. FEXIST) THEN
!       Last, check for file in C:\DSSAT45 directory
        DATAX = trim(STDPATH) // FILECDE
        INQUIRE (FILE = DATAX, EXIST = FEXIST)
      ENDIF

      IF (FEXIST) THEN
        CALL GETLUN('SIMCDE',LUN)
        OPEN (LUN, FILE=DATAX, STATUS = 'OLD',IOSTAT=ERR)
        IF (ERR .NE. 0) CALL ERROR(ERRKEY, ERR, DATAX, 0)
        SECTION = '*Simulation/Crop Models'
        FOUND = FIND_IN_FILE(SECTION, LUN)
        IF (FOUND .EQ. 0) CALL ERROR(SECTION, 42, DATAX, LNUM)
        I = 0
        DO WHILE (.TRUE.)   !.NOT. EOF(LUN)
          CALL IGNORE(LUN,LNUM,ISECT,CHAR)
          IF (ISECT == 1) THEN
            I = I + 1
            READ(CHAR,*) ModelName(I), CropName(I)
            IF (I == MaxNum) EXIT
          ELSE
            EXIT
          ENDIF
        ENDDO
        CLOSE(LUN)
      ELSE
!       Simulation.CDE file is missing -- stop program with message.
        CALL ERROR(ERRKEY, 29, FILECDE, 0)
      ENDIF

      RETURN
      END SUBROUTINE ReadCropModels
C=======================================================================


C=======================================================================
C  ERROR, Subroutine, N.B. Pickering
C  Outputs error messages to screen from file ERROR.DAT
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  08/30/1991 NBP Written
C  10/31/1994 GH  Add option to read MODEL.ERR from model EXE path
C  12/04/2001 CHP Added call to GETLUN for unit number.
!  11/05/2002 AJG Increased the size of PATHX.
C  07/21/2003 CHP Added call to WARNING.out for error messages.  
C                 Added generic messages for open and read errors.
C  11/23/2004 CHP Increased length of PATHX (path for executable) to 120.
C-----------------------------------------------------------------------
C  Input : ERRKEY,ERRNUM,FILE,LNUM)
C  Output: message to screen
C  Local :
C  Ifile : MODEL.ERR
C  FN/SUB: FIND
C=======================================================================

      SUBROUTINE ERROR (ERRKEY,ERRNUM,FILE,LNUM)

      USE ModuleDefs
      USE ModuleData
      IMPLICIT      NONE

      CHARACTER*(*) ERRKEY,FILE
      CHARACTER*9   EFILE
      CHARACTER     AKEY*6,BLANK*80,KEY*6,LINE*80
      CHARACTER*78  MSG(10)
      CHARACTER*100 ERRORX, SAVE_ERRORX
      CHARACTER*120 PATHX

      INTEGER       ANUM,ERRNUM,LNUM, LUN, I , ELUN
      INTEGER       IMSG

      LOGICAL       FEXIST, FOUND  !, EOF


      PARAMETER     (BLANK = ' ')

      TYPE (ControlType) CONTROL
      CALL GET(CONTROL)

      IMSG = 1
      EFILE = 'ERROR.OUT'
      CALL GETLUN('ERRORO', ELUN)

      INQUIRE (FILE = EFILE, EXIST = FEXIST)
      IF (FEXIST) THEN
        OPEN (UNIT = ELUN, FILE = EFILE, STATUS = 'OLD', 
     &    POSITION = 'APPEND')
      ELSE
        OPEN (UNIT = ELUN, FILE = EFILE, STATUS = 'NEW')
        WRITE(ELUN,'("*RUN-TIME ERRORS OUTPUT FILE",//)')
      ENDIF

      CALL HEADER(SEASINIT, ELUN, CONTROL%RUN)
      WRITE(ELUN,'(A,", Trt",I5)') CONTROL%FILEX, CONTROL%TRTNUM

      CALL GETARG(0,PATHX)
!      call path_adj(pathx)
      call get_dir(pathx,errorx)
      errorx = trim(errorx)//'MODEL.ERR'

!     If ERRORX file is not in executable directory, try std. location
      INQUIRE (FILE = ERRORX, EXIST = FEXIST)
      IF (.NOT. FEXIST) THEN
        SAVE_ERRORX = ERRORX
        ERRORX = trim(STDPATH) // 'MODEL.ERR'
      ENDIF

      INQUIRE (FILE = ERRORX,EXIST = FEXIST)
      IF (FEXIST) THEN

         CALL GETLUN('ERRORX', LUN)
         OPEN (LUN,FILE=ERRORX,STATUS='OLD')
C
C        Initialization
C
         FOUND = .FALSE.
         IF (ERRNUM .GT. 6000 .OR. ERRNUM .LT. 0) THEN
            KEY = 'MISC  '
         ELSE
            KEY = ERRKEY
         ENDIF
C
C        Loop to search for error message in file MODEL.ERR.
C
   10    DO WHILE(.TRUE.)
           READ (LUN,'(A)',END=20) LINE
           AKEY = LINE(1:6)
           IF (AKEY .EQ. KEY) THEN
              READ (LINE,'(6X,I5)') ANUM
              IF (ANUM .EQ. ERRNUM) THEN
                 FOUND = .TRUE.
                 GOTO 20
              ENDIF
            ELSE
              FOUND = .FALSE.
           ENDIF
         ENDDO

   20    IF (FOUND) THEN
            WRITE (*,*)
            WRITE (ELUN,*)
   30       READ  (LUN,'(A)',END=40) LINE
            IF (LINE .NE. BLANK) THEN
               WRITE (*,*) LINE
               WRITE (ELUN,*) LINE
               WRITE(MSG(IMSG),'(A77)') LINE  ; IMSG = IMSG+1
               GOTO 30
            ENDIF
          ELSEIF (KEY .NE. 'GENERI') THEN
          !Check for generic message numbers
             KEY = 'GENERI'
             REWIND (LUN)
             GO TO 10

!        As an alternate, could have generic messages generated in code.
!            CALL GENERIC_MSG(ERRNUM, LINE)
!            WRITE (*,'(/,A78,/)') LINE
!            WRITE (ELUN,'(/,A78,/)') LINE
!            WRITE (MSG(IMSG),'(A78)') LINE
!            IMSG = IMSG + 1

          ELSE
!           Could not find error message in file
            WRITE (MSG(IMSG),'(A,A,I5)') 'Unknown ERROR. ',
     &           'Error number: ',ERRNUM
            WRITE (ELUN,'(/,A78,/)') MSG(IMSG)
            WRITE (*,'(/,A78)') MSG(IMSG)
            IMSG = IMSG + 1
          ENDIF

   40    IF (FILE .EQ. ' ') THEN
            WRITE (*,'(2A/)') 'Error key: ',ERRKEY
            WRITE (ELUN,'(2A/)') 'Error key: ',ERRKEY
            WRITE (MSG(IMSG),'(2A)') 'Error key: ',ERRKEY
            IMSG = IMSG + 1
          ELSE
            I = MIN(LEN(TRIM(FILE)),37)
            WRITE (*,'(3A,I5,2A/)')
     &    'File: ',FILE(1:I),'   Line: ',LNUM,'   Error key: ',ERRKEY
            WRITE (ELUN,'(3A,I5,2A/)')
     &    'File: ',FILE(1:I),'   Line: ',LNUM,'   Error key: ',ERRKEY
            WRITE (MSG(IMSG),'(2A)') 'File: ',TRIM(FILE(1:I))
            WRITE (MSG(IMSG+1),'(A,I5)') '   Line: ',LNUM
            WRITE (MSG(IMSG+2),'(2A)') '   Error key: ',ERRKEY
            IMSG = IMSG + 3
         ENDIF
         CLOSE (LUN)
      ELSE
C                                                                !BDB
C        Tell user that error file can not be found and give a   !BDB
C        generic error message.                                  !BDB
C                                                                !BDB
         ERRORX = SAVE_ERRORX
         WRITE (*,50) TRIM(ERRORX)
         WRITE (ELUN,50) TRIM(ERRORX)
         WRITE (MSG(IMSG),51) TRIM(ERRORX)
         IMSG = IMSG + 1
   50    FORMAT('Could not locate error file: ',A,/)
   51    FORMAT('Could not locate error file: ',A48)

         !Check for generic message numbers
         CALL GENERIC_MSG(ERRNUM, LINE)
         WRITE (*,'(/,A78,/)') LINE(1:78)
         WRITE (ELUN,'(/,A78,/)') LINE(1:78)
         WRITE (MSG(IMSG),'(A78)') LINE(1:78)
         IMSG = IMSG + 1

         WRITE (*,60)  FILE, LNUM, ERRKEY   
         WRITE (ELUN,60)  FILE, LNUM, ERRKEY   
         WRITE (MSG(IMSG),60) FILE, LNUM, ERRKEY
         IMSG = IMSG + 1
   60    FORMAT('File: ',A12,'   Line: ',I5,' Error key: ',A)
      ENDIF

      WRITE(ELUN,70)
   70 FORMAT("Additional information may be available ",
     &            "in WARNING.OUT file.")
      WRITE (*,70) 
      WRITE (*, *) CHAR(7)
      WRITE (*,260)
260   FORMAT (/,1X,'Please press < ENTER > key to continue ',2X,$)
C-GH      READ  (*, *)

      CLOSE (ELUN)

      WRITE(MSG(IMSG),'(A)') "Simulations terminated."
      CALL WARNING(IMSG, ERRKEY, MSG)

!      INQUIRE (FILE = "LUN.LST", EXIST = FEXIST)
!      IF (FEXIST) THEN
!        CALL GETLUN('LUN.LST', LUN)
!        INQUIRE (UNIT = LUN, OPENED = FOPEN) 
!        IF (.NOT. FOPEN) THEN
!          OPEN (FILE="LUN.LST", UNIT=LUN, ERR=99, STATUS='OLD')
!        ELSE
!          REWIND(LUN)
!        ENDIF
!
!        !Skip over first 3 lines in LUN.LST file
!        DO I=1,3
!          READ(LUN,'(A80)') LINE
!        ENDDO
!
!        !Read list of unit numbers that have been opened and close each
!!       EOF not portable
!!       DO WHILE (.NOT. EOF(LUN))
!        ERR = 0
!        DO WHILE (ERR == 0)
!          READ(LUN, '(A)', IOSTAT=ERRNUM, ERR=99, END=99) LINE
!          READ(LINE,'(I5)',IOSTAT=ERRNUM, ERR=99) LUNIT
!          IF (ERRNUM /= 0) EXIT
!          ERR = ERRNUM
!          IF (LUNIT .NE. LUN) THEN
!            CLOSE (LUNIT)
!          ENDIF
!        ENDDO
!        CLOSE (LUN)
!      ENDIF
!
   99 STOP 99
      END SUBROUTINE ERROR

!=========================================================================
      SUBROUTINE GENERIC_MSG(ERRNUM, MESSAGE)
!     If error messages cannot be found in MODEL.ERR file, or if MODEL.ERR
!     file cannot be found, check for generic message type.

      IMPLICIT NONE
      INTEGER ERRNUM
      CHARACTER*(*) MESSAGE

      !Check for generic message numbers
      SELECT CASE(ERRNUM)
        CASE(29)
          WRITE(MESSAGE,35) 'File not found. Please check ',
     &      'file name or create file. Error number: ', ERRNUM 
        CASE(33)
          WRITE(MESSAGE,35) 'End of file encountered. ',
     &      'Error number: ',ERRNUM
        CASE(59)
          WRITE(MESSAGE,35) 'Syntax error. ',
     &      'Error number: ',ERRNUM
        CASE(64)
          WRITE(MESSAGE,35) 'Invalid format in file. ',
     &      'Error number: ', ERRNUM
        CASE DEFAULT 
          WRITE(MESSAGE,35) 'Unknown ERROR. ',
     &      'Error number: ',ERRNUM
      END SELECT

   35 FORMAT(A,A,I5)

      END SUBROUTINE GENERIC_MSG
!=========================================================================

!=======================================================================
! ErrorCode, Subroutine, C.H. Porter, 02/09/2010
! Ends a run for errors by setting YREND variable.  Continue with next
!     simulation in batch.  Stops sequence simulation.

!-----------------------------------------------------------------------
! REVISION HISTORY
! 02/09/2010 CHP Written
!-----------------------------------------------------------------------
      SUBROUTINE ErrorCode(CONTROL, ErrCode, ERRKEY, YREND)

      USE ModuleDefs
      USE ModuleData
      IMPLICIT NONE

      CHARACTER(*) ERRKEY 
      CHARACTER*78 MSG(4)
      INTEGER ErrCode, YREND
      TYPE (ControlType) CONTROL

!-----------------------------------------------------------------------
      YREND = CONTROL%YRDOY
      CONTROL % ErrCode = ErrCode
      CALL PUT(CONTROL)

!     For sequence runs, stop run with any error
      IF(INDEX('FQ',CONTROL%RNMODE) > 0)CALL ERROR(ERRKEY,ErrCode,' ',0)

      WRITE(MSG(1),'(A,I8,A)') "Run",CONTROL%RUN, " will be terminated."
      CALL WARNING(1,ERRKEY,MSG)

      RETURN
      END SUBROUTINE ErrorCode
!=======================================================================
! Current error codes:

! Daily weather data
!  1 Header section not found in weather file.
!  2 Solar radiation data error
!  3 Precipitation data error
!  4 Tmax and Tmin are both set to 0
!  5 Tmax and Tmin have identical values
!  6 Tmax is less than Tmin
!  8 Non-sequential data in file
! 10 Weather record not found
! 29 Weather file not found
! 30 Error opening weather file
! 59 Invalid format in weather file
! 64 Syntax error.
!
! Weather modification
! 72 Solar radiation data error
! 73 Precipitation data error
! 74 Tmax and Tmin are both set to 0
! 75 Tmax and Tmin have identical values
! 76 Tmax is less than Tmin
!
! Generated weather data
! 82 Solar radiation data error
! 83 Precipitation data error
! 84 Tmax and Tmin are both set to 0
! 85 Tmax and Tmin have identical values
! 86 Tmax is less than Tmin

!100 Number of cohorts exceeds maximum.


C=======================================================================
C  INFO, Subroutine, C.H.PORTER
C  Writes informational messages to INFO.OUT file
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  07/27/2006 CHP Written
!  01/11/2007 CHP Changed GETPUT calls to GET and PUT
C=======================================================================

      SUBROUTINE INFO (ICOUNT, ERRKEY, MESSAGE)

!     FILEIO and RUN needed to generate header for INFO.OUT file

      USE ModuleDefs
      USE ModuleData
      USE HeaderMod
      IMPLICIT NONE
      SAVE

      CHARACTER*(*) ERRKEY
      CHARACTER*11, PARAMETER :: InfoOut = 'INFO.OUT'
      CHARACTER*30  FILEIO
      CHARACTER*(*) MESSAGE(*)

      INTEGER ICOUNT, DOY, I, LUN, OLDRUN, RUN, YEAR, YRDOY
      LOGICAL FIRST, FEXIST, FOPEN

      TYPE (ControlType) CONTROL
      TYPE (SwitchType)  ISWITCH

      DATA FIRST /.TRUE./
      DATA OLDRUN /0/

!-----------------------------------------------------------------------
!     Suppress INFO.OUT if IDETL = '0' (zero) or 'N' 
      CALL GET(ISWITCH)
      IF (INDEX('0N',ISWITCH % IDETL) > 0) RETURN

      CALL GET(CONTROL)
      FILEIO = CONTROL % FILEIO
      RUN    = CONTROL % RUN
      YRDOY  = CONTROL % YRDOY
      
!      IF (INDEX(ERRKEY,'ENDRUN') <= 0) THEN
!       First time routine is called to print, open file.
!       File will remain open until program execution is stopped.
        IF (FIRST) THEN

!         Check for IDETL = '0' (zero) --> suppress output
          CALL GET(ISWITCH)
          IF (ISWITCH % IDETL == '0') RETURN

          CALL GETLUN('OUTINFO', LUN)
          INQUIRE (FILE = InfoOut, EXIST = FEXIST)
          IF (FEXIST) THEN
            INQUIRE (FILE = InfoOut, OPENED = FOPEN)
            IF (.NOT. FOPEN) THEN
              OPEN (UNIT=LUN, FILE=InfoOut, STATUS='OLD',
     &            POSITION='APPEND')
            ENDIF
          ELSE
            OPEN (UNIT=LUN, FILE=InfoOut, STATUS='NEW')
            WRITE(LUN,'("*INFO DETAIL FILE")')
          ENDIF

          WRITE(LUN,'(/,78("*"))')
!          IF (CONTROL % MULTI > 1) CALL MULTIRUN(RUN, 0)
          IF (Headers % RUN == RUN) THEN
            CALL HEADER(SEASINIT, LUN, RUN)
            FIRST = .FALSE.
            OLDRUN = RUN
          ENDIF 
		ELSE
!         VSH
          CALL GETLUN('OUTINFO', LUN)
          INQUIRE (FILE = InfoOut, OPENED = FOPEN)
          IF (.NOT. FOPEN) THEN
             OPEN (UNIT=LUN, FILE=InfoOut, STATUS='OLD',
     &             POSITION='APPEND')
          ENDIF               
        ENDIF   ! for FIRST

        IF (ICOUNT > 0) THEN
          !Print header if this is a new run.
          IF (OLDRUN .NE. RUN .AND. RUN .NE. 0 .AND. FILEIO .NE. "")THEN
            IF (Headers % RUN == RUN) THEN
              CALL HEADER(SEASINIT,LUN,RUN)
              OLDRUN = RUN
            ENDIF
          ENDIF

!         Print the INFO.  Message is sent from calling routine as text.
          CALL YR_DOY(YRDOY, YEAR, DOY)
          WRITE(LUN,'(/,1X,A,"  YEAR DOY = ",I4,1X,I3)')ERRKEY,YEAR,DOY
          DO I = 1, ICOUNT
            WRITE(LUN,'(1X,A)') MESSAGE(I)
          ENDDO
        ENDIF

!      ELSE    !ERRKEY = 'ENDRUN' -> End of season
      IF (INDEX(ERRKEY,'ENDRUN') > 0) THEN
        FIRST = .TRUE.
        CLOSE(LUN)
      ENDIF

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE INFO



C=======================================================================
C  DATES, File, Nigel Pickering, G. Hoogenboom, P.W. Wilkens and B. Baer
C  General functions related to time and date calculations
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  09/13/1991 NBP Developed
C  01/30/1998 GH  Modified ENDYR function for leap year
C  01/30/1998 GH  Added NAILUJ routine
C  02/02/1998 GH  Integer in YR_DOY
C  07/01/2000 GH  Added INCDAT
C  06/09/2002 GH  Modified for Y2K
C  11/29/2004 CHP Added ETAD_NAILUJ for Ponding routine -- provides 
C                   integer month given date.
!  10/11/2005 CHP Fix problem in Y2K_DOYW, sequenced runs spanning Y2K 
!  11/16/2007 CHP Added leap year function
C=======================================================================
C=======================================================================
C  DOYC, Integer function, N.B. Pickering, 09/13/91
C  Calculates "TRUE" julian day, assuming that all leap years are 
C  divisible by 4. This is incorrect for certain years.
C-----------------------------------------------------------------------
C  09/13/1991 NBP Devfeloped
C  11/24/2002 GH  Modified for Y2K
C-----------------------------------------------------------------------
C  Input : YR,DOY
C  Output: DOC
C  Local : NLEAP
C=======================================================================

      INTEGER FUNCTION DOYC(YR,DOY)

      IMPLICIT NONE
      INTEGER DOY,NLEAP,YR

      NLEAP = INT((YR-1)/4)
      DOYC = NLEAP*366 + (YR-NLEAP-1)*365 + DOY

      END FUNCTION DOYC

C=======================================================================
C  YR_DOY, Subroutine, N.B. Pickering, 09/13/91
C  Converts YRDOY to YR and DOY.
C-----------------------------------------------------------------------
C  Input : YRDOY
C  Output: YR,DOY
C=======================================================================

      SUBROUTINE YR_DOY(YRDOY,YR,DOY)

      IMPLICIT NONE

      INTEGER DOY,YR,YRDOY

      YR  = INT(YRDOY / 1000)
      DOY = YRDOY - YR * 1000

      END SUBROUTINE YR_DOY

C=======================================================================
C  Y2K_DOY, Subroutine, Gerrit Hoogenboom 6/7/2002
C  Converts YRDOY to Y2K
C-----------------------------------------------------------------------
C  Input : YRDOY
C  Output: YRDOY
C=======================================================================

      SUBROUTINE Y2K_DOY(YRDOY)

      IMPLICIT NONE

      INTEGER DOY,YR,YRDOY

      IF (YRDOY .LE. 99365) THEN
        YR  = INT(YRDOY / 1000)
        DOY = YRDOY - YR * 1000
        IF (YRDOY .GT. 0) THEN
!     CHP 09/11/2009 - change "cross-over" year from 2010 to 2015
!     CHP 03/26/2014 - change "cross-over" year from 2015 to 2020
!     CHP 07/06/2017 - change "cross-over" year from 2020 to 2025
          IF (YR .LE. 25) THEN
            YRDOY = (2000 + YR) * 1000 + DOY
          ELSE
            YRDOY = (1900 + YR) * 1000 + DOY
          ENDIF
        ENDIF
      ENDIF
        
      END SUBROUTINE Y2K_DOY

C=======================================================================
C  Y2K_DOYW, Subroutine, C. Porter, 02/05/2004
C  Converts YRDOYW to Y2K for weather files
C  If this is a sequenced run, then days are forced to be sequential
C    when going from year 2010 to 2011,
C    and from 1999 to 2000.
! 09/11/2009 CHP changed "cross-over" year from 2010 to 2015
C-----------------------------------------------------------------------
C  Input : RNMODE, YRDOYWY, YRDOYW
C  Output: YRDOYW
C=======================================================================

      SUBROUTINE Y2K_DOYW(MULTI, YRDOYWY, YRDOYW, CENTURY)

      USE ModuleDefs
      IMPLICIT NONE

!      CHARACTER*1 RNMODE
      INTEGER MULTI   !, RUN
      INTEGER CENTURY,  DOY,  YEAR,  YR,  YRDOYW
      INTEGER CENTURYY, DOYY, YEARY, YRY, YRDOYWY !, YRINC

      DATA YRY /0/

      IF (MULTI .LE. 1) YRY = 0

      IF (YRDOYW .LE. 99365) THEN
        YR  = INT(YRDOYW / 1000)
        DOY = YRDOYW - YR * 1000

        IF (YRDOYW .GT. 0) THEN
!!         IF (YR .LE. 10 .OR. YRY .GE. 2000) THEN
!          IF (YR .LE. 15 .OR. YRY .GE. 2000) THEN
!            CENTURY = 20
!          ELSE
!            CENTURY = 19
!          ENDIF
          YEAR = CENTURY * 100 + YR
          YRDOYW = YEAR * 1000 + DOY
        ENDIF
      ELSE
        CALL YR_DOY(YRDOYW, YEAR, DOY)
        CENTURY = INT(YEAR / 100)
      ENDIF

      CALL YR_DOY(YRDOYWY, YEARY, DOYY)
      CENTURYY = INT(YEARY / 100)
    
!-----------------------------------
!!     RNMODE = Q CHECK RUN
!!     RNMODE = N CHECK MULTI
!!CHP 10/11/2005 Fix problem with sequenced runs spanning Y2K
!!CHP 10/9/06      IF (INDEX('QFNS',RNMODE) .GT. 0 .OR. MULTI .GT. 1) THEN
!      IF ((INDEX('QF',RNMODE) > 0 .AND. RUN   > 1) .OR.
!!     &    (INDEX('N', RNMODE) > 0 .AND. MULTI > 1)) THEN
!     &    (MULTI > 1)) THEN
!
!        YRINC = YEAR - YEARY
!!       IF (YRINC .EQ. 100 .OR. YRINC .EQ. 101) THEN
!        IF (YRINC .GT. 1) THEN
!          YEAR = YEAR - 100
!          YRDOYW = YEAR * 1000 + DOY
!        ELSEIF (YRINC .LT. 0) THEN
!          YEAR = YEAR + 100
!          YRDOYW = YEAR * 1000 + DOY
!        ENDIF
!      ENDIF 

!     10/10/2006 CHP
!     Fixes problem with model going from 2010 to 1911 during simulation
      IF (CENTURYY > CENTURY) THEN
        CENTURY = CENTURYY
        YEAR = CENTURY * 100 + YR
        YRDOYW = YEAR * 1000 + DOY

!     Fixes problem with crossing centuries
      ELSEIF (CENTURYY == CENTURY .AND. 
     &        YEAR < YEARY .AND. MOD(YEARY,100) == 99) THEN
        CENTURY = CENTURY + 1
        YEAR = CENTURY * 100 + YR
        YRDOYW = YEAR * 1000 + DOY
      ENDIF

      RETURN
      END SUBROUTINE Y2K_DOYW

C=======================================================================
C  YDOY, Integer Function, N.B. Pickering, 09/13/91
C  Converts YR and DOY to YRDOY.
C-----------------------------------------------------------------------
C  Input : YR,DOY
C  Output: YRDOY
C=======================================================================

      INTEGER FUNCTION YDOY(YR,DOY)

      IMPLICIT NONE
      INTEGER DOY,YR

      YDOY = YR * 1000 + DOY
      
      END FUNCTION YDOY

C=======================================================================
C  TIMDIF, Integer function, N.B. Pickering, 09/13/91
C  Calculates the time difference between two YRDOY dates (days).
C-----------------------------------------------------------------------
C  Input : YRDOY1,YRDOY2
C  Output: DAYDIF
C=======================================================================

      INTEGER FUNCTION TIMDIF(YRDOY1,YRDOY2)

      IMPLICIT NONE
      INTEGER DOYC,DOY1,DOY2,YR1,YR2,YRDOY1,YRDOY2

C     Simple time difference of two days in the same year attempted first.

      TIMDIF = YRDOY2 - YRDOY1

C     If time difference involves a year change, use DOC calculations.

      IF (TIMDIF .GT. 365 .OR. TIMDIF .LT. -365) THEN
        CALL YR_DOY(YRDOY1,YR1,DOY1)
        CALL YR_DOY(YRDOY2,YR2,DOY2)
        TIMDIF = DOYC(YR2,DOY2) - DOYC(YR1,DOY1)
      ENDIF

      END FUNCTION TIMDIF

C=======================================================================
C  MTHEND, Integer Function, N.B. Pickering, 06/05/92
C  Calculates day-of-year that is end of month.
C-----------------------------------------------------------------------
C  Input : MTH,YR
C  Output: MTHEND
C  Local : MEND,LEAPYR
C=======================================================================

      INTEGER FUNCTION MTHEND(YR,MTH)

      IMPLICIT NONE
      INTEGER MTH,MEND(12),YR
      LOGICAL LEAP
      DATA MEND/31,59,90,120,151,181,212,243,273,304,334,365/

      IF (LEAP(YR) .AND. MTH.GE.2) THEN
        MTHEND = MEND(MTH) + 1
      ELSE
        MTHEND = MEND(MTH)
      ENDIF

      END FUNCTION MTHEND

C=======================================================================
C  MTHMID, Integer Function, N.B. Pickering, 06/05/92
C  Calculates day-of-year that is midpoint of month.
C-----------------------------------------------------------------------
C  Input : MTH,YR
C  Output: MTHMID
C  Local : MAXDOY,MBEG,MEND,LEAPYR,MTHBEG,MTHEND
C=======================================================================

      INTEGER FUNCTION MTHMID(YR,MTH)

      IMPLICIT NONE
      INTEGER MTH,YR
      LOGICAL LEAP
      INTEGER MIDPT(12)
      DATA MIDPT/16,46,75,106,136,167,197,228,259,289,320,350/

      IF (LEAP(YR) .AND. MTH.GE.2) THEN
        MTHMID = MIDPT(MTH) + 1
      ELSE
        MTHMID = MIDPT(MTH)
      ENDIF

      END FUNCTION MTHMID

C=======================================================================
C  INCYD, Integer Function, N.B. Pickering, 06/05/92
C  Increases/decreases YRDOY based on INC (ABS(INC)<=365).
C-----------------------------------------------------------------------
C  Input : YRDOY
C  Output: INCYD
C  Local : YDEND,YR,DOY
C=======================================================================

      INTEGER FUNCTION INCYD(YRDOY,INC)

      IMPLICIT NONE
      INTEGER ENDYR,INC,NDYR,YRDOY,YR,DOY,YDOY

      CALL YR_DOY(YRDOY,YR,DOY)
      NDYR = ENDYR(YR)
      DOY = DOY + INC
      IF (DOY .GT. NDYR) THEN
        YR = YR + 1
        DOY = DOY - NDYR
      ELSE IF (DOY .LE. 0) THEN
        YR = YR - 1
        NDYR = ENDYR(YR)
        DOY = NDYR + DOY
      ENDIF
      INCYD = YDOY(YR,DOY)
      
      END FUNCTION INCYD

C=======================================================================
C  INCDAT, Integer Function,J.Hansen
C  Similar to INCYD without the restriction that DELTA <= 365.
C-----------------------------------------------------------------------
C  Input : YRDOY(ADATE)
C  Output: INCDAT
C  Local : NDYR,AYR,ADOY,DELTA,ENDYR,YDOY
C-----------------------------------------------------------------------

      INTEGER FUNCTION INCDAT(ADATE, DELTA)

      IMPLICIT NONE
      INTEGER NDYR, AYR, ADOY, ADATE, DELTA, ENDYR, YDOY
      EXTERNAL ENDYR, YDOY

      CALL YR_DOY(ADATE, AYR, ADOY)
      NDYR = ENDYR(AYR)
      ADOY = ADOY + DELTA
  100 CONTINUE
      IF (ADOY .GT. NDYR) THEN
        AYR = AYR + 1
        ADOY = ADOY - NDYR
        GO TO 100
      END IF
  200 IF (ADOY .LE. 0) THEN
        AYR = AYR - 1
        NDYR = ENDYR(AYR)
        ADOY = ADOY + NDYR
        GO TO 200
      END IF
      INCDAT = YDOY(AYR, ADOY)

      RETURN
      END FUNCTION INCDAT


C=======================================================================
C  INCMTH, Integer Function, N.B. Pickering, 06/05/92
C  Increases/decreases month and adjusts YR based on INC (ABS(INC)<=12).
C-----------------------------------------------------------------------
C  Input : YR,MTH,SIGN
C  Output: INCMTH
C  Local :
C=======================================================================

      SUBROUTINE INCMTH(YR,MTH,INC)

      INTEGER YR,MTH,INC

      MTH = MTH + INC
      IF (MTH .GT. 12) THEN
        YR = YR + 1
        MTH = MTH - 12
      ELSE IF (MTH .LT. 1) THEN
        YR = YR - 1
        MTH = MTH + 12
      ENDIF

      END SUBROUTINE INCMTH

C=======================================================================
C  YDEND, Integer Function, N.B. Pickering, 06/05/92
C  Computes end-of-year in YRDOY format.  Input can be YRDOY or YR.
C-----------------------------------------------------------------------
C  Input : YRDOY or YR
C  Output: YDEND
C  Local : DOY,NDYR,YR
C=======================================================================

      INTEGER FUNCTION YDEND(YRDOY)

      IMPLICIT NONE
      INTEGER ENDYR,YRDOY,YR,YDOY

      IF (YRDOY/1000 .NE. 0) THEN
        YR = YRDOY / 1000
      ELSE
        YR = YRDOY
      ENDIF
      YDEND = YDOY(YR,ENDYR(YR))

      END FUNCTION YDEND

C=======================================================================
C  ENDYR, Integer Function, N.B. Pickering, 06/05/92
C  Computes end-of-year (365 or 366) depending if leap year.
C-----------------------------------------------------------------------
C  Input : YR
C  Output: ENDYR
C  Local :
C=======================================================================

      INTEGER FUNCTION ENDYR(YR)

      INTEGER YR
      LOGICAL LEAP

      IF (LEAP(YR)) THEN; ENDYR = 366
      ELSE;               ENDYR = 365
      ENDIF

      END FUNCTION ENDYR

C=======================================================================
C  JULIAN, Function
C
C  Determines day, month
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2  Modified by
C  3. Header revision and minor changes             P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : NDAY,RMON,NYRCHK
C
C  LOCAL  : MonthTxt,UPCASE,I,JCOUNT,DAYS
C
C  OUTPUT : JULIAN
C-----------------------------------------------------------------------
C  Called : SEHARV SEPLT SETIME
C
C  Calls  : None
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  RMON   :
C  MonthTxt():
C  UPCASE :
C  I      :
C  JCOUNT :
C  NDAY   :
C  NYRCHK :
C  DAYS()  :
C=======================================================================

      INTEGER FUNCTION JULIAN (NDAY,RMON,YR)

      USE ModuleDefs
      IMPLICIT    NONE

      CHARACTER*3 RMON    !,MonthTxt(12)
      CHARACTER*1 UPCASE

      INTEGER     I,JCOUNT,NDAY,YR,DAYS(12)
      LOGICAL     LEAP

      DATA DAYS   /31,28,31,30,31,30,31,31,30,31,30,31/
!      DATA MonthTxt /'JAN','FEB','MAR','APR','MAY','JUN',
!     &            'JUL','AUG','SEP','OCT','NOV','DEC'/

      IF (LEAP(YR)) THEN; DAYS(2) = 29
      ELSE; DAYS(2) = 28
      ENDIF

      DO I = 1, 3
         RMON(I:I) = UPCASE(RMON(I:I))
      END DO
      DO JCOUNT = 1, 12
         IF (RMON .EQ. MonthTxt(JCOUNT)) GO TO 200
      END DO

C-----------------------------------------------------------------------
C     Month name cannot be recognized
C-----------------------------------------------------------------------

      JULIAN = 370
      RETURN
  200 JULIAN = 0

      DO 400 JCOUNT = 1, 12
           IF (RMON .EQ. MonthTxt(JCOUNT))
     &          GO TO 300
           JULIAN = JULIAN + DAYS(JCOUNT)
           GO TO 400
  300      JULIAN = JULIAN + NDAY
           RETURN
  400 CONTINUE

      END FUNCTION JULIAN


C=======================================================================
C  NAILUJ, Subroutine
C
C  Determines Julian date
C-----------------------------------------------------------------------
C  Revision history
C
C  1. Written
C  2  Modified by
C  3. Header revision and minor changes             P.W.W.      5-28-93
C-----------------------------------------------------------------------
C  INPUT  : JULD,NYRCHK
C
C  LOCAL  : MonthTxt(),NSUM,JCOUNT,NDIF,DAYS()
C
C  OUTPUT : NDAY RMON
C-----------------------------------------------------------------------
C  Called : SEHARV SENS SEPLT SETIME OPDAY OPHEAD
C
C  Calls  : None
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  RMON   :
C  MonthTxt():
C  NSUM   :
C  JCOUNT :
C  NDIF   :
C  JULD   :
C  NYRCHK :
C  NDAY   :
C  DAYS()  :
C=======================================================================

      SUBROUTINE NAILUJ (JULD,YR,RMON,NDAY)

      USE ModuleDefs
      IMPLICIT    NONE

      CHARACTER*3 RMON    !,MonthTxt(12)
      INTEGER     NSUM,JCOUNT,NDIF,JULD,YR,NDAY,DAYS(12)
      LOGICAL     LEAP

      DATA DAYS   /31,28,31,30,31,30,31,31,30,31,30,31/
!      DATA MonthTxt /'JAN','FEB','MAR','APR','MAY','JUN',
!     &            'JUL','AUG','SEP','OCT','NOV','DEC'/

      IF (LEAP(YR)) THEN; DAYS(2) = 29
      ELSE; DAYS(2) = 28
      ENDIF

      NSUM = 0

      DO JCOUNT = 1, 12
         NDIF = JULD - NSUM
         IF (NDIF .LE. DAYS(JCOUNT)) THEN
            NDAY = NDIF
            RMON = MonthTxt(JCOUNT)
            RETURN
         ENDIF
         NSUM = NSUM + DAYS(JCOUNT)
      END DO

      RETURN

      END SUBROUTINE NAILUJ



C=======================================================================
C  FullYear, Subroutine
C
C  Converts 2 digit year to 4 digit year. On first call per simulation, 
C    assumes that years 10 and lower refer to 2010 and years 11 and higher
C    refer to 1911.  After the first call, years are converted based on
C    previous years (i.e., year 2011 follows year 2010.)
C    
C-----------------------------------------------------------------------
C  Revision history
C
C  02/07/2002 CHP Written
C  11/24/2002 GH  Modified Y2K to 10
C=======================================================================
      SUBROUTINE FullYear (YRDOY, YEAR, DOY)

      IMPLICIT    NONE

      INTEGER CENTURY, DOY
      INTEGER YEAR, YRDOY, YR
      LOGICAL FIRST
      DATA FIRST /.TRUE./

      IF (YRDOY .LE. 99365) THEN
        YR  = INT(YRDOY / 1000)
        DOY = YRDOY - YR * 1000

        IF (FIRST) THEN
          IF (YR .LE. 20) THEN
            CENTURY = 2000
          ELSE
            CENTURY = 1900
          ENDIF
          FIRST = .FALSE.
        ENDIF

!       This will work if the YRDOY after 99365 is 100001.
         YEAR = CENTURY + YR

      ENDIF 
      RETURN
      END SUBROUTINE FullYear

C=======================================================================
C  ETAD_NAILUJ, Subroutine
C
C  Determines Julian date, exports integer year, month, day
C-----------------------------------------------------------------------
C  Revision history
C
C  11/29/2004 CHP Written based on NAILUJ
C-----------------------------------------------------------------------
C  INPUT  : JULD
C  OUTPUT : YR, iMON, NDAY
C=======================================================================

      SUBROUTINE ETAD_NAILUJ (JULD, YR, iMON, NDAY)

      IMPLICIT    NONE

      INTEGER     NSUM,NDIF,JULD,YR,NDAY,DAYS(12), iMON
      LOGICAL     LEAP

      DATA DAYS   /31,28,31,30,31,30,31,31,30,31,30,31/

      IF (LEAP(YR)) THEN; DAYS(2) = 29
      ELSE;               DAYS(2) = 28
      ENDIF
   
      NSUM = 0

      DO iMON = 1, 12
         NDIF = JULD - NSUM
         IF (NDIF .LE. DAYS(iMON)) THEN
            NDAY = NDIF
            RETURN
         ENDIF
         NSUM = NSUM + DAYS(iMON)
      END DO

      RETURN

      END SUBROUTINE ETAD_NAILUJ

C=======================================================================

C=======================================================================
C  LEAP, Function
C
C  Determines leap years
C-----------------------------------------------------------------------
C  Revision history
C
C  11/16/2007 CHP Written 
C-----------------------------------------------------------------------
C  INPUT  : YR
C  OUTPUT : LEAP
C=======================================================================

      LOGICAL FUNCTION LEAP (YR)

      IMPLICIT    NONE
      INTEGER YR

      IF     (MOD(YR,400) == 0) THEN; LEAP = .TRUE.
      ELSEIF (MOD(YR,100) == 0) THEN; LEAP = .FALSE.
      ELSEIF (MOD(YR,  4) == 0) THEN; LEAP = .TRUE.
      ELSE;                           LEAP = .FALSE.
      ENDIF
   
      RETURN
      END FUNCTION LEAP

C=======================================================================




C=======================================================================
C  OPHEAD, Subroutine
C
C  Prints inputs for the simulation to the screen
C-----------------------------------------------------------------------
C  Revision history
C
C  01/01/1993 GH  Written
C  05/28/1993 PWW Header revision and minor changes
C  05/28/1993 PWW Added switch block, etc.
C  04/01/1996 GH  Added comsoi.blk and comibs.blk
C  12/11/2000 GH  Minor output fixes due to 80 character stringlimit
C  01/29/2002 CHP OPHEAD writes to generic HEADER.OUT file for use
C                   by model output routines.
C  08/19/2002 GH  Modified for Y2K
C  05/08/2003 CHP Added version information and date-time stamp to header
!  09/12/2007 CHP Save to array instead of write to file.
C
C-----------------------------------------------------------------------
C  INPUT  : WSTA,PEDON,TRTNO,CUMDEP,NFERT,TOTNAP,NARES,RESAMT,NAPW,TOTAPW,
C           SLDESC,TPESW,ROWSPC,PLTPOP,VRNAME,AINO3,AINH4,YEAR,SLTX,LUNOV,
C           YRSIM,YRPLT,SOILNX,DSOILN,SOILNC,ECONAM,RUN,MODEL,
C           EXPER,CROP,CROPD,DSOIL,THETAC,TITLET
C
C  LOCAL  :
C
C  OUTPUT :
C-----------------------------------------------------------------------
C  Called : OPDAY
C
C  Calls  : YR_DOY NAILUJ WTHSUM
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  HDLAY  :
C=======================================================================

      SUBROUTINE OPHEAD (DYNAMIC, LUNOV,CUMDEP,TPESW,VRNAME,AINO3,AINH4,
     &     ECONO,RUN,MODEL,TITLET,WTHSTR, RNMODE,
     &     CONTROL, ISWITCH, UseSimCtr, PATHEX)

      USE ModuleDefs
      USE HeaderMod
      IMPLICIT NONE
      SAVE

      INCLUDE 'COMSWI.blk'
      INCLUDE 'COMSOI.blk'
      INCLUDE 'COMIBS.blk'

      CHARACTER*1   RNMODE
      CHARACTER*3   RMP,RMS
      CHARACTER*6   ECONO
      CHARACTER*8   MODEL
      CHARACTER*16  VRNAME
      CHARACTER*25  TITLET
      CHARACTER*80  PATHEX
      CHARACTER*80  HEADER(100) !Simulation header
      CHARACTER*120 WTHSTR

      INTEGER       DYNAMIC
      INTEGER       ICOUNT, I, IDYP,IDYS,IPYRP,IPYRS,NNFERT
      INTEGER       NNAPW,ISIM,IPLT,LUNOV,RUN
      INTEGER       SimLen, LenString

      REAL          AINH4,AINO3,CUMDEP,TPESW

c     MJ, Mar 2008: added HDATE_YR and HDATE_DOY
      INTEGER HDATE_YR, HDATE_DOY

      TYPE (ControlType) CONTROL
      TYPE (SwitchType)  ISWITCH

      INTEGER DATE_TIME(8)
      LOGICAL UseSimCtr

!     Write header to console when LUNOV=6
      IF (LUNOV == 6) THEN
        DO I = 1, HEADERS%ICOUNT
          WRITE(LUNOV,'(A)') HEADER(I)
        ENDDO 
        RETURN
      ENDIF

	 IF (TITLER(1:5) .EQ. '     ') THEN
         TITLER = TITLET
       ENDIF
	
!     ******************************************************************
!     ******************************************************************
!     Generate short header 
      IF (DYNAMIC == RUNINIT) THEN
!     ------------------------------------------------------------------
      CALL DATE_AND_TIME (VALUES=DATE_TIME)
!      date_time(1)  The 4-digit year  
!      date_time(2)  The month of the year  
!      date_time(3)  The day of the month  
!      date_time(4)  The time difference with respect to Coordinated Universal Time (UTC) in minutes  
!      date_time(5)  The hour of the day (range 0 to 23) - local time  
!      date_time(6)  The minutes of the hour (range 0 to 59) - local time  
!      date_time(7)  The seconds of the minute (range 0 to 59) - local time  
!      date_time(8)  The milliseconds of the second (range 0 to 999) - local time  
      
!     Version information stored in ModuleDefs.for
      WRITE (HEADER(1),100) Version, VBranch, MonthTxt(DATE_TIME(2)),
     &    DATE_TIME(3), DATE_TIME(1), DATE_TIME(5), DATE_TIME(6), 
     &    DATE_TIME(7)
      WRITE (HEADER(2),'(" ")')
!     WRITE (HEADER(3),200) MOD(RUN,1000),TITLET
!     WRITE (HEADER(3),201) MOD(RUN,1000),TITLET,MODEL

      WRITE (HEADER(3),201) MOD(RUN,1000),TITLET,MODEL,EXPER,TRTNO
  201 FORMAT ('*RUN ',I3,8X,': ',A25,1X,A8,1X,A8,I5)

!      IF (TRTNO.LT.10) THEN
!        WRITE (HEADER(3),201) MOD(RUN,1000),TITLET,MODEL
!      ELSEIF (TRTNO.GE.10.AND.TRTNO.LT.100) THEN
!        WRITE (HEADER(3),202) MOD(RUN,1000),TRTNO,TITLET,EXPER,MODEL
!      ELSEIF (TRTNO.GE.100.AND.TRTNO.LT.1000) THEN
!        WRITE (HEADER(3),203) MOD(RUN,1000),TRTNO,TITLET,EXPER,MODEL
!      ELSE
!        WRITE (HEADER(3),204) MOD(RUN,1000),TRTNO,TITLET,EXPER,MODEL
!      ENDIF
!  201 FORMAT ('*RUN ',I3,2X,'Tr ',I1,'  : ',A25,1X,A8,1X,A8)
!  202 FORMAT ('*RUN ',I3,2X,'Tr ',I2,' : ',A25,1X,A8,1X,A8)
!  203 FORMAT ('*RUN ',I3,2X,'Tr ',I3,': ',A25,1X,A8,1X,A8)
!  204 FORMAT ('*RUN ',I3,2X,'Tr',I4,': ',A25,1X,A8,1X,A8)

      I = INDEX(MODEL," ")
      IF (I < 1) THEN
        WRITE (HEADER(4),300) MODEL(1:8),CROPD
      ELSE
        WRITE(HEADER(4),*) " "
      ENDIF
      WRITE (HEADER(5),310) EXPER,CG,ENAME(1:50)
      WRITE (HEADER(6),312) PATHEX(1:62)
  312 FORMAT (1X,'DATA PATH ',5X,':',1X,A)
      IF (INDEX('FQ',RNMODE) > 0) THEN
        WRITE (HEADER(7),320) MOD(CONTROL%ROTNUM,1000),TITLET, MODEL
      ELSE
        WRITE (HEADER(7),320) MOD(TRTNO,1000),TITLET, MODEL
      ENDIF
      WRITE (HEADER(8),'(" ")')
      I = 9; HEADERS%ShortCount = I-2
      HEADERS % ICOUNT  = I-1
      HEADERS % HEADER = HEADER
      HEADERS % RUN    = RUN

!     ******************************************************************
!     ******************************************************************
      ELSE
!     Continue with long header
!     ------------------------------------------------------------------
      ICO2   = ISWITCH % ICO2
      IFERI  = ISWITCH % IFERI
      IHARI  = ISWITCH % IHARI
      IIRRI  = ISWITCH % IIRRI
      IPLTI  = ISWITCH % IPLTI
      IRESI  = ISWITCH % IRESI
      ISWNIT = ISWITCH % ISWNIT
      ISWPHO = ISWITCH % ISWPHO
      ISWSYM = ISWITCH % ISWSYM
      ISWTIL = ISWITCH % ISWTIL
      ISWWAT = ISWITCH % ISWWAT
      MEEVP  = ISWITCH % MEEVP
      MEHYD  = ISWITCH % MEHYD
      MEINF  = ISWITCH % MEINF
      MEPHO  = ISWITCH % MEPHO
      MESEV  = ISWITCH % MESEV
      MESOL  = ISWITCH % MESOL
      METMP  = ISWITCH % METMP
      MEWTH  = ISWITCH % MEWTH
      NSWITCH= ISWITCH % NSWI

      IF (UseSimCtr) THEN
        SimLen = LenString(CONTROL % SimControl)
        WRITE (HEADER(I),1105); I=I+1
        WRITE (HEADER(I),1106) CONTROL % SimControl(1:SimLen); I=I+1
        WRITE (HEADER(I),1107); I=I+1; ; HEADERS%ShortCount = I-1
 1105   FORMAT("!Simulation control file used for this simulation.")
 1106   FORMAT("!File: ",A)
 1107   FORMAT("!See top of WARNING.OUT file for specific controls.")
      ENDIF

      WRITE (HEADER(I),'(" ")'); I=I+1
      WRITE (HEADER(I),330) CROPD,VRNAME,ECONO; I=I+1

      CALL YR_DOY (CONTROL%YRSIM,IPYRS,ISIM)
      CALL NAILUJ (ISIM,IPYRS,RMS,IDYS)
      WRITE (HEADER(I),400) RMS,IDYS,IPYRS; I=I+1
      CALL YR_DOY (YRPLT,IPYRP,IPLT)

c     MJ, Mar 2008: output the line with planting date, density and rowspacing
      IF (IPLT .LE. 366 .AND. IPLTI .EQ. 'R' ) THEN
         CALL NAILUJ (IPLT,IPYRP,RMP,IDYP)
         WRITE (HEADER(I),450) RMP,IDYP,IPYRP,PLTPOP,ROWSPC; I=I+1
       ELSE
         WRITE (HEADER(I),475) PLTPOP,ROWSPC; I=I+1
      ENDIF

c     MJ, Mar 2008: output harvest date in HEADER.OUT
c     (OVERVIEW.OUT).  Only do this if harvest is on a 
c     'reported date'.
c     IHARI is harvest management option 
c     (e.g. 'R' = 'On reported date', 'M' = 'At maturity')
c     HDATE(1) is harvest date (YYYYDOY)
c     ::::::::::::::::::::::::::::::::::
      IF (IHARI  .EQ. 'R' .AND. INDEX('FQN',RNMODE)<1) THEN

c         Get Day Of Year (DOY) date
          HDATE_YR = HDATE(1)/1000
          HDATE_DOY = HDATE(1) - (HDATE_YR*1000)
          CALL NAILUJ (HDATE_DOY,HDATE_YR,RMS,IDYS)

c         Write to output
          WRITE (HEADER(I),425) RMS,IDYS,HDATE_YR; I=I+1
      ENDIF
c     ::::::::::::::::::::::::::::::::::

c     MJ, Mar 2008: weather station and year:
      WRITE (HEADER(I),500) WSTA, YEAR; I=I+1

c     MJ, Mar 2008: Soil information
      WRITE (HEADER(I),600) PEDON,SLTX,SLDESC; I=I+1

      IF (ISWWAT .NE. 'N') THEN
!        CHP 08/12/2005 Don't report initial conditions for
!             sequenced runs.
         IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN == 1) THEN
           WRITE (HEADER(I),625) NINT(CUMDEP),TPESW*10,AINO3,AINH4
           I=I+1
         ENDIF

         IF (IIRRI .EQ. 'R' .OR. IIRRI .EQ. 'D') THEN
            IF (IIRRI .EQ. 'R') THEN
               WRITE (HEADER(I),650)
             ELSE IF (IIRRI .EQ. 'D') THEN
               WRITE (HEADER(I),655)
            ENDIF
            I=I+1
            IF (TOTAPW .EQ. 0 .AND. NAPW .GE. 1) THEN
               NNAPW = NAPW
             ELSE
               NNAPW = NAPW
            ENDIF
            WRITE (HEADER(I),660) NINT(TOTAPW),NNAPW; I=I+1
          ELSE IF (IIRRI .EQ. 'A') THEN
            WRITE (HEADER(I),665); I=I+1
            WRITE (HEADER(I),666) DSOIL/100.,THETAC; I=I+1
          ELSE IF (IIRRI .EQ. 'F') THEN
            WRITE (HEADER(I),670); I=I+1
            WRITE (HEADER(I),666) DSOIL/100.,THETAC; I=I+1
          ELSE IF (IIRRI .EQ. 'E') THEN
            WRITE (HEADER(I),675); I=I+1
            WRITE (HEADER(I),676) DSOIL; I=I+1
          ELSE IF (IIRRI .EQ. 'T') THEN
            WRITE (HEADER(I),680); I=I+1
            WRITE (HEADER(I),676) DSOIL; I=I+1
          ELSE IF (IIRRI .EQ. 'N') THEN
            WRITE (HEADER(I),690); I=I+1
            WRITE (HEADER(I),691); I=I+1
         ENDIF

  660 FORMAT(' IRRIGATION     : ',I8,' mm IN ',I5,' APPLICATIONS')
  665 FORMAT(' WATER BALANCE  : AUTOMATIC IRRIGATION - REFILL PROFILE')
  666 FORMAT(' IRRIGATION     : AUTOMATIC [SOIL DEPTH:',F5.2,' m',1X,
     &            F3.0,'%]')
  670 FORMAT(' WATER BALANCE  : AUTOMATIC IRRIGATION - FIXED AMOUNT')
  675 FORMAT(' WATER BALANCE  : ET AUTO IRRIGATION - REFILL PROFILE')
  676 FORMAT(' IRRIGATION     : AUTOMATIC [ET ACCUM:',F5.2,' mm ]')
  680 FORMAT(' WATER BALANCE  : ET AUTOMATIC IRRIGATION - FIXED AMOUNT')
  690 FORMAT(' WATER BALANCE  : RAINFED')
  691 FORMAT(' IRRIGATION     : NOT IRRIGATED')

      ELSE IF (ISWWAT .EQ. 'N') THEN
         WRITE (HEADER(I),705); I=I+1
         WRITE (HEADER(I),710); I=I+1
      ENDIF



      IF (ISWNIT .EQ. 'Y') THEN
         IF (ISWSYM .EQ. 'Y') THEN
            WRITE (HEADER(I),720); I=I+1
          ELSE IF (ISWSYM .EQ. 'U') THEN
            WRITE (HEADER(I),730); I=I+1
          ELSE IF (ISWSYM .EQ. 'N') THEN
            WRITE (HEADER(I),740); I=I+1
         ENDIF
       ELSE
         WRITE (HEADER(I),750); I=I+1
         WRITE (HEADER(I),751); I=I+1
         WRITE (HEADER(I),752); I=I+1
      ENDIF

      IF (ISWNIT .EQ. 'Y') THEN
         IF (IFERI .EQ. 'R' .OR. IFERI .EQ. 'D') THEN
            IF (TOTNAP .EQ. 0 .AND. NFERT .GE. 1) THEN
               NNFERT = 0
             ELSE
               NNFERT = NFERT
            ENDIF
            WRITE (HEADER(I),800) NINT(TOTNAP),NNFERT; I=I+1
!         10/14/2008 CHP added "F" option
          ELSE IF (IFERI .EQ. 'A' .OR. IFERI .EQ. 'F') THEN
            WRITE (HEADER(I),810) SOILNX,DSOILN,SOILNC; I=I+1
          ELSE IF (IFERI .EQ. 'N') THEN
            WRITE (HEADER(I),820); I=I+1
         ENDIF
         WRITE (HEADER(I),1000) NINT(ICRES),NINT(RESAMT),NARES; I=I+1
      ENDIF

      IF (LenString(WTHSTR) > 1) THEN
        WRITE (HEADER(I),1200) WTHSTR(1:60); I=I+1
        WRITE (HEADER(I),1210) WTHSTR(61:120); I=I+1
      ENDIF
      WRITE (HEADER(I),1300) ISWWAT,ISWNIT,ISWSYM,ISWPHO,ISWDIS; I=I+1
      WRITE (HEADER(I),1350) MEPHO,MEEVP,MEINF,MEHYD,MESOM; I=I+1
      WRITE (HEADER(I),1355) ICO2, NSWITCH, MESEV, MESOL, METMP; I=I+1
      WRITE (HEADER(I),1400) IPLTI,IIRRI,IFERI,IRESI,IHARI; I=I+1
      WRITE (HEADER(I),1405) MEWTH, ISWTIL; I=I+1


      ICOUNT = I-1
      Headers % ICOUNT  = ICOUNT
      Headers % Header = HEADER
      Headers % RUN    = RUN

!     ******************************************************************
!     ******************************************************************
      ENDIF
C-----------------------------------------------------------------------
C     Format Strings
C-----------------------------------------------------------------------

  100 FORMAT ("*DSSAT Cropping System Model Ver. ",I1,".",I1,".",I1,
     &    ".",I3.3,1X,A10,4X,
     &    A3," ",I2.2,", ",I4,"; ",I2.2,":",I2.2,":",I2.2)
  200 FORMAT ('*RUN ',I3,8X,': ',A25)
! 201 FORMAT ('*RUN ',I3,8X,': ',A25,1X,A8)

  300 FORMAT (1X,'MODEL',10X,':',1X,A8,' - ',A16)
  310 FORMAT (1X,'EXPERIMENT',5X,':',1X,A8,1X,A2,1X,A50)
  320 FORMAT (1X,'TREATMENT',I3, 3X,':',1X,A25,1X,A8)
  330 FORMAT (1X,'CROP',11X,':',1X,A16,1X,'CULTIVAR :',A17,1X,
     &        'ECOTYPE :',A6)

  400 FORMAT (1X,'STARTING DATE  :',1X,A3,1X,I2,1X,I4)
  425 FORMAT (1X,'HARVEST DATE   :',1X,A3,1X,I2,1X,I4)
  450 FORMAT (1X,'PLANTING DATE  :',1X,A3,1X,I2,1X,I4,8X,
     &       'PLANTS/m2 :',F5.1,5X,'ROW SPACING :',F5.0,'cm ')
  475 FORMAT (1X,'PLANTING DATE  :',1X,'AUTOMATIC PLANTING',1X,
     &       'PLANTS/m2 :',F5.1,5X,'ROW SPACING :',F5.0,'cm ')
  500 FORMAT (1X,'WEATHER',8X,':',1X,A4,3X,I4)
  600 FORMAT (1X,'SOIL',11X,':',1X,A10,5X,'TEXTURE : ',A5,' - ',A25)
  625 FORMAT (1X,'SOIL INITIAL C ',':',1X,'DEPTH:',I3,'cm',1X,
     &     'EXTR. H2O:',F5.1,'mm  NO3:',F5.1,'kg/ha  NH4:',F5.1,'kg/ha')
  650 FORMAT (1X,'WATER BALANCE',2X,':',1X,'IRRIGATE ON',
     &           ' REPORTED DATE(S)')
  655 FORMAT (1X,'WATER BALANCE',2X,':',1X,'IRRIGATE ON REPORTED',
     &           ' DAP')
  705 FORMAT (1X,'WATER BALANCE',2X,':',1X,'NOT SIMULATED ;',
     &           ' NO H2O-STRESS')
  710 FORMAT (1X,'IRRIGATION',5X,':')
  720 FORMAT (1X,'NITROGEN BAL.',2X,':',1X,
     &           'SOIL-N, N-UPTAKE & DYNAMIC N-FIXATION SIMULATION')
  730 FORMAT (1X,'NITROGEN BAL.',2X,':',1X,
     &           'SOIL-N, N-UPTAKE & UNLIMITED N-FIXATION SIMULATION')
  740 FORMAT (1X,'NITROGEN BAL.',2X,':',1X,
     &           'SOIL-N & N-UPTAKE SIMULATION; NO N-FIXATION')
  750 FORMAT (1X,'NITROGEN BAL.',2X,':',1X,
     &           'NOT SIMULATED ; NO N-STRESS')
  751 FORMAT (1X,'N-FERTILIZER',3X,':')
  752 FORMAT (1X,'RESIDUE/MANURE',1X,':')
  800 FORMAT (1X,'N-FERTILIZER',3X,':',1X,I8,' kg/ha IN ',I5,
     &           ' APPLICATIONS')
  810 FORMAT (1X,'N-FERTILIZER',3X,':',1X,'AUTO APPLICATIONS ',F5.0,
     &           ' kg/ha AT ',F6.2,' cm AND',F5.2,' % STRESS')
  820 FORMAT (1X,'N-FERTILIZER',3X,':',1X,'NO N-FERTILIZER APPLIED')
 1000 FORMAT (1X,'RESIDUE/MANURE',1X,':',1X,'INITIAL : ',I5,' kg/ha ;',
     &        I8,' kg/ha IN ',I5,
     &           ' APPLICATIONS')

 1200 FORMAT (1X,'ENVIRONM. OPT. :',1X,A60)
 1210 FORMAT (18X,A60)

 1300 FORMAT (1X,'SIMULATION OPT : WATER',3X,':',A1,2X,'NITROGEN:',A1,
     & 2X,'N-FIX:',A1,2X,'PHOSPH :',A1,2X,'PESTS  :',A1)
 1350 FORMAT (18X,'PHOTO',3X,':',A1,2X,'ET      :',A1,
     & 2X,'INFIL:',A1,2X,'HYDROL :',A1,2X,'SOM    :',A1)
 1355 FORMAT (18X,'CO2  ',3X,':',A1,2X,'NSWIT   :',I1,
     & 2X,'EVAP :',A1,2X,'SOIL   :',A1,2X,'STEMP  :',A1)

 1400 FORMAT (1X,'MANAGEMENT OPT : PLANTING:',A1,2X,'IRRIG',3X,':',A1,
     & 2X,'FERT :',A1,2X,'RESIDUE:',A1,2X,'HARVEST:',A1,2X)
 1405 FORMAT (18X,'WEATHER :',A1,2X,'TILLAGE :',A1)

      RETURN
      END SUBROUTINE OPHEAD
C=======================================================================

C=======================================================================
C  OPSOIL, Subroutine

C  Generates output for soil data
C-----------------------------------------------------------------------
C  Revision history

C  01/01/1990 GH  Written
C  05/28/1999 PWW Header revision and minor changes 
C  03/11/2005 GH  Remove ANS, RNMODE and NYRS
!  02/06/2007 CHP Added alternate sugarcane parameters for CASUPRO
!  11/26/2007 CHP THRESH, SDPRO, SDLIP moved from eco to cul file
C  08/09/2012 GH  Updated for cassava
C-----------------------------------------------------------------------
C  INPUT  : IDETO,NOUTDO,NYRS,LL,DUL,SAT,DLAYR,SWINIT,DS,NLAYR,ESW
C           SHF,BD,PH,INO3,INH4,OC,TLL,TDUL,TSAT,TPESW,TSWINI,AINO3,AINH4
C           TSOC,SWCON,U,SALB,CN2,CROPD,VRNAME,VARTY,SLPF,ECONO
C           SLNF,LUNOV,CROP

C  LOCAL  :

C  OUTPUT :
C-----------------------------------------------------------------------
C  Called : OPIBS3
C
C  Calls  : CLEAR
C-----------------------------------------------------------------------
C                         DEFINITIONS
C
C  HDLAY  :
C=======================================================================

      SUBROUTINE OPSOIL (LL,DUL,SAT,
     &   DLAYR,SWINIT,DS,NLAYR,ESW,SHF,BD,PH,INO3,INH4,OC,
     &   TLL,TDUL,TSAT,TPESW,TSWINI,AINO3,AINH4,TSOC,
     &   SWCON,U,SALB,CN2,CROPD,VRNAME,VARTY,SLPF,
     &   ECONO,SLNF,CROP, RNMODE, RUN, MODEL, ISWITCH, ATLINE)

      USE ModuleDefs
      USE HeaderMod
      IMPLICIT NONE

      INCLUDE 'COMGEN.blk'

      CHARACTER*1  ISWWAT, RNMODE
      CHARACTER*2  CROP
      CHARACTER*6  VARTY,ECONO
      CHARACTER*8  MODEL
      CHARACTER*16 CROPD
      CHARACTER*16 VRNAME
      CHARACTER*80  HEADER(100) !Simulation header
      CHARACTER*1000 ATLINE

      INTEGER      NLAYR,I,L, RUN
      INTEGER      J, J1, J2, LENGTH, LENSTRING

      REAL         LL(NL),DUL(NL),SAT(NL),DLAYR(NL),DS(NL),SWINIT(NL)
      REAL         ESW(NL),SHF(NL),BD(NL),PH(NL),INO3(NL),INH4(NL)
      REAL         OC(NL),TLL,TDUL,TSAT,TPESW,TSWINI,AINO3,AINH4,TSOC
      REAL         SWCON,U,SALB,CN2,SLPF,SLNF

      TYPE (SwitchType) ISWITCH

      I = HEADERS % ICOUNT + 1
      HEADER = HEADERS % Header

      ISWWAT = ISWITCH % ISWWAT

!=======================================================================
!     SOILS
!-----------------------------------------------------------------------
!      CHP 08/12/2005 Don't report initial conditions for
!           sequenced runs.
      IF (INDEX('FQ',RNMODE) .LE. 0 .OR. RUN == 1) THEN

        WRITE (HEADER(I),'(" ")'); I=I+1
        WRITE (HEADER(I),300); I=I+1
  300   FORMAT ('*SUMMARY OF SOIL AND GENETIC INPUT PARAMETERS')
        WRITE (HEADER(I),'(" ")'); I=I+1
!-----------------------------------------------------------------------
!       Write soils info
        IF (ISWWAT .NE. 'N') THEN
          WRITE(HEADER(I),360); I=I+1
          WRITE(HEADER(I),361); I=I+1
          WRITE(HEADER(I),362); I=I+1
          WRITE(HEADER(I),363); I=I+1

          DO L = 1, NLAYR
            WRITE (HEADER(I),410) NINT(DS(L)-DLAYR(L)),NINT(DS(L)),
     &        LL(L),DUL(L),SAT(L),ESW(L),SWINIT(L),SHF(L),BD(L),
     &        PH(L),INO3(L),INH4(L),OC(L)
            I=I+1
          ENDDO
          WRITE (HEADER(I),'(" ")') ; I=I+1
          WRITE (HEADER(I),610) NINT(DS(NLAYR)),TLL,TDUL,TSAT,TPESW,
     &                        TSWINI,AINO3,AINH4,NINT(TSOC); I=I+1
          WRITE (HEADER(I),710) SALB,U,SLNF; I=I+1
          WRITE (HEADER(I),711) CN2,SWCON,SLPF; I=I+1
          WRITE (HEADER(I),'(" ")') ; I=I+1
        ENDIF

      ELSE
        WRITE(HEADER(I),310); I=I+1
  310   FORMAT ('*SUMMARY OF GENETIC INPUT PARAMETERS')
        WRITE(HEADER(I),'(" ")'); I=I+1
      ENDIF

!=======================================================================
!     GENOTYPE
!-----------------------------------------------------------------------
!     Write genetic coefficients
      WRITE (HEADER(I),800) CROPD(1:10),VARTY,VRNAME,ECONO; I=I+1

      SELECT CASE (MODEL(1:5))

!-----------------------------------------------------------------------
!     CROPGRO
      CASE ('CRGRO','PRFRM')
!      IF (INDEX (MODEL, 'CRGRO') > 0) THEN
        IF (INDEX ('BN,PN,SB,PE,CH,PP,VB,CP,BR,FB,NP,GB,PE,LT',CROP) 
     &    > 0) THEN
           WRITE (HEADER(I), 850) CSDVAR,PPSEN,PH2T5,
     &                        PHTHRS(8),PHTHRS(10); I=I+1
           WRITE (HEADER(I),851) WTPSD,SDPDVR,SFDUR,PODUR,XFRUIT; I=I+1

        ELSEIF (INDEX ('TM,PR,CB,CO,CT,CN',CROP) .GT. 0) THEN
           WRITE (HEADER(I), 850) CSDVAR,PPSEN,PH2T5,
     &                        PHTHRS(8),PHTHRS(10); I=I+1
           WRITE (HEADER(I),852) WTPSD,SDPDVR,SFDUR,PODUR,XFRUIT; I=I+1
        ENDIF

        WRITE (HEADER(I),853) THRESH, SDPRO, SDLIP; I=I+1

!-----------------------------------------------------------------------
!     CSCER - Wheat, barley
      CASE ('CSCER')
        WRITE (HEADER(I),'(A)')
     &    "   P1V   P1D    P5    G1    G2    G3 PHINT"
        I=I+1
        WRITE (HEADER(I),'(5(1X,F5.1),1X,F5.2,1X,F5.1)') 
     &    P1V,P1D,P5,G1,G2,G3,PHINT
        I=I+1

!       Print optional extra stuff from ecotype file
        LENGTH = LenString(PLAINTXT)
        IF (LENGTH > 0) THEN
          DO J=1,5
            J1 = (J-1)*78+1
            J2 = J1 + 77
            WRITE( HEADER(I),'(A)') TRIM(ATLINE(J1+29:J2+29))
            WRITE( HEADER(I+1),'(A)') TRIM(PLAINTXT(J1:J2))
            I = I + 2
            IF (J2 > LENGTH) EXIT
          ENDDO
        ENDIF

!-----------------------------------------------------------------------
!     CSCRP - Wheat, barley, cassava
      CASE ('CSCRP')
!       -----------------------------------
!       wheat, barley
        IF (INDEX('WH,BA',CROP) > 0) THEN
          WRITE (HEADER(I),'(A,A)')
     &      "  VREQ VBASE  VEFF  PPS1  PPS2",
     &      "    P1    P2    P3    P4    P5    P6    P7    P8"
          I=I+1
          WRITE (HEADER(I),'(2F6.1,2F6.2,9F6.1)') 
     &      VREQ, VBASE, VEFF, PPS1, PPS2, 
     &      P1, P2, P3, P4, P5, P6, P7, P8
          I=I+1
          WRITE (HEADER(I),'(A)')
     &      " GNOWT  GWTS SHWTS PHINT  LA1S  LAFV  LAFR"
          I=I+1
          WRITE (HEADER(I),'(2F6.1, F6.2,F6.1, 3F6.2)') 
     &      GNOWT, GWTS, SHWTS, PHINT, LA1S, LAFV, LAFR 
          I=I+1
!       -----------------------------------
!       cassava
        ELSEIF (INDEX('CS',CROP) > 0) THEN
          WRITE (HEADER(I),'(A,A)')
     &      "  PPS1   P1L   P2L   P4L   P5L SRNOW  SRFR"
C    &      "  LA1S  LAXS"
C    &      " LAXNO  LAFS LAFNO  LAWS"
          I=I+1
c          WRITE (HEADER(I),'(4F6.1,F6.0,2F6.2,6F6.0)') 
c     &      PPS1, P1L, P2L, P4L, P5L, SRNOW, SRFR, LA1S, LAXS, 
c     &      LAXNO, LAFS, LAFNO, LAWS
          I=I+1
          WRITE (HEADER(I),'(A)') " PHINT LLIFA  STFR"
          I=I+1
c          WRITE (HEADER(I),'(2F6.0,F6.2)') PHINT, LLIFA, STFR
          I=I+1
        ENDIF

!       Print optional extra stuff from ecotype file
        LENGTH = LenString(PLAINTXT)
        IF (LENGTH > 0) THEN
          DO J=1,3
            J1 = (J-1)*78+1
            J2 = J1 + 77
            WRITE( HEADER(I),'(A)') TRIM(ATLINE(J1+149:J2+149))
            WRITE( HEADER(I+1),'(A)') TRIM(PLAINTXT(J1:J2))
            I = I + 2
            IF (J2 > LENGTH) EXIT
          ENDDO
        ENDIF
!-----------------------------------------------------------------------
!     Cassava       
      CASE ('CSCAS')
         WRITE (HEADER(I),'(A,A)')
     &     "  PPS1 B01ND B12ND B23ND B34ND B45ND B56ND",
     &     " SR#WT  SRFR  HMPC PHINT"
         I=I+1
          WRITE (HEADER(I),'(F6.2,6F6.1,4F6.2)') 
     &     PPS1, B01ND, B12ND, B23ND, B34ND, B45ND, B56ND,
     &     SRNWT, SRFR, HMPC, PHINT
         I=I+1
         WRITE (HEADER(I),'(A,A)') 
     &     "  LA1S  LAXS LAXND LAXN2  LAFS LAFND  SLAS",
     &     " LLIFA LPEFR  STFR"
        I=I+1
         WRITE (HEADER(I),'(F6.1,F6.0,6F6.1,2F6.2)') 
     &    LA1S, LAXS, LAXND, LAXN2, LAFS, LAFND, SLASS, LLIFA,
     &    LPEFR, STFR
       I=I+1
!       Print optional extra stuff from ecotype file
        LENGTH = LenString(PLAINTXT)
        IF (LENGTH > 0) THEN
          DO J=1,3
            J1 = (J-1)*78+1
            J2 = J1 + 77
            WRITE( HEADER(I),'(A)') TRIM(ATLINE(J1+149:J2+149))
            WRITE( HEADER(I+1),'(A)') TRIM(PLAINTXT(J1:J2))
            I = I + 2
            IF (J2 > LENGTH) EXIT
          ENDDO
        ENDIF
!-----------------------------------------------------------------------
!     Cassava CIAT      
      CASE ('CSYCA')
         WRITE (HEADER(I),'(A,A)')
     &     "  PPS1 B01ND B12ND SR#WT  HMPC "
          I=I+1
          WRITE (HEADER(I),'(F6.2,2F6.0,2F6.2)') 
     &     PPS1, B01ND, B12ND, SRNWT, HMPC
         I=I+1
         WRITE (HEADER(I),'(A,A)') 
     &     "  LA1S  LAXS  SLAS",
     &     " LLIFA LPEFR LNSLP NODWT NODLT"
        I=I+1
         WRITE (HEADER(I),'(F6.1,F6.0,2F6.1,3F6.2, 1F6.1)') 
     &    LA1S, LAXS, SLASS, LLIFA,
     &    LPEFR, LNSLP, NODWT, NODLT
       I=I+1
!       Print optional extra stuff from ecotype file
        LENGTH = LenString(PLAINTXT)
        IF (LENGTH > 0) THEN
          DO J=1,3
            J1 = (J-1)*78+1
            J2 = J1 + 77
            WRITE( HEADER(I),'(A)') TRIM(ATLINE(J1+149:J2+149))
            WRITE( HEADER(I+1),'(A)') TRIM(PLAINTXT(J1:J2))
            I = I + 2
            IF (J2 > LENGTH) EXIT
          ENDDO
        ENDIF

!-----------------------------------------------------------------------
!     CERES-Maize
      CASE ('MZCER')
            WRITE (HEADER(I),900) P1,P2,P5; I=I+1
            WRITE (HEADER(I),901) G2,G3,PHINT; I=I+1

!-----------------------------------------------------------------------
!     IXIM-Maize
      CASE ('MZIXM')
             WRITE (HEADER(I),915) P1,P2,P5,AX; I=I+1
             WRITE (HEADER(I),916) G2,G3,PHINT,LX; I=I+1
             
!-----------------------------------------------------------------------
!     Sorghum
      CASE ('SGCER')
            WRITE (HEADER(I), 902) P1,P2O,P2R,P5; I=I+1
            WRITE (HEADER(I),1002) G1,G2,PHINT,P3,P4; I=I+1
            !Print optional parameters if used.
            IF (PBASE > 1.E-2 .AND. PSAT > 1.E-2) THEN
              WRITE(HEADER(I), 1010) PBASE, PSAT; I=I+1
            ENDIF

!-----------------------------------------------------------------------
!     Millet
      CASE ('MLCER')
            WRITE (HEADER(I), 903) P1,P2O,P2R,P5; I=I+1
            WRITE (HEADER(I),1003) G1,G4,PHINT; I=I+1

!-----------------------------------------------------------------------
!     Potato
      CASE ('PTSUB')
!      ELSEIF (INDEX ('PT',CROP) .GT. 0) THEN
         WRITE (HEADER(I), 905) G2,G3; I=I+1
!         WRITE (HEADER(I), 905) G2,G3,G4; I=I+1
         WRITE (HEADER(I),1005) PD,P2,TC; I=I+1

!-----------------------------------------------------------------------
!     Rice
      CASE ('RICER')
!      ELSEIF (INDEX ('RI',CROP) .GT. 0) THEN
         WRITE (HEADER(I), 906) P1,P2R,P5,P2O; I=I+1
         WRITE (HEADER(I),1006) G1,G2,G3,G4,G5; I=I+1

!-----------------------------------------------------------------------
!     Aroids
      CASE ('TNARO','TRARO')
!      ELSEIF (INDEX ('TNTR',CROP) .GT. 0) THEN
         WRITE (HEADER(I), 911) P1,P3,P4,P5; I=I+1
         WRITE (HEADER(I),1011) G3,G4,PHINT,PCINT,PCGRD; I=I+1

!-----------------------------------------------------------------------
!     Pineapple **
      CASE ('PIALO')
         WRITE (HEADER(I),2010) P1,P2,P3,P4,P5,P6; I=I+1
         WRITE (HEADER(I),2011) G2,G3,PHINT; I=I+1
 2010 FORMAT (1X,'    P1:',F6.1,'    P2:',F6.1,
     &           '    P3:',F6.1,'    P4:',F6.0,
     &           '    P5:',F6.1,'    P6:',F6.1)
 2011 FORMAT (1X,'    G2:',F6.1,'    G3:',F6.2,
     &           ' PHINT:',F6.1)

!-----------------------------------------------------------------------
!     Sugarcane - Canegro
      CASE ('SCCAN')

!-----------------------------------------------------------------------
!     Sugarcane - CASUPRO
      CASE ('SCCSP')
!     Not currently correct for either CaneGro or Casupro
!     CHP removed 8/3/07
!      ELSEIF (INDEX ('SC',CROP) .GT. 0) THEN
!        IF (INDEX(MODEL,'SCCAN') > 0) THEN
!          WRITE (HEADER(I), 907) P1,RATPOT,LFMAX; I=I+1
!          WRITE (HEADER(I),1007) G1,PI1,PI2,DTTPI; I=I+1
!        ELSEIF (INDEX(MODEL,'SCCSP') > 0) THEN
!          WRITE (HEADER(I),957) PI1, PI2, DTPI, SMAX, SO, GMAX, GO; I=I+1
!          WRITE (HEADER(I),1057) M1, M2, LA, WTPSD, SLAVAR, SIZLF; I=I+1
!        ENDIF

!-----------------------------------------------------------------------
!!     Sunflower
!      ELSEIF (INDEX ('SU',CROP) .GT. 0) THEN
!         WRITE (HEADER(I), 908) P1,P2,P5; I=I+1
!         WRITE (HEADER(I),1008) G2,G3,O1; I=I+1
!!     Pineapple
!      ELSEIF (INDEX ('PI',CROP) .GT. 0) THEN
!         WRITE (HEADER(I), 909) P2,P3,P4; I=I+1
!         WRITE (HEADER(I),1009) G2,G3,PHINT; I=I+1

!!     ?? old cotton model??
!      ELSEIF (INDEX ('CO',CROP) .GT. 0) THEN
!         WRITE (HEADER(I), 912) SCPB,RESPC,SQCON; I=I+1
!         WRITE (HEADER(I),1012) FCUT,FLAI,DDISQ
!      ENDIF

      END SELECT

      Headers % ICOUNT  = I - 1
      Headers % Header = HEADER

      RETURN

C-----------------------------------------------------------------------
C     FORMAT Strings
C-----------------------------------------------------------------------

  360 FORMAT (3X,'SOIL LOWER UPPER   SAT  EXTR  INIT   ROOT   BULK',
     &    5X,'pH    NO3    NH4    ORG')
  361 FORMAT (2X,
     &    'DEPTH LIMIT LIMIT    SW    SW    SW   DIST   DENS',26X,'C')
  362 FORMAT (3X,'cm',3X,3('cm3/cm3',4X),5X,'g/cm3',9X,'ugN/g  ugN/g',
     &    5X,'%')
  363 FORMAT (79('-'))

  410 FORMAT (I3,'-',I3,5(1X,F5.3),7(1X,F6.2))
  610 FORMAT ('TOT-',I3,5F6.1,2X,'<--cm   -','  kg/ha-->',2F7.1,I7)

  710 FORMAT ('SOIL ALBEDO    :',F5.2,6X,'EVAPORATION LIMIT :',F5.2,
     &        9X,'MIN. FACTOR  :',F5.2)
  711 FORMAT ('RUNOFF CURVE # :',F5.2,
     &        6X,'DRAINAGE RATE     :',F5.2,9X,'FERT. FACTOR :',F5.2)

  800 FORMAT (1X,A10,1X,'CULTIVAR :',A6,'-',A16,3X,'ECOTYPE :',
     &        A6)
  850 FORMAT (1X,'CSDVAR :',F5.2,'  PPSEN  :',F5.2,
     &         '  EMG-FLW:',F5.2,'  FLW-FSD:',F5.2,'  FSD-PHM :',F6.2)
  851 FORMAT (1X,'WTPSD  :',F5.3,'  SDPDVR :',F5.2,
     &         '  SDFDUR :',F5.2,'  PODDUR :',F5.2,'  XFRUIT  :',F6.2)
  853 FORMAT (1X,'THRESH :',F5.1,'  SDPRO  :',F5.3,'  SDLIP   :',F6.3)

  852 FORMAT (1X,'WTPSD  :',F5.3,'  SDPDVR :',F5.1,
     &         '  SDFDUR :',F5.2,'  PODDUR :',F5.2,'  XFRUIT  :',F6.2)

  870 FORMAT (1X,'VREQ   :',F6.1,'  VBASE  :',F6.1,'  VEFF   :',F6.2,
     &         '  PPS1   :',F6.3,'  PPS2   :',F6.1)
  871 FORMAT (1X,'P1     :',F6.1,'  P2     :',F6.1,'  P3     :',F6.1,
     &         '  P4     :',F6.1)
  872 FORMAT (1X,'P5     :',F6.1,'  P6     :',F6.1,'  P7     :',F6.1,
     &         '  P8     :',F6.1)
  873 FORMAT (1X,'GRNOW  :',F6.1,'  GRWTX  :',F6.1,'  SHWMS  :',F6.2,
     &         '  PHINT  :',F6.1)

  880 FORMAT (1X,'PPS1   :',F6.1,'  P1L    :',F6.1,'  P2L    :',F6.1,
     &         '  P4L    :',F6.1)
  881 FORMAT (1X,'P5L    :',F6.0,'  SRNOW  :',F6.2,'  SRFR   :',F6.2,
     &         '  LA1S   :',F6.1)
  882 FORMAT (1X,'LAXS   :',F6.1,'  LAXNO  :',F6.1,'  LAFS   :',F6.1,
     &         '  LAFNO  :',F6.1)
  883 FORMAT (1X,'LAWS   :',F6.1,'  PHINT  :',F6.1,'  LLIFA  :',F6.1,
     &         '  STFR   :',F6.2)

  900 FORMAT (1X,'P1     :',F7.2,'  P2     :',F7.4,
     &         '  P5     :',F7.2)
  915 FORMAT (1X,'P1     :',F7.2,'  P2     :',F7.4,
     &         '  P5     :',F7.2,'  AX     :',F6.1)
     
  902 FORMAT (1X,'P1     :',F5.1,'  P2O    :',F5.2,
     &         '  P2R    :',F6.1,'  P5     :',F6.2)
  903 FORMAT (1X,'P1     :',F6.2,'  P2O    :',F6.3,
     &         '  P2R    :',F6.1,'  P5     :',F6.2)

  904 FORMAT (1X,'P1V    :',F8.3,'  P1D    :',F8.3,
     &         '  P5     :',F8.2)
  954 FORMAT (1X,'P1V    :',F8.3,'  P1D    :',F8.4,
     &         '  P5     :',F8.2)      !,'  LT50H  :',F8.2)
  905 FORMAT (1X,'G2     :',F7.2,'  G3     :',F7.4)
!  905 FORMAT (1X,'G2     :',F7.2,'  G3     :',F7.4,'  G4     :',F7.2)
  906 FORMAT (1X,'P1     :',F6.1,'  P2R    :',F6.1,
     &         '  P5     :',F6.1,'  P2O    :',F6.1)
  907 FORMAT (1X,'P1     :',F6.1,'  RATPOT :',F6.1,
     &         '  LFMAX  :',F6.1)
  957 FORMAT (1X,' PI1   :',F6.1,'   PI2  :',F6.1,
     &         '   DTPI  :',F6.1,'   SMAX :',F6.1,
     &         '   SO    :',F6.1,'   GMAX :',F6.1,
     &         '   GO    :',F6.1)
  908 FORMAT (1X,'P1     :',F7.2,'  P2     :',F7.4,
     &         '  P5     :',F7.2)
  909 FORMAT (1X,'P2     :',F6.1,'  P3     :',F6.1,
     &         '  P4     :',F6.0)
  911 FORMAT (1X,'P1     :',F7.1,' P3     :',F7.2,
     &          ' P4     :',F7.1,' P5     :',F7.2)
  912 FORMAT (1X,'SCPB   :',F7.1,' RESPC  :',F7.3,
     &          ' SQCON  :',F7.3)
  901 FORMAT (1X,'G2     :',F7.2,'  G3     :',F7.3,'  PHINT  :',F7.3)
  916 FORMAT (1X,'G2     :',F7.2,'  G3     :',F7.3,
     &         '  PHINT  :',F7.3,'  LX     :',F6.1)  
 1002 FORMAT (1X,'G1     :',F5.1,'  G2     :',F5.2,'  PHINT  :',F6.2,
     &           '  P3     :',F6.1,'  P4    :',F6.1)     
 1003 FORMAT (1X,'G1     :',F6.2,'  G4     :',F6.2,'  PHINT  :',F6.2)
 1004 FORMAT (1X,'G1     :',F8.3,'  G2     :',F8.3,'  G3     :',F8.3,
     &         '  PHINT  :',F8.3)
 1054 FORMAT (1X,'G1     :',F8.3,'  G2     :',F8.3,'  G3     :',F8.3,
     &         '  PHINT  :',F8.3)
 1005 FORMAT (1X,'PD     :',F7.2,'  P2     :',F7.3,'  TC     :',F7.3)
 1006 FORMAT (1X,'G1     :',F6.1,'  G2     :',F6.4,
     &         '  G3     :',F6.2,'  G4     :',F6.2,'  G5     :',F6.2)
 1007 FORMAT (1X,'G1     :',F6.1,'  PI1    :',F6.1,
     &         '  PI2    :',F6.1,'  DTTPI  :',F6.1)
 1057 FORMAT (1X,' M1    :',F6.1,'   M2   :',F6.1,
     &         '   LA    :',I6  ,'   WTPSD: ',F6.2,
     &         '  SLAVAR:',F6.1,'   SIZLF:',F6.1)

 1008 FORMAT (1X,'G2     :',F7.2,'  G3     :',F7.3,'  O1     :',F4.0)
 1009 FORMAT (1X,'G2     :',F6.1,'  G3     :',F6.2,'  PHINT  :',F6.1)
 1011 FORMAT (1X,'G3     :',F7.1,' G4     :',F7.1,
     &          ' PHINT  :',F7.1,' PCINT  :',F7.1,' PCGRD  :',F7.1)
 1012 FORMAT (1X,'FCUT   :',F7.3,' FLAI   :',F7.2,
     &          ' DDISQ  :',F7.1)

 2000 FORMAT (1X,'DUB1   :',F6.1,'  DUBR   :',F6.1,'  DESP   :',F6.2,
     &         '  PHCX   :',F6.2,'  S#PE   :',F6.1)
 2001 FORMAT (1X,'S#FX   :',F6.1,'  S#PX   :',F6.1,'  SWNX   :',F6.1,
     &         '  L#IS   :',F6.2,'  L#IP   :',F6.2)
 2002 FORMAT (1X,'LALX   :',F6.0,'  LAXA   :',F6.2,'  LAL3   :',F6.0,
     &         '  LAWS   :',F6.0,'  LFLI   :',F6.0)

 1010 FORMAT (1X,'PBASE  :',F6.2,'  PSAT   :',F6.2)
 
      END SUBROUTINE OPSOIL
C=======================================================================

C=======================================================================
C  HEADER, Subroutine, C. H. Porter
C-----------------------------------------------------------------------
C  Writes simulation header info to file.

!     Currently, the routine reads header info from a file (written 
!     previously by input module or elsewhere in the code).  Probably
!     should generate the header info here.

C-----------------------------------------------------------------------
C  REVISION HISTORY
C  10/18/2001 CHP Written.
C  08/12/2003 CHP Added I/O error checking
!  05/09/2007 CHP Can get CONTROL variable from here, don't need to send
!                 name of FILEIO from suborutine.  Keep sending DYNAMIC 
!                 and RUN, because these do not always have the same 
!                 values as that in the CONTROL variable.
C-----------------------------------------------------------------------
! Called by: IRRIG, OPWBAL, OPGROW, . . . 
! Calls: None
C========================================================================

      SUBROUTINE HEADER(DYNAMIC, LUNDES, RUN)

!     Reads up to 100 lines of header information and prints to output
!       files as needed.
!     DYNAMIC = 0 : Prints version number and date/time only
!     DYNAMIC = 1 (RUNINIT) : Complete header is printed
!     DYNAMIC = 2 (SEASINIT): Short header is printed 
!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      USE HeaderMod
      IMPLICIT NONE
      INCLUDE 'COMIBS.blk'
      INCLUDE 'COMSWI.blk'
      SAVE

      CHARACTER*6, PARAMETER :: ERRKEY = 'HEADER'

      INTEGER DATE_TIME(8)
      INTEGER DYNAMIC, I, LUNDES
      INTEGER ICOUNT, RUN, PREV_RUN
      INTEGER ShortCount

      DATA PREV_RUN /0/
      INTEGER, PARAMETER :: MAXLUN = 200
      logical NOHEADER(MAXLUN)

      TYPE (ControlType) CONTROL
      CALL GET(CONTROL)

      ICOUNT = HEADERS % ICOUNT
      ShortCount = HEADERS % ShortCount

!-----------------------------------------------------------------------
      IF (RUN /= PREV_RUN) THEN
        NOHEADER = .TRUE.
        PREV_RUN = RUN
!        IF (CONTROL % MULTI > 1 .OR. CONTROL % RNMODE == 'Q') THEN
!          CALL MULTIRUN(RUN)
!        ENDIF
      ENDIF

!     Write Model info and date-time stamp to all headers
      IF (ICOUNT .GT. 0) THEN
        WRITE(LUNDES,'(/,A80)') HEADERS % Header(1)
      ELSE
        CALL DATE_AND_TIME (VALUES=DATE_TIME)
        WRITE (LUNDES,500)Version, VBranch, MonthTxt(DATE_TIME(2)),
     &    DATE_TIME(3), DATE_TIME(1), DATE_TIME(5), DATE_TIME(6), 
     &    DATE_TIME(7)
        WRITE(LUNDES,1200) MOD(RUN,1000)
        RETURN
      ENDIF

!***********************************************************************
!***********************************************************************
!     Run Initialization
!***********************************************************************
      IF (DYNAMIC .EQ. RUNINIT) THEN
!-----------------------------------------------------------------------
!     Write OVERVIEW header info to destination file.
      DO I = 2, ICOUNT
        WRITE(LUNDES,'(A80)',ERR=200) HEADERS % Header(I)
      ENDDO
  200 CONTINUE

!***********************************************************************
!***********************************************************************
!     Seasonal initialization
!***********************************************************************
      ELSEIF (DYNAMIC .EQ. SEASINIT) THEN
!-----------------------------------------------------------------------
      IF (LUNDES > MAXLUN) RETURN

      IF (NOHEADER(LUNDES) .AND. RUN > 0 .AND. ICOUNT > 0)THEN
!       Header exists, write it
        DO I = 2, ShortCount
          WRITE (LUNDES,'(A)') HEADERS % HEADER(I)
        ENDDO
        WRITE(LUNDES,*) " "  
        NOHEADER(LUNDES) = .FALSE.

!      ELSE
!!       Header already written for this run, or header not avail yet.
!        CALL DATE_AND_TIME (VALUES=DATE_TIME)
!        WRITE (LUNDES,500)Version,MonthTxt(DATE_TIME(2)),DATE_TIME(3), 
!     &    DATE_TIME(1), DATE_TIME(5), DATE_TIME(6), DATE_TIME(7)
!        WRITE(LUNDES,1200) MOD(RUN,1000)
      ENDIF

!***********************************************************************
      ENDIF
!***********************************************************************
!***********************************************************************
!     END OF DYNAMIC IF CONSTRUCT
!***********************************************************************
      RETURN

  500   FORMAT ("*DSSAT Cropping System Model Ver. ",I1,".",I1,".",I1,
     &    ".",I3.3,1X,A10,4X,
     &    A3," ",I2.2,", ",I4,"; ",I2.2,":",I2.2,":",I2.2)
 1200   FORMAT (/,'*RUN ',I3,/)

      END SUBROUTINE HEADER

!=======================================================================

!=======================================================================
      SUBROUTINE CO2Header(CO2)
!     Writes value of CO2 at beginning of run to header
      USE HeaderMod
      IMPLICIT NONE

      INTEGER I
      REAL CO2
      CHARACTER*80 TEXT

      DO I = Headers%ShortCount + 1, Headers%ICOUNT
        TEXT = Headers%Header(I)
        IF (TEXT(19:21) == 'CO2') THEN
          WRITE(Headers%Header(I),'(A,I4,A,A)') 
     &      TEXT(1:21), NINT(CO2), "ppm", TEXT(29:80)
          EXIT
        ENDIF
      ENDDO

      RETURN
      END SUBROUTINE CO2Header
!=======================================================================




C=======================================================================
C  WARNING, Subroutine, C.H.PORTER
C  Writes warning messages to Warning.OUT file
C-----------------------------------------------------------------------
C  REVISION HISTORY
C  03/21/2002 CHP Written
C  09/03/2004 CHP Modified call to GETPUT_CONTROL 
C  03/22/2005 CHP Added option to suppress Warning.OUT messages with 
C                 IDETL = 0 (zero).
!  05/04/2005 CHP Added date to warning message.
!  01/11/2007 CHP Changed GETPUT calls to GET and PUT
C=======================================================================

      SUBROUTINE WARNING (ICOUNT, ERRKEY, MESSAGE)

!     FILEIO and RUN needed to generate header for WARNING.OUT file

      USE ModuleDefs
      USE ModuleData
      USE HeaderMod
      IMPLICIT NONE
      SAVE

      CHARACTER*(*) ERRKEY
      CHARACTER*11, PARAMETER :: WarnOut = 'WARNING.OUT'
      CHARACTER*30  FILEIO
      CHARACTER*78  MESSAGE(*)

      INTEGER ICOUNT, DOY, I, LUN, OLDRUN, RUN, YEAR, YRDOY, ErrCode
      LOGICAL FIRST, FEXIST, FOPEN

      TYPE (ControlType) CONTROL
      TYPE (SwitchType)  ISWITCH

      DATA FIRST /.TRUE./
      DATA OLDRUN /0/

!-----------------------------------------------------------------------
!     Suppress Warning.OUT if IDETL = '0' (zero)
      CALL GET(ISWITCH)
!     IF (ISWITCH % IDETL == '0') RETURN

      CALL GET(CONTROL)
      FILEIO = CONTROL % FILEIO
      RUN    = CONTROL % RUN
      YRDOY  = CONTROL % YRDOY
      ErrCode = CONTROL % ErrCode

      IF (INDEX(ERRKEY,'ENDRUN') <= 0) THEN
!       First time routine is called to print, open file.
!       File will remain open until program execution is stopped.
        IF (FIRST) THEN

!         Check for IDETL = '0' (zero) --> suppress output
          CALL GET(ISWITCH)
          IF (ISWITCH % IDETL == '0' .AND. ErrCode <=0) RETURN

          CALL GETLUN(WarnOut, LUN)
          INQUIRE (FILE = WarnOut, EXIST = FEXIST)
          IF (FEXIST) THEN
            INQUIRE (FILE = WarnOut, OPENED = FOPEN)
            IF (.NOT. FOPEN) THEN
              OPEN (UNIT=LUN, FILE=WarnOut, STATUS='OLD',
     &            POSITION='APPEND')
            ENDIF
          ELSE
            OPEN (UNIT=LUN, FILE=WarnOut, STATUS='NEW')
            WRITE(LUN,'("*WARNING DETAIL FILE")')
          ENDIF

          WRITE(LUN,'(/,78("*"))')
          IF (CONTROL % MULTI > 1) CALL MULTIRUN(RUN,0)
          IF (Headers % RUN == RUN) THEN
            CALL HEADER(SEASINIT, LUN, RUN)
            FIRST = .FALSE.
            OLDRUN = RUN
          ENDIF
        ENDIF
!         VSH
          CALL GETLUN('OUTWARN', LUN)
          INQUIRE (FILE = WarnOut, OPENED = FOPEN)
          IF (.NOT. FOPEN) THEN
             OPEN (UNIT=LUN, FILE=WarnOut, STATUS='OLD',
     &             POSITION='APPEND')
          ENDIF          
      ENDIF

      IF (ICOUNT > 0) THEN
        !Print header if this is a new run.
        IF (OLDRUN .NE. RUN .AND. RUN .NE. 0 .AND. FILEIO .NE. "")THEN
          IF (Headers % RUN == RUN) THEN
            CALL HEADER(SEASINIT,LUN,RUN)
            OLDRUN = RUN
          ENDIF
        ENDIF

!       Print the warning.  Message is sent from calling routine as text.
        CALL YR_DOY(YRDOY, YEAR, DOY)
        WRITE(LUN,'(/,1X,A,"  YEAR DOY = ",I4,1X,I3)')ERRKEY,YEAR,DOY
        DO I = 1, ICOUNT
          WRITE(LUN,'(1X,A78)') MESSAGE(I)
        ENDDO
      ENDIF

      IF (INDEX(ERRKEY,'ENDRUN') > 0) THEN    !ERRKEY = 'ENDRUN' -> End of season
        FIRST = .TRUE.
        CLOSE(LUN)
      ENDIF

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE WARNING



!======================================================================
!     Paddy Managment routines.
!======================================================================
      MODULE FloodModule
!=======================================================================
!     Data construct for flood data. 
      Type FloodWatType
        !From IRRIG
        LOGICAL BUNDED        
        INTEGER NBUND         
        REAL ABUND            
        REAL PUDPERC, PERC
        REAL PLOWPAN    !Depth of puddling (m) (ORYZA)

        !From Paddy_Mgmt
        INTEGER YRDRY, YRWET  
        REAL FLOOD, FRUNOFF   
        REAL TOTBUNDRO        
        LOGICAL PUDDLED       

        REAL CEF, EF          !From SPAM
        REAL INFILT, RUNOFF   !From WATBAL
      End Type FloodWatType

      Type FloodNType
        INTEGER NDAT
        REAL    ALGFON                        !Algae kill or dry-up
        REAL    FLDH4C, FLDN3C                !Flood N concentrations
        REAL    FLDU, FLDN3, FLDH4            !Flood N mass (kg/ha)
        REAL    FRNH4U, FRNO3U                !Flood N uptake (kg/ha)
        REAL    DLTFUREA, DLTFNH4, DLTFNO3    !Flood N flux (kg/ha)
      End Type FloodNType

      END MODULE FloodModule
!======================================================================

C=======================================================================
C  PETASCE, Subroutine, K. R. Thorp
C  Calculates reference evapotranspiration using the ASCE
C  Standardized Reference Evapotranspiration Equation.
C  Adjusts reference evapotranspiration to potential evapotranspiration
C  using dual crop coefficients.
C  DeJonge K. C., Thorp, K. R., 2017. Implementing standardized refernce
C  evapotranspiration and dual crop coefficient approach in the DSSAT
C  Cropping System Model. Transactions of the ASABE. 60(6):1965-1981.
!-----------------------------------------------------------------------
C  REVISION HISTORY
C  08/19/2013 KRT Added the ASCE Standardize Reference ET approach
C  01/26/2015 KRT Added the dual crop coefficient (Kc) approach
C  01/18/2018 KRT Merged ASCE dual Kc ET method into develop branch
!-----------------------------------------------------------------------
!  Called from:   PET
!  Calls:         None
C=======================================================================
      SUBROUTINE PETASCE(
     &        CANHT, DOY, MSALB, MEEVP, SRAD, TDEW,       !Input
     &        TMAX, TMIN, WINDHT, WINDRUN, XHLAI,         !Input
     &        XLAT, XELEV,                                !Input
     &        EO)                                         !Output
!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      IMPLICIT NONE
      SAVE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL CANHT, MSALB, SRAD, TDEW, TMAX, TMIN, WINDHT, WINDRUN
      REAL XHLAI, XLAT, XELEV
      INTEGER DOY
      CHARACTER*1 MEEVP
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LOCAL VARIABLES:
      REAL TAVG, PATM, PSYCON, UDELTA, EMAX, EMIN, ES, EA, FC, FEW, FW
      REAL ALBEDO, RNS, PIE, DR, LDELTA, WS, RA1, RA2, RA, RSO, RATIO
      REAL FCD, TK4, RNL, RN, G, WINDSP, WIND2m, Cn, Cd, KCMAX, RHMIN
      REAL WND, CHT
      REAL REFET, SKC, KCBMIN, KCBMAX, KCB, KE, KC
!-----------------------------------------------------------------------

!     ASCE Standardized Reference Evapotranspiration
!     Average temperature (ASCE Standard Eq. 2)
      TAVG = (TMAX + TMIN) / 2.0 !deg C

!     Atmospheric pressure (ASCE Standard Eq. 3)
      PATM = 101.3 * ((293.0 - 0.0065 * XELEV)/293.0) ** 5.26 !kPa

!     Psychrometric constant (ASCE Standard Eq. 4)
      PSYCON = 0.000665 * PATM !kPa/deg C

!     Slope of the saturation vapor pressure-temperature curve
!     (ASCE Standard Eq. 5)                                !kPa/degC
      UDELTA = 2503.0*EXP(17.27*TAVG/(TAVG+237.3))/(TAVG+237.3)**2.0

!     Saturation vapor pressure (ASCE Standard Eqs. 6 and 7)
      EMAX = 0.6108*EXP((17.27*TMAX)/(TMAX+237.3)) !kPa
      EMIN = 0.6108*EXP((17.27*TMIN)/(TMIN+237.3)) !kPa
      ES = (EMAX + EMIN) / 2.0                     !kPa

!     Actual vapor pressure (ASCE Standard Eq. 8)
      EA = 0.6108*EXP((17.27*TDEW)/(TDEW+237.3)) !kPa

!     RHmin (ASCE Standard Eq. 13, RHmin limits from FAO-56 Eq. 70)
      RHMIN = MAX(20.0, MIN(80.0, EA/EMAX*100.0))

!     Net shortwave radiation (ASCE Standard Eq. 16)
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23
      ENDIF
      RNS = (1.0-ALBEDO)*SRAD !MJ/m2/d

!     Extraterrestrial radiation (ASCE Standard Eqs. 21,23,24,27)
      PIE = 3.14159265359
      DR = 1.0+0.033*COS(2.0*PIE/365.0*DOY) !Eq. 23
      LDELTA = 0.409*SIN(2.0*PIE/365.0*DOY-1.39) !Eq. 24
      WS = ACOS(-1.0*TAN(XLAT*PIE/180.0)*TAN(LDELTA)) !Eq. 27
      RA1 = WS*SIN(XLAT*PIE/180.0)*SIN(LDELTA) !Eq. 21
      RA2 = COS(XLAT*PIE/180.0)*COS(LDELTA)*SIN(WS) !Eq. 21
      RA = 24.0/PIE*4.92*DR*(RA1+RA2) !MJ/m2/d Eq. 21

!     Clear sky solar radiation (ASCE Standard Eq. 19)
      RSO = (0.75+2E-5*XELEV)*RA !MJ/m2/d

!     Net longwave radiation (ASCE Standard Eqs. 17 and 18)
      RATIO = SRAD/RSO
      IF (RATIO .LT. 0.3) THEN
        RATIO = 0.3
      ELSEIF (RATIO .GT. 1.0) THEN
        RATIO = 1.0
      END IF
      FCD = 1.35*RATIO-0.35 !Eq 18
      TK4 = ((TMAX+273.16)**4.0+(TMIN+273.16)**4.0)/2.0 !Eq. 17
      RNL = 4.901E-9*FCD*(0.34-0.14*SQRT(EA))*TK4 !MJ/m2/d Eq. 17

!     Net radiation (ASCE Standard Eq. 15)
      RN = RNS - RNL !MJ/m2/d

!     Soil heat flux (ASCE Standard Eq. 30)
      G = 0.0 !MJ/m2/d

!     Wind speed (ASCE Standard Eq. 33)
      WINDSP = WINDRUN * 1000.0 / 24.0 / 60.0 / 60.0 !m/s
      WIND2m = WINDSP * (4.87/LOG(67.8*WINDHT-5.42))

!     Aerodynamic roughness and surface resistance daily timestep constants
!     (ASCE Standard Table 1)
      SELECT CASE(MEEVP) !
        CASE('A') !Alfalfa reference
          Cn = 1600.0 !K mm s^3 Mg^-1 d^-1
          Cd = 0.38 !s m^-1
        CASE('G') !Grass reference
          Cn = 900.0 !K mm s^3 Mg^-1 d^-1
          Cd = 0.34 !s m^-1
      END SELECT

!     Standardized reference evapotranspiration (ASCE Standard Eq. 1)
      REFET =0.408*UDELTA*(RN-G)+PSYCON*(Cn/(TAVG+273.0))*WIND2m*(ES-EA)
      REFET = REFET/(UDELTA+PSYCON*(1.0+Cd*WIND2m)) !mm/d
      REFET = MAX(0.0001, REFET)

!     FAO-56 dual crop coefficient approach
!     Basal crop coefficient (Kcb)
!     Also similar to FAO-56 Eq. 97
!     KCB is zero when LAI is zero
      CALL GET('SPAM', 'SKC', SKC)
      CALL GET('SPAM', 'KCBMIN', KCBMIN)
      CALL GET('SPAM', 'KCBMAX', KCBMAX)
      IF (XHLAI .LE. 0.0) THEN
         KCB = 0.0
      ELSE
         !DeJonge et al. (2012) equation
         KCB = MAX(0.0,KCBMIN+(KCBMAX-KCBMIN)*(1.0-EXP(-1.0*SKC*XHLAI)))
      ENDIF

      !Maximum crop coefficient (Kcmax) (FAO-56 Eq. 72)
      WND = MAX(1.0,MIN(WIND2m,6.0))
      CHT = MAX(0.001,CANHT)
      SELECT CASE(MEEVP)
        CASE('A') !Alfalfa reference
            KCMAX = MAX(1.0,KCB+0.05)
        CASE('G') !Grass reference
            KCMAX = MAX((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))
     &                      *(CHT/3.0)**(0.3)),KCB+0.05)
      END SELECT

      !Effective canopy cover (fc) (FAO-56 Eq. 76)
      IF (KCB .LE. KCBMIN) THEN
         FC = 0.0
      ELSE
         FC = ((KCB-KCBMIN)/(KCMAX-KCBMIN))**(1.0+0.5*CANHT)
      ENDIF

      !Exposed and wetted soil fraction (FAO-56 Eq. 75)
      !Unresolved issue with FW (fraction wetted soil surface).
      !Some argue FW should not be used to adjust demand.
      !Rather wetting fraction issue should be addressed on supply side.
      !Difficult to do with a 1-D soil water model
      FW = 1.0
      FEW = MIN(1.0-FC,FW)

      !Potential evaporation coefficient (Ke) (Based on FAO-56 Eq. 71)
      !Kr = 1.0 since this is potential Ke. Model routines handle stress
      KE = MAX(0.0, MIN(1.0*(KCMAX-KCB), FEW*KCMAX))

      !Potential crop coefficient (Kc) (FAO-56 Eqs. 58 & 69)
      KC = KCB + KE

      !Potential evapotranspiration (FAO-56 Eq. 69)
      EO = (KCB + KE) * REFET

      EO = MAX(EO,0.0001)

      CALL PUT('SPAM', 'REFET', REFET)
      CALL PUT('SPAM', 'KCB', KCB)
      CALL PUT('SPAM', 'KE', KE)
      CALL PUT('SPAM', 'KC', KC)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETASCE
!-------------------------------------------------------------------

C=======================================================================
