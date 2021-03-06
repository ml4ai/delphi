!***********************************************************************
!  SoilNBalSum, Subroutine 
!
!  Purpose: Writes out a one-line seasonal soil N balance  
!
!  REVISION   HISTORY
!  07/11/2006 CHP Written
!***********************************************************************
!     HJ added CNTILEDR
      SUBROUTINE SoilNBalSum (CONTROL, 
     &    AMTFER, Balance, CUMIMMOB, CUMMINER, CUMRESN, 
     &    CumSenN, HARVRESN, LITE, SOM1E, CLeach, CNTILEDR, TLITN, 
     &    N_inorganic, TSOM1N, TSOM2N, TSOM3N, WTNUP,
     &    CUMFNRO, NGasLoss)

!***********************************************************************

!     ------------------------------------------------------------------
      USE N2O_mod 
      IMPLICIT NONE
      SAVE

      TYPE (ControlType), INTENT(IN) :: CONTROL
      REAL, INTENT(IN), OPTIONAL :: AMTFER, Balance, CUMIMMOB, 
     &    CUMFNRO, CUMMINER, CUMRESN, CumSenN, HARVRESN, CLeach,
     &    CNTILEDR, TLITN, N_inorganic, TSOM1N, TSOM2N, TSOM3N,	 
     &    WTNUP, NGasLoss
      REAL, DIMENSION(0:NL,3), INTENT(IN), OPTIONAL :: LITE, SOM1E

      CHARACTER(LEN=14), PARAMETER :: SNSUM = 'SolNBalSum.OUT'
      INTEGER COUNT, ERRNUM, LUNSNS, Num
      LOGICAL FEXIST, FIRST
      REAL State(6), Add(4), Sub(5), Bal(2), Miner(2) !HJ changed Sub(4)

      DATA FIRST /.TRUE./
      DATA COUNT /0/

      IF (FIRST) THEN
        FIRST = .FALSE.

!       Initialize output file
        CALL GETLUN(SNSUM, LUNSNS)
!!!        INQUIRE (FILE = SNSUM, EXIST = FEXIST)
        FEXIST = .TRUE.
        IF (FEXIST) THEN
          OPEN (UNIT = LUNSNS, FILE = SNSUM, STATUS = 'OLD',
     &      IOSTAT = ERRNUM, POSITION = 'APPEND')
        ELSE
          OPEN (UNIT = LUNSNS, FILE = SNSUM, STATUS = 'NEW',
     &      IOSTAT = ERRNUM)
        ENDIF
        WRITE(LUNSNS,'(/,"*SOIL N BALANCE - CENTURY ROUTINES")')
        CALL HEADER(0, LUNSNS, CONTROL%RUN)

        WRITE(LUNSNS,5000) 
 5000   FORMAT(/,"!",T23,"|------------------- N State Variables ----",
     &    "---------------| |------------ N Additions ------------| |",
     &    "--------------- N Subtractions ----------------|",
     &    " |-Mineralization--|",
     &  /,"!",T85,"Harvest   Applied",T137,"Tile-     N gas     Flood",
     &    "    Miner-    Immob-  Seasonal",
     &  /,"!",T25,"Surface      SOM1      SOM2      SOM3    Litter    ",
     &    " Inorg   Residue   Residue  Fertiliz   Senesed   Leached   ",
     &    "drained    Uptake    Losses    Losses    alized    ilized",
     &    "   Balance",
     &  /,"@Run FILEX         TN      SN0D     S1NTD",
     &     "     S2NTD     S3NTD      LNTD      NIAD      HRNH",
     &   "     RESNC      NICM     SNNTC      NLCM      TDFC      NUCM",
     &     "     NGasC      RNRO      NMNC      NIMC   SEASBAL")
      ENDIF

!     Organic
      IF (PRESENT(LITE) .AND. PRESENT(SOM1E)) THEN
        State(1) = LITE(0,N) + SOM1E(0,N)
        IF (PRESENT(Balance)) Bal(1) = Balance
      ENDIF
      IF (PRESENT(TSOM1N)) State(2) = TSOM1N
      IF (PRESENT(TSOM2N)) State(3) = TSOM2N
      IF (PRESENT(TSOM3N)) State(4) = TSOM3N
      IF (PRESENT(TLITN))  State(5) = TLITN

      IF (PRESENT(HARVRESN)) Add(1) = HARVRESN
      IF (PRESENT(CUMRESN))  Add(2) = CUMRESN
      IF (PRESENT(CumSenN))  Add(4) = CumSenN

!     Inorganic
      
      IF (PRESENT(N_inorganic)) THEN
        State(6) = N_inorganic
        IF (PRESENT(Balance)) Bal(2) = Balance
      ENDIF
      IF (PRESENT(AMTFER))   Add(3) = AMTFER
      IF (PRESENT(CLeach))   Sub(1) = CLeach
	  IF (PRESENT(CNTILEDR)) Sub(2) = CNTILEDR     !HJ added
      IF (PRESENT(WTNUP))    Sub(3) = WTNUP
      IF (PRESENT(NGasLoss)) Sub(4) = NGasLoss
      IF (PRESENT(CUMFNRO))  Sub(5) = CUMFNRO

!     Mineralization/immobilization
      IF (PRESENT(CUMMINER)) Miner(1) = CUMMINER
      IF (PRESENT(CUMIMMOB)) Miner(2) = CUMIMMOB

!***********************************************************************
!***********************************************************************
!     Seasonal Initialization 
!***********************************************************************
      IF (CONTROL % DYNAMIC .EQ. SEASINIT) THEN
!     ------------------------------------------------------------------
      IF (CONTROL%RUN == 1) THEN
        COUNT = COUNT + 1
        IF (COUNT == 2) THEN
          WRITE(LUNSNS,'(I4,1X,A12," INIT",F9.2,6F10.2)') 
     &      CONTROL%RUN, CONTROL%FILEX, State
          COUNT = 0
          State = 0.
        ENDIF
      ENDIF

!***********************************************************************
!***********************************************************************
!     Seasonal Output 
!***********************************************************************
      ELSEIF (CONTROL % DYNAMIC .EQ. SEASEND) THEN
!     ------------------------------------------------------------------
      COUNT = COUNT + 1

      IF (COUNT == 2) THEN

        IF (CONTROL % RNMODE == 'Q') THEN
          Num = CONTROL % ROTNUM
        ELSE
          Num = CONTROL % TRTNUM
        ENDIF

        WRITE(LUNSNS,'(I4,1X,A12,I4,18F10.2)')   !HJ changed 17F10.2
     &    CONTROL%RUN, CONTROL%FILEX, Num, 
     &    State, Add, Sub, Miner, Bal(1)+Bal(2)
        COUNT = 0
        State = 0.
        Add   = 0.
        Sub   = 0.
        Miner = 0.
        Bal   = 0.
      ENDIF

!***********************************************************************
!***********************************************************************
!     End of DYNAMIC IF construct
!***********************************************************************
      END IF

      RETURN
      END SUBROUTINE SoilNBalSum


!=======================================================================
      MODULE Interface_SoilNBalSum
!     Interface needed for optional arguments with SoilNBalSum
!     HJ added CNTILEDR following
      INTERFACE
        SUBROUTINE SoilNBalSum (CONTROL, 
     &    AMTFER, Balance, CUMIMMOB, CUMMINER, CUMRESN, 
     &    CumSenN, HARVRESN, LITE, SOM1E, CLeach, CNTILEDR, TLITN, 
     &    N_inorganic, TSOM1N, TSOM2N, TSOM3N, WTNUP,
     &    CUMFNRO, NGasLoss)
          USE ModuleDefs
          TYPE (ControlType), INTENT(IN) :: CONTROL
          REAL, INTENT(IN), OPTIONAL :: AMTFER, Balance, CUMIMMOB, 
     &      CUMFNRO, CUMMINER, CUMRESN, CumSenN, HARVRESN, CLeach, 
     &      CNTILEDR, TLITN, N_inorganic, TSOM1N, TSOM2N, TSOM3N,
     &      WTNUP, NGasLoss
          REAL, DIMENSION(0:NL,3), INTENT(IN), OPTIONAL :: LITE, SOM1E
        END SUBROUTINE
      END INTERFACE
      END MODULE Interface_SoilNBalSum

!=======================================================================
