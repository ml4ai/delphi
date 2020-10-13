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

C!-----------------------------------------------------------------------
C!     Suppress Warning.OUT if IDETL = '0' (zero)
C      CALL GET(ISWITCH)
C!     IF (ISWITCH % IDETL == '0') RETURN
C
C      CALL GET(CONTROL)
C      FILEIO = CONTROL % FILEIO
C      RUN    = CONTROL % RUN
C      YRDOY  = CONTROL % YRDOY
C      ErrCode = CONTROL % ErrCode
C
C      IF (INDEX(ERRKEY,'ENDRUN') <= 0) THEN
C!       First time routine is called to print, open file.
C!       File will remain open until program execution is stopped.
C        IF (FIRST) THEN
C
C!         Check for IDETL = '0' (zero) --> suppress output
C          CALL GET(ISWITCH)
C          IF (ISWITCH % IDETL == '0' .AND. ErrCode <=0) RETURN
C
C          CALL GETLUN(WarnOut, LUN)
C          INQUIRE (FILE = WarnOut, EXIST = FEXIST)
C          IF (FEXIST) THEN
C            INQUIRE (FILE = WarnOut, OPENED = FOPEN)
C            IF (.NOT. FOPEN) THEN
C              OPEN (UNIT=LUN, FILE=WarnOut, STATUS='OLD',
C     &            POSITION='APPEND')
C            ENDIF
C          ELSE
C            OPEN (UNIT=LUN, FILE=WarnOut, STATUS='NEW')
C            WRITE(LUN,'("*WARNING DETAIL FILE")')
C          ENDIF
C
C          WRITE(LUN,'(/,78("*"))')
C          IF (CONTROL % MULTI > 1) CALL MULTIRUN(RUN,0)
C          IF (Headers % RUN == RUN) THEN
C            CALL HEADER(SEASINIT, LUN, RUN)
C            FIRST = .FALSE.
C            OLDRUN = RUN
C          ENDIF
C        ENDIF
C!         VSH
C          CALL GETLUN('OUTWARN', LUN)
C          INQUIRE (FILE = WarnOut, OPENED = FOPEN)
C          IF (.NOT. FOPEN) THEN
C             OPEN (UNIT=LUN, FILE=WarnOut, STATUS='OLD',
C     &             POSITION='APPEND')
C          ENDIF          
C      ENDIF
C
C      IF (ICOUNT > 0) THEN
C        !Print header if this is a new run.
C        IF (OLDRUN .NE. RUN .AND. RUN .NE. 0 .AND. FILEIO .NE. "")THEN
C          IF (Headers % RUN == RUN) THEN
C            CALL HEADER(SEASINIT,LUN,RUN)
C            OLDRUN = RUN
C          ENDIF
C        ENDIF
C
C!       Print the warning.  Message is sent from calling routine as text.
C        CALL YR_DOY(YRDOY, YEAR, DOY)
C        WRITE(LUN,'(/,1X,A,"  YEAR DOY = ",I4,1X,I3)')ERRKEY,YEAR,DOY
C        DO I = 1, ICOUNT
C          WRITE(LUN,'(1X,A78)') MESSAGE(I)
C        ENDDO
C      ENDIF
C
C      IF (INDEX(ERRKEY,'ENDRUN') > 0) THEN    !ERRKEY = 'ENDRUN' -> End of season
C        FIRST = .TRUE.
C        CLOSE(LUN)
C      ENDIF
C
C!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE WARNING
