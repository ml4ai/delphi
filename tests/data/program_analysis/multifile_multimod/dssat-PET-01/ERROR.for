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

C      TYPE (ControlType) CONTROL
C      CALL GET(CONTROL)
C
C      IMSG = 1
C      EFILE = 'ERROR.OUT'
C      CALL GETLUN('ERRORO', ELUN)
C
C      INQUIRE (FILE = EFILE, EXIST = FEXIST)
C      IF (FEXIST) THEN
C        OPEN (UNIT = ELUN, FILE = EFILE, STATUS = 'OLD', 
C     &    POSITION = 'APPEND')
C      ELSE
C        OPEN (UNIT = ELUN, FILE = EFILE, STATUS = 'NEW')
C        WRITE(ELUN,'("*RUN-TIME ERRORS OUTPUT FILE",//)')
C      ENDIF
C
C      CALL HEADER(SEASINIT, ELUN, CONTROL%RUN)
C      WRITE(ELUN,'(A,", Trt",I5)') CONTROL%FILEX, CONTROL%TRTNUM
C
C      CALL GETARG(0,PATHX)
C!      call path_adj(pathx)
C      call get_dir(pathx,errorx)
C      errorx = trim(errorx)//'MODEL.ERR'
C
C!     If ERRORX file is not in executable directory, try std. location
C      INQUIRE (FILE = ERRORX, EXIST = FEXIST)
C      IF (.NOT. FEXIST) THEN
C        SAVE_ERRORX = ERRORX
C        ERRORX = trim(STDPATH) // 'MODEL.ERR'
C      ENDIF
C
C      INQUIRE (FILE = ERRORX,EXIST = FEXIST)
C      IF (FEXIST) THEN
C
C         CALL GETLUN('ERRORX', LUN)
C         OPEN (LUN,FILE=ERRORX,STATUS='OLD')
CC
CC        Initialization
CC
C         FOUND = .FALSE.
C         IF (ERRNUM .GT. 6000 .OR. ERRNUM .LT. 0) THEN
C            KEY = 'MISC  '
C         ELSE
C            KEY = ERRKEY
C         ENDIF
CC
CC        Loop to search for error message in file MODEL.ERR.
CC
C   10    DO WHILE(.TRUE.)
C           READ (LUN,'(A)',END=20) LINE
C           AKEY = LINE(1:6)
C           IF (AKEY .EQ. KEY) THEN
C              READ (LINE,'(6X,I5)') ANUM
C              IF (ANUM .EQ. ERRNUM) THEN
C                 FOUND = .TRUE.
C                 GOTO 20
C              ENDIF
C            ELSE
C              FOUND = .FALSE.
C           ENDIF
C         ENDDO
C
C   20    IF (FOUND) THEN
C            WRITE (*,*)
C            WRITE (ELUN,*)
C   30       READ  (LUN,'(A)',END=40) LINE
C            IF (LINE .NE. BLANK) THEN
C               WRITE (*,*) LINE
C               WRITE (ELUN,*) LINE
C               WRITE(MSG(IMSG),'(A77)') LINE  ; IMSG = IMSG+1
C               GOTO 30
C            ENDIF
C          ELSEIF (KEY .NE. 'GENERI') THEN
C          !Check for generic message numbers
C             KEY = 'GENERI'
C             REWIND (LUN)
C             GO TO 10
C
C!        As an alternate, could have generic messages generated in code.
C!            CALL GENERIC_MSG(ERRNUM, LINE)
C!            WRITE (*,'(/,A78,/)') LINE
C!            WRITE (ELUN,'(/,A78,/)') LINE
C!            WRITE (MSG(IMSG),'(A78)') LINE
C!            IMSG = IMSG + 1
C
C          ELSE
C!           Could not find error message in file
C            WRITE (MSG(IMSG),'(A,A,I5)') 'Unknown ERROR. ',
C     &           'Error number: ',ERRNUM
C            WRITE (ELUN,'(/,A78,/)') MSG(IMSG)
C            WRITE (*,'(/,A78)') MSG(IMSG)
C            IMSG = IMSG + 1
C          ENDIF
C
C   40    IF (FILE .EQ. ' ') THEN
C            WRITE (*,'(2A/)') 'Error key: ',ERRKEY
C            WRITE (ELUN,'(2A/)') 'Error key: ',ERRKEY
C            WRITE (MSG(IMSG),'(2A)') 'Error key: ',ERRKEY
C            IMSG = IMSG + 1
C          ELSE
C            I = MIN(LEN(TRIM(FILE)),37)
C            WRITE (*,'(3A,I5,2A/)')
C     &    'File: ',FILE(1:I),'   Line: ',LNUM,'   Error key: ',ERRKEY
C            WRITE (ELUN,'(3A,I5,2A/)')
C     &    'File: ',FILE(1:I),'   Line: ',LNUM,'   Error key: ',ERRKEY
C            WRITE (MSG(IMSG),'(2A)') 'File: ',TRIM(FILE(1:I))
C            WRITE (MSG(IMSG+1),'(A,I5)') '   Line: ',LNUM
C            WRITE (MSG(IMSG+2),'(2A)') '   Error key: ',ERRKEY
C            IMSG = IMSG + 3
C         ENDIF
C         CLOSE (LUN)
C      ELSE
CC                                                                !BDB
CC        Tell user that error file can not be found and give a   !BDB
CC        generic error message.                                  !BDB
CC                                                                !BDB
C         ERRORX = SAVE_ERRORX
C         WRITE (*,50) TRIM(ERRORX)
C         WRITE (ELUN,50) TRIM(ERRORX)
C         WRITE (MSG(IMSG),51) TRIM(ERRORX)
C         IMSG = IMSG + 1
C   50    FORMAT('Could not locate error file: ',A,/)
C   51    FORMAT('Could not locate error file: ',A48)
C
C         !Check for generic message numbers
C         CALL GENERIC_MSG(ERRNUM, LINE)
C         WRITE (*,'(/,A78,/)') LINE(1:78)
C         WRITE (ELUN,'(/,A78,/)') LINE(1:78)
C         WRITE (MSG(IMSG),'(A78)') LINE(1:78)
C         IMSG = IMSG + 1
C
C         WRITE (*,60)  FILE, LNUM, ERRKEY   
C         WRITE (ELUN,60)  FILE, LNUM, ERRKEY   
C         WRITE (MSG(IMSG),60) FILE, LNUM, ERRKEY
C         IMSG = IMSG + 1
C   60    FORMAT('File: ',A12,'   Line: ',I5,' Error key: ',A)
C      ENDIF
C
C      WRITE(ELUN,70)
C   70 FORMAT("Additional information may be available ",
C     &            "in WARNING.OUT file.")
C      WRITE (*,70) 
C      WRITE (*, *) CHAR(7)
C      WRITE (*,260)
C260   FORMAT (/,1X,'Please press < ENTER > key to continue ',2X,$)
CC-GH      READ  (*, *)
C
C      CLOSE (ELUN)
C
C      WRITE(MSG(IMSG),'(A)') "Simulations terminated."
C      CALL WARNING(IMSG, ERRKEY, MSG)
C
C!      INQUIRE (FILE = "LUN.LST", EXIST = FEXIST)
C!      IF (FEXIST) THEN
C!        CALL GETLUN('LUN.LST', LUN)
C!        INQUIRE (UNIT = LUN, OPENED = FOPEN) 
C!        IF (.NOT. FOPEN) THEN
C!          OPEN (FILE="LUN.LST", UNIT=LUN, ERR=99, STATUS='OLD')
C!        ELSE
C!          REWIND(LUN)
C!        ENDIF
C!
C!        !Skip over first 3 lines in LUN.LST file
C!        DO I=1,3
C!          READ(LUN,'(A80)') LINE
C!        ENDDO
C!
C!        !Read list of unit numbers that have been opened and close each
C!!       EOF not portable
C!!       DO WHILE (.NOT. EOF(LUN))
C!        ERR = 0
C!        DO WHILE (ERR == 0)
C!          READ(LUN, '(A)', IOSTAT=ERRNUM, ERR=99, END=99) LINE
C!          READ(LINE,'(I5)',IOSTAT=ERRNUM, ERR=99) LUNIT
C!          IF (ERRNUM /= 0) EXIT
C!          ERR = ERRNUM
C!          IF (LUNIT .NE. LUN) THEN
C!            CLOSE (LUNIT)
C!          ENDIF
C!        ENDDO
C!        CLOSE (LUN)
C!      ENDIF
C!
C   99 STOP 99
      END SUBROUTINE ERROR
C
C!=========================================================================
C      SUBROUTINE GENERIC_MSG(ERRNUM, MESSAGE)
C!     If error messages cannot be found in MODEL.ERR file, or if MODEL.ERR
C!     file cannot be found, check for generic message type.
C
C      IMPLICIT NONE
C      INTEGER ERRNUM
C      CHARACTER*(*) MESSAGE
C
C      !Check for generic message numbers
C      SELECT CASE(ERRNUM)
C        CASE(29)
C          WRITE(MESSAGE,35) 'File not found. Please check ',
C     &      'file name or create file. Error number: ', ERRNUM 
C        CASE(33)
C          WRITE(MESSAGE,35) 'End of file encountered. ',
C     &      'Error number: ',ERRNUM
C        CASE(59)
C          WRITE(MESSAGE,35) 'Syntax error. ',
C     &      'Error number: ',ERRNUM
C        CASE(64)
C          WRITE(MESSAGE,35) 'Invalid format in file. ',
C     &      'Error number: ', ERRNUM
C        CASE DEFAULT 
C          WRITE(MESSAGE,35) 'Unknown ERROR. ',
C     &      'Error number: ',ERRNUM
C      END SELECT
C
C   35 FORMAT(A,A,I5)
C
C      END SUBROUTINE GENERIC_MSG
C!=========================================================================
C
C!=======================================================================
C! ErrorCode, Subroutine, C.H. Porter, 02/09/2010
C! Ends a run for errors by setting YREND variable.  Continue with next
C!     simulation in batch.  Stops sequence simulation.
C
C!-----------------------------------------------------------------------
C! REVISION HISTORY
C! 02/09/2010 CHP Written
C!-----------------------------------------------------------------------
      SUBROUTINE ErrorCode(CONTROL, ErrCode, ERRKEY, YREND)

      USE ModuleDefs
      USE ModuleData
      IMPLICIT NONE

      CHARACTER(*) ERRKEY 
      CHARACTER*78 MSG(4)
      INTEGER ErrCode, YREND
      TYPE (ControlType) CONTROL

C!-----------------------------------------------------------------------
C      YREND = CONTROL%YRDOY
C      CONTROL % ErrCode = ErrCode
C      CALL PUT(CONTROL)
C
C!     For sequence runs, stop run with any error
C      IF(INDEX('FQ',CONTROL%RNMODE) > 0)CALL ERROR(ERRKEY,ErrCode,' ',0)
C
C      WRITE(MSG(1),'(A,I8,A)') "Run",CONTROL%RUN, " will be terminated."
C      CALL WARNING(1,ERRKEY,MSG)
C
      RETURN
      END SUBROUTINE ErrorCode
C!=======================================================================
C! Current error codes:
C
C! Daily weather data
C!  1 Header section not found in weather file.
C!  2 Solar radiation data error
C!  3 Precipitation data error
C!  4 Tmax and Tmin are both set to 0
C!  5 Tmax and Tmin have identical values
C!  6 Tmax is less than Tmin
C!  8 Non-sequential data in file
C! 10 Weather record not found
C! 29 Weather file not found
C! 30 Error opening weather file
C! 59 Invalid format in weather file
C! 64 Syntax error.
C!
C! Weather modification
C! 72 Solar radiation data error
C! 73 Precipitation data error
C! 74 Tmax and Tmin are both set to 0
C! 75 Tmax and Tmin have identical values
C! 76 Tmax is less than Tmin
C!
C! Generated weather data
C! 82 Solar radiation data error
C! 83 Precipitation data error
C! 84 Tmax and Tmin are both set to 0
C! 85 Tmax and Tmin have identical values
C! 86 Tmax is less than Tmin
C
C!100 Number of cohorts exceeds maximum.
