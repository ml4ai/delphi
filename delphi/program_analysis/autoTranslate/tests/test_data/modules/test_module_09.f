C File: test_module_09.f
C Purpose: This code is taken from the DSSAT file ModuleDefs.for.  It tests
C          the handling of multiple constant declarations.
C
C Compile and run this program as follows:
C
C    gfortran -c test_module_09.f     # << this will create a file "mymod8.mod"
C    gfortran test_module_09.f        # << this will create a file "a.out"
C
C When executed this program generates the following output:
C
C          20
C          24
C        9000
C         300
C           3
C        1000
C          42
C          40
C         500
C         500
C   3.14159274    
C   1.74532924E-02
C           1
C           2
C           2
C           3
C           3


      MODULE MYMOD9
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
     &    EMERG    = 3
      END MODULE MYMOD9

      PROGRAM PGM
      USE mymod9

      write (*,*) NL
      write (*,*) TS
      write (*,*) NAPPL
      write (*,*) NCOHORTS
      write (*,*) NELEM
      write (*,*) NumOfDays
      write (*,*) NumOfStalks
      write (*,*) EvaluateNum
      write (*,*) MaxFiles
      write (*,*) MaxPest
      write (*,*) PI
      write (*,*) RAD
      write (*,*) RUNINIT
      write (*,*) INIT
      write (*,*) SEASINIT
      write (*,*) RATE
      write (*,*) EMERG

      stop
      end program PGM
