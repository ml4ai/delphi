**************************************************************************
**************************************************************************
*        DRIVER for the  MODEL TO SIMULATE CROP GROWTH SUBJECTED TO             
*           DAILY VARIATIONS OF WEATHER AND SOIL WATER CONDITIONS      
*   Written in Microsoft FORTRAN for PC-compatible machines              
*   Authors: RICARDO BRAGA and JADIR ROSA                                
*   Obs: This program is an assignment of the course AGE 5646-Agricultural
*      and Biological Systems Simulation.
*   Date: 03/31/1997
*   Modified 7/99 CHP - modified modular format, revised output format, 
*         modified soil water routines, added water balance 
C*************************************************************************
*
*     LIST OF VARIABLES
*
*     DOY    = Julian day 
*     DOYP   = date of planting (Julian day)
*     endsim = code signifying physiological maturity (end of simulation)
*     FROP   = frequency of printout (days)
*     IPRINT = code for printout (=0 for printout)
*     LAI    = canopy leaf area index (m2 m-2)
*     PAR    = photosynthetically active radiation (MJ/m2/d)
*     RAIN   = daily rainfall (mm)
*     SRAD   = daily solar radiation (MJ/m2/d)
*     SWFAC1 = soil water deficit stress factor 
*     SWFAC2 = soil water excess stress factor
*     TAIRHR = hourly average temperature (Celsius)
*     TMAX   = daily maximum temperature (Celsius)
*     TMIN   = daily minimum temperature (Celsius)
*
***********************************************************************

      PROGRAM MAIN

!-----------------------------------------------------------------------
      USE DFLIB
      IMPLICIT NONE

      REAL LAI, SWFAC1, SWFAC2
      REAL SRAD, TMAX, TMIN, PAR, RAIN
      INTEGER DOY,DOYP, endsim
      INTEGER FROP, IPRINT

!************************************************************************
!************************************************************************
!     INITIALIZATION AND INPUT OF DATA
!************************************************************************
      CALL OPENF(DOYP, FROP)
      
      CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'INITIAL   ')

      CALL SW(
     &    DOY, LAI, RAIN, SRAD, TMAX, TMIN,               !Input
     &    SWFAC1, SWFAC2,                                 !Output
     &    'INITIAL   ')                                   !Control

      CALL PLANT(DOY, endsim, TMAX, TMIN,                 !Input
     &    PAR, SWFAC1, SWFAC2,                            !Input
     &    LAI,                                            !Output
     &    'INITIAL   ')                                   !Control

!-----------------------------------------------------------------------
!     DAILY TIME LOOP 
!-----------------------------------------------------------------------
      DO 500 DOY = 0,1000
        IF (DOY .NE. 0) THEN

          CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'RATE      ')


!************************************************************************
!************************************************************************
!     RATE CALCULATIONS
!************************************************************************
          CALL SW(
     &      DOY, LAI, RAIN, SRAD, TMAX, TMIN,             !Input
     &      SWFAC1, SWFAC2,                               !Output
     &      'RATE      ')                                 !Control

          IF (DOY .GT. DOYP) THEN
            CALL PLANT(DOY, endsim,TMAX,TMIN,             !Input
     &        PAR, SWFAC1, SWFAC2,                        !Input
     &        LAI,                                        !Output
     &        'RATE      ')                               !Control
          ENDIF

!************************************************************************
!************************************************************************
!     INTEGRATION OF STATE VARIABLES
!************************************************************************
          CALL SW(
     &      DOY, LAI, RAIN, SRAD, TMAX, TMIN,             !Input
     &      SWFAC1, SWFAC2,                               !Output
     &      'INTEG     ')                                 !Control

          IF (DOY .GT. DOYP) THEN
            CALL PLANT(DOY, endsim, TMAX,TMIN,            !Input
     &        PAR, SWFAC1, SWFAC2,                        !Input
     &        LAI,                                        !Output
     &        'INTEG     ')                               !Control
          ENDIF

        ENDIF

!************************************************************************
!************************************************************************
!     WRITE DAILY OUTPUT
!************************************************************************

        IPRINT = MOD(DOY, FROP)
        IF ((IPRINT .EQ. 0) .OR. (endsim .EQ. 1) .OR. 
     &        (DOY .EQ. DOYP)) THEN

          CALL SW(
     &      DOY, LAI, RAIN, SRAD, TMAX, TMIN,             !Input
     &      SWFAC1, SWFAC2,                               !Output
     &      'OUTPUT    ')                                 !Control

          IF (DOY .GE. DOYP) THEN
            CALL PLANT(DOY, endsim, TMAX,TMIN,            !Input
     &        PAR, SWFAC1, SWFAC2,                        !Input
     &        LAI,                                        !Output
     &        'OUTPUT    ')                               !Control
          ENDIF

        ENDIF

        IF (ENDSIM .EQ. 1) EXIT

!-----------------------------------------------------------------------
!     END OF DAILY TIME LOOP 
!-----------------------------------------------------------------------
  500 ENDDO

!************************************************************************
!************************************************************************
!     CLOSE FILES AND WRITE SUMMARY REPORTS
!************************************************************************
      CALL WEATHR(SRAD,TMAX,TMIN,RAIN,PAR,'CLOSE     ') 

      CALL SW(
     &  DOY, LAI, RAIN, SRAD, TMAX, TMIN,                 !Input
     &  SWFAC1, SWFAC2,                                   !Output
     &  'CLOSE     ')                                     !Control
      
      CALL PLANT(DOY, endsim, TMAX,TMIN,                  !Input    
     &    PAR, SWFAC1, SWFAC2,                            !Input
     &    LAI,                                            !Output
     &    'CLOSE     ') 

      PAUSE 'End of Program - hit enter key to end'

!-----------------------------------------------------------------------  
      STOP
      END PROGRAM MAIN
****************************************************************************



************************************************************************
*     SUBROUTINE OPENF(DOYP)
*     This subroutine opens the simulation control file, and reads date of
*     planting (DOYP)
*
*     SIMCTRL.INP => date of planting, frequency of printout
************************************************************************

      SUBROUTINE OPENF(DOYP, FROP)

      IMPLICIT NONE
      INTEGER DOYP, FROP

      OPEN (UNIT=8, FILE='SIMCTRL.INP',STATUS='UNKNOWN')
      READ(8,5) DOYP, FROP
      IF (FROP .LE. 0) FROP = 1
    5 FORMAT(2I6)
      CLOSE(8)

!-----------------------------------------------------------------------  
      RETURN
      END SUBROUTINE OPENF
************************************************************************
************************************************************************
