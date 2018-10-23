************************************************************************
*     UPDATE_EST - Updates the estimated yield of magic beans given 
*       some additional amount of rainfall
************************************************************************
*
*     VARIABLES
*     
*     INPUT RAIN      = Additional rainfall
*
*     INPUT YIELD_EST = Crop yield to update
*
************************************************************************
      SUBROUTINE UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        DOUBLE PRECISION RAIN, YIELD_EST, TOTAL_RAIN
        TOTAL_RAIN = TOTAL_RAIN + RAIN

*       Yield increases up to a point
        IF(TOTAL_RAIN .le. 40) THEN
            YIELD_EST = -(TOTAL_RAIN - 40) ** 2 / 16 + 100

*       Then sharply declines
        ELSE
            YIELD_EST = -TOTAL_RAIN + 140
        ENDIF

      END SUBROUTINE UPDATE_EST

************************************************************************
*     CROP_YIELD - Estimate the yield of magic beans given a simple 
*       model for rainfall
************************************************************************
*
*     VARIABLES
*     
*     INPUT MAX_RAIN   = The maximum rain for the month
*     INPUT CONSISTENCY = The consistency of the rainfall 
*       (higher = more consistent)
*     INPUT ABSORPTION = Estimates the % of rainfall absorbed into the
*       soil (i.e. % lost due to evaporation, runoff)
*
*     OUTPUT YIELD_EST = The estimated yield of magic beans
*
*     DAY              = The current day of the month
*     RAIN             = The rainfall estimate for the current day
*
************************************************************************
      PROGRAM CROP_YIELD
      IMPLICIT NONE

      INTEGER DAY,DOM,DESC
      DOUBLE PRECISION RAIN, YIELD_EST, TOTAL_RAIN
      DOUBLE PRECISION  MAX_RAIN, CONSISTENCY, ABSORPTION

*      MAX_RAIN = 4.0
*      CONSISTENCY = 64.0
*      ABSORPTION = 0.6
      
*     Open the file
      OPEN (5,FILE='CROP_YIELD_DATA.INP',STATUS='OLD',ACTION='READ')

      YIELD_EST = 0
      TOTAL_RAIN = 0
 
      DO 20 DAY=1,31
*       Compute rainfall for the current day
        READ(5,30,IOSTAT=DESC) DOM,MAX_RAIN,CONSISTENCY,ABSORPTION
   30   FORMAT(I2,2X,F3.1,2X,F4.1,2X,F3.1)
        IF (DESC<0) THEN
            GOTO 50
        END IF
       
        RAIN = (-(DOM - 16) ** 2 / CONSISTENCY + MAX_RAIN) * ABSORPTION

*       UPDATE RAINFALL ESTIMATE 
        CALL UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        PRINT *, "Day ", DAY, " Estimate: ", YIELD_EST

   20 END DO

      PRINT *, "Crop Yield(%): ", YIELD_EST

   50 CLOSE(5) 

      END PROGRAM CROP_YIELD
