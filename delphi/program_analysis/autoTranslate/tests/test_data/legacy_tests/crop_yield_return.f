************************************************************************
*     UPDATE_EST - Updates the estimated yield of magic beans given 
*       some additional amount of rainfall
************************************************************************
*
*     VARIABLES
*     
*     INPUT RAIN      = Additional rainfall
*
*     INOUT YIELD_EST = Crop yield to update
*
************************************************************************

************************************************************************
*     CROP_YIELD - Estimate the yield of magic beans given a simple 
*       model for rainfall
************************************************************************
*
*     VARIABLES
*     
*     INPUT MAX_RAIN   = The maximum rain for the month
*     INPUT CONSITENCY = The consistency of the rainfall 
*       (higher = more consistent)
*     INPUT ABSORBTION = Estimates the % of rainfall absorbed into the
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

      INTEGER DAY
      DOUBLE PRECISION RAIN, YIELD_EST, TOTAL_RAIN
      DOUBLE PRECISION MAX_RAIN, CONSISTENCY, ABSORBTION

      MAX_RAIN = 4.0
      CONSISTENCY = 64.0
      ABSORBTION = 0.6
      
      YIELD_EST = 0
      TOTAL_RAIN = 0
 
      DO 20 DAY=1,31
        PRINT *, "(", DAY, CONSISTENCY, MAX_RAIN, ABSORBTION, ")"
*       Compute rainfall for the current day
        RAIN = (-(DAY - 16) ** 2 / CONSISTENCY + MAX_RAIN) * ABSORBTION
        PRINT *, RAIN

*       Update rainfall estimate
        YIELD_EST = UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        PRINT *, "Day ", DAY, " Estimate: ", YIELD_EST

   20 ENDDO

      PRINT *, "Crop Yield(%): ", YIELD_EST

      CONTAINS
       DOUBLE PRECISION FUNCTION UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
          IMPLICIT NONE
          DOUBLE PRECISION RAIN, YIELD_EST, TOTAL_RAIN
          TOTAL_RAIN = TOTAL_RAIN + RAIN

*         Yield increases up to a point
          IF(TOTAL_RAIN .le. 40) THEN
            YIELD_EST = -(TOTAL_RAIN - 40) ** 2 / 16 + 100

*         Then sharply declines
          ELSE
              YIELD_EST = -TOTAL_RAIN + 140
          ENDIF
          UPDATE_EST = YIELD_EST

        END FUNCTION UPDATE_EST

      END PROGRAM CROP_YIELD
