def UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST):  # UPDATE_EST
  TOTAL_RAIN[0] = (TOTAL_RAIN[0] + RAIN[0])  # UPDATE_EST__assign___TOTAL_RAIN
  if (TOTAL_RAIN[0] <= 40):
    YIELD_EST[0] = (-((((TOTAL_RAIN[0] - 40) ** 2) / 16)) + 100)  # UPDATE_EST__assign___YIELD_EST_1
  else:
    YIELD_EST[0] = (-(TOTAL_RAIN[0]) + 140)  # UPDATE_EST__assign___YIELD_EST_2
def CROP_YIELD():  # CROP_YIELD
  DAY = [0]
  RAIN = [0.0]
  YIELD_EST = [0.0]
  TOTAL_RAIN = [0.0]
  MAX_RAIN = [0.0]
  CONSISTENCY = [0.0]
  ABSORBTION = [0.0]
  MAX_RAIN[0] = 4.0      # CROP_YIELD__assign___CROP_YIELD__MAX_RAIN
  CONSISTENCY[0] = 64.0  # CROP_YIELD__assign___CROP_YIELD__CONSISTENCY
  ABSORBTION[0] = 0.6    # CROP_YIELD__assign___CROP_YIELD__ABSORPTION
  YIELD_EST[0] = 0       # CROP_YIELD__assign___CROP_YIELD__YIELD_EST
  TOTAL_RAIN[0] = 0      # CROP_YIELD__assign___CROP_YIELD__TOTAL_RAIN
  for DAY[0] in range(1, 31+1):  # CROP_YIELD__loop_1
    RAIN[0] = ((-((((DAY[0] - 16) ** 2) / CONSISTENCY[0])) + MAX_RAIN[0]) * ABSORBTION[0])  # CROP_YIELD__loop_1__assign___CROP_YIELD_RAIN
    UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
    print("Day ", DAY, " Estimate: ", YIELD_EST)
  print("Crop Yield(%): ", YIELD_EST)

if __name__ == "__main__":
    CROP_YIELD()
