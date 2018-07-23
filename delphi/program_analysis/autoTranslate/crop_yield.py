from typing import List

def UPDATE_EST(RAIN: List[float], TOTAL_RAIN: List[float], YIELD_EST: List[float]):
    TOTAL_RAIN[0] = (TOTAL_RAIN[0] + RAIN[0])
    if (TOTAL_RAIN[0] <= 40):
        YIELD_EST[0] = (-((((TOTAL_RAIN[0] - 40) ** 2) / 16)) + 100)
    else:
        YIELD_EST[0] = (-(TOTAL_RAIN[0]) + 140)

def CROP_YIELD():
    DAY: List[int] = [0]
    RAIN: List[float] = [0.0]
    YIELD_EST: List[float] = [0.0]
    TOTAL_RAIN: List[float] = [0.0]
    MAX_RAIN: List[float] = [0.0]
    CONSISTENCY: List[float] = [0.0]
    ABSORBTION: List[float] = [0.0]
    MAX_RAIN[0] = 4.0
    CONSISTENCY[0] = 64.0
    ABSORBTION[0] = 0.6
    YIELD_EST[0] = 0
    TOTAL_RAIN[0] = 0
    for DAY[0] in range(1, 31+1):
        RAIN[0] = ((-((((DAY[0] - 16) ** 2) / CONSISTENCY[0])) + MAX_RAIN[0]) * ABSORBTION[0])
        UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        print("Day ", DAY, " Estimate: ", YIELD_EST)
    print("Crop Yield(%): ", YIELD_EST)

CROP_YIELD()
