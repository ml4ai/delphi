

def UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST):
    TOTAL_RAIN = TOTAL_RAIN + RAIN

    if(TOTAL_RAIN <= 40):
        YIELD_EST = -(TOTAL_RAIN - 40) ** 2 / 16 + 100
    else:
        YIELD_EST = -TOTAL_RAIN + 140

    return TOTAL_RAIN, YIELD_EST


def CROP_YIELD():
    """
    In here we have a doc string for crop_yield. Do we want to put the variable stuff in here? Note, it does not seem like you can attach a docstring to any arbitrary place.
    """
    MAX_RAIN = 4.0
    CONSISTENCY = 64.0
    ABSORBTION = 0.6

    YIELD_EST = 0
    TOTAL_RAIN = 0

    for DAY in range(1,31+1):
        RAIN = (-(DAY - 16) ** 2 / CONSISTENCY + MAX_RAIN) * ABSORBTION

        TOTAL_RAIN, YIELD_EST = UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        print("Day " + str(DAY) + " Estimate: " + str(YIELD_EST))

    print("Crop Yield(%): " + str(YIELD_EST))

    return YIELD_EST


if __name__ == "__main__":
    CROP_YIELD()
