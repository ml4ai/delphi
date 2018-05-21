from typing import Tuple

def UPDATE_EST(RAIN: float, TOTAL_RAIN: float,
        YIELD_EST: float) -> Tuple[float, float]:

    TOTAL_RAIN += RAIN

    if(TOTAL_RAIN <= 40.):
        YIELD_EST = -(TOTAL_RAIN - 40) ** 2 / 16 + 100
    else:
        YIELD_EST = -TOTAL_RAIN + 140

    return TOTAL_RAIN, YIELD_EST


def CROP_YIELD() -> float:
    """
    In here we have a doc string for crop_yield. Do we want to put the variable
    stuff in here? Note, it does not seem like you can attach a docstring to any
    arbitrary place.
    """

    MAX_RAIN = 4.0
    CONSISTENCY = 64.0
    ABSORPTION = 0.6

    YIELD_EST = 0.0
    TOTAL_RAIN = 0.0

    for DAY in range(1, 31+1):
        RAIN = (-(DAY - 16) ** 2 / CONSISTENCY + MAX_RAIN) * ABSORPTION

        TOTAL_RAIN, YIELD_EST = UPDATE_EST(RAIN, TOTAL_RAIN, YIELD_EST)
        print(f"Day {DAY} Estimate: {YIELD_EST}")

    print(f"Crop Yield(%): {YIELD_EST}")

    return YIELD_EST


if __name__ == "__main__":
    CROP_YIELD()
