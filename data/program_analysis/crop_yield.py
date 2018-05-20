from typing import Tuple

def update_est(rain: float, total_rain: float,
        yield_est: float) -> Tuple[float, float]:

    total_rain += rain

    if(total_rain <= 40.):
        yield_est = -(total_rain - 40) ** 2 / 16 + 100
    else:
        yield_est = -total_rain + 140

    return total_rain, yield_est


def crop_yield() -> float:
    """
    In here we have a doc string for crop_yield. Do we want to put the variable
    stuff in here? Note, it does not seem like you can attach a docstring to any
    arbitrary place.
    """

    max_rain = 4.0
    consistency = 64.0
    absorption = 0.6

    yield_est = 0.0
    total_rain = 0.0

    for day in range(1, 31+1):
        rain = (-(day - 16) ** 2 / consistency + max_rain) * absorption

        total_rain, yield_est = update_est(rain, total_rain, yield_est)
        print(f"Day {day} Estimate: {yield_est}")

    print(f"Crop Yield(%): {yield_est}")

    return yield_est


if __name__ == "__main__":
    crop_yield()
