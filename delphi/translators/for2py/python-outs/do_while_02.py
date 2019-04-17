import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def main():
    day: List[int] = [0]
    rain: List[float] = [0.0]
    yield_est: List[float] = [0.0]
    total_rain: List[float] = [0.0]
    news: List[float] = [0.0]
    max_rain: List[float] = [0.0]
    consistency: List[float] = [0.0]
    absorbtion: List[float] = [0.0]
    max_rain[0] = 4.0
    consistency[0] = 64.0
    absorbtion[0] = 0.6
    yield_est[0] = 0
    total_rain[0] = 0
    day[0] = 1
    while (day[0] <= 31):
        print("(", day, consistency, max_rain, absorbtion, ")")
        rain[0] = ((-((((day[0] - 16) ** 2) / consistency[0])) + max_rain[0]) * absorbtion[0])
        print(rain)
        yield_est[0] = update_est(rain, total_rain, yield_est)
        news[0] = test_func(total_rain, yield_est)
        print("Day ", day, " Estimate: ", yield_est)
        day[0] = (day[0] + 1)
    print("Crop Yield(%): ", yield_est)
    print("News: ", news)
    
def update_est(rain, total_rain, yield_est):
    rain: List[float]
    yield_est: List[float]
    total_rain: List[float]
    total_rain[0] = (total_rain[0] + rain[0])
    if (total_rain[0] <= 40):
        yield_est[0] = (-((((total_rain[0] - 40) ** 2) / 16)) + 100)
    else:
        yield_est[0] = (-(total_rain[0]) + 140)
    return yield_est[0]

def test_func(total_rain, yield_est):
    total_rain: List[float]
    yield_est: List[float]
    new_var: List[float] = [0.0]
    new_var[0] = 5.0
    if (new_var[0] <= 4.0):
        return 17.0
    else:
        return yield_est[0]

main()
