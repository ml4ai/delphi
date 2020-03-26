import sys
import os
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from delphi.translators.for2py.static_save import *
from delphi.translators.for2py.strings import *
from delphi.translators.for2py import intrinsics
from dataclasses import dataclass
from delphi.translators.for2py.types_ext import Float32
import delphi.translators.for2py.math_ext as math
from numbers import Real
from random import random


def update_est(rain: List[float], total_rain: List[float], yield_est: List[float]):
    total_rain[0] = (total_rain[0] + rain[0])
    if (total_rain[0] <= 40):
        yield_est[0] = (-((((total_rain[0] - 40) ** 2) / 16)) + 100)
    else:
        yield_est[0] = (-(total_rain[0]) + 140)

def crop_yield():
    day: List[int] = [None]
    rain: List[float] = [None]
    yield_est: List[float] = [None]
    total_rain: List[float] = [None]
    max_rain: List[float] = [None]
    consistency: List[float] = [None]
    absorption: List[float] = [None]
    max_rain[0] = 4.0
    consistency[0] = 64.0
    absorption[0] = 0.6
    yield_est[0] = 0
    total_rain[0] = 0
    for day[0] in range(1, 31+1):
        rain[0] = ((-((((day[0] - 16) ** 2) / consistency[0])) + max_rain[0]) * absorption[0])
        update_est(rain, total_rain, yield_est)
        print("day ", day, " estimate: ", yield_est)
    print("crop yield(%): ", yield_est)

crop_yield()
