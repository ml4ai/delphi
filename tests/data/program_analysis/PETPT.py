import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


def petpt(msalb: List[float], srad: List[float], tmax: List[float], tmin: List[float], xhlai: List[float], eo: List[float]):
    albedo: List[float] = [None]
    eeq: List[float] = [None]
    slang: List[float] = [None]
    td: List[float] = [None]
    td[0] = ((0.60 * tmax[0]) + (0.40 * tmin[0]))
    if (xhlai[0] <= 0.0):
        albedo[0] = msalb[0]
    else:
        albedo[0] = (0.23 - ((0.23 - msalb[0]) * math.exp(-((0.75 * xhlai[0])))))
    slang[0] = (srad[0] * 23.923)
    eeq[0] = ((slang[0] * (2.04E-4 - (1.83E-4 * albedo[0]))) * (td[0] + 29.0))
    eo[0] = (eeq[0] * 1.1)
    if (tmax[0] > 35.0):
        eo[0] = (eeq[0] * (((tmax[0] - 35.0) * 0.05) + 1.1))
    else:
        if (tmax[0] < 5.0):
            eo[0] = ((eeq[0] * 0.01) * math.exp((0.18 * (tmax[0] + 20.0))))
    eo[0] = max(eo[0], 0.0001)
    