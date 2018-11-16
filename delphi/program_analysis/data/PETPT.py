from typing import List
from math import *

def PETPT(MSALB: List[float], SRAD: List[float], TMAX: List[float], TMIN: List[float], XHLAI: List[float], EO: List[float]):
    ALBEDO: List[float] = [0.0]
    EEQ: List[float] = [0.0]
    SLANG: List[float] = [0.0]
    TD: List[float] = [0.0]
    TD[0] = ((0.60 * TMAX[0]) + (0.40 * TMIN[0]))
    if (XHLAI[0] <= 0.0):
        ALBEDO[0] = MSALB[0]
    else:
        ALBEDO[0] = (0.23 - ((0.23 - MSALB[0]) * exp(-((0.75 * XHLAI[0])))))
    SLANG[0] = (SRAD[0] * 23.923)
    EEQ[0] = ((SLANG[0] * (2.04E-4 - (1.83E-4 * ALBEDO[0]))) * (TD[0] + 29.0))
    EO[0] = (EEQ[0] * 1.1)
    if (TMAX[0] > 35.0):
        EO[0] = (EEQ[0] * (((TMAX[0] - 35.0) * 0.05) + 1.1))
    else:
        if (TMAX[0] < 5.0):
            EO[0] = ((EEQ[0] * 0.01) * exp((0.18 * (TMAX[0] + 20.0))))
    EO[0] = max(EO[0], 0.0001)
    return True