import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass


tmax: List[float] = [30.0]
tmin: List[float] = [20.0]
windrun: List[float] = [10.0]
xhlai: List[float] = [0.8]
xlat: List[float] = [40.0]
meevp: List[str] = ['A']
canht: List[float] = [10.0]
msalb: List[float] = [0.9]
srad: List[float] = [20.41]
tdew: List[float] = [25.0]
windht: List[float] = [10.0]
xelev: List[float] = [100.0]

def update_vars():
    canht[0] = (canht[0] + 1.0)
    msalb[0] = (msalb[0] * 1.2)
    srad[0] = (srad[0] + 0.02)
    tdew[0] = (tdew[0] + 2.0)
    windht[0] = (windht[0] * 1.2)
    xelev[0] = (xelev[0] + 500.0)