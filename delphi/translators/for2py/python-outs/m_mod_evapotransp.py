import sys
from typing import List
import math
from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *
from dataclasses import dataclass



def _refet(ud, rn, g, pcon, cn, tavg, w2m, es, ea, cd):
    ud: List[float]
    rn: List[float]
    g: List[float]
    pcon: List[float]
    cn: List[float]
    tavg: List[float]
    w2m: List[float]
    es: List[float]
    ea: List[float]
    cd: List[float]
    rval: List[float] = [0.0]
    rval[0] = (((0.408 * ud[0]) * (rn[0] - g[0])) + (((pcon[0] * (cn[0] / (tavg[0] + 273.0))) * w2m[0]) * (es[0] - ea[0])))
    rval[0] = (rval[0] / (ud[0] + (pcon[0] * (1.0 + (cd[0] * w2m[0])))))
    return max(0.0001, rval[0])

def ev_transp(ud, rn, g, pcon, cn, tavg, w2m, es, ea, cd, kc):
    ud: List[float]
    rn: List[float]
    g: List[float]
    pcon: List[float]
    cn: List[float]
    tavg: List[float]
    w2m: List[float]
    es: List[float]
    ea: List[float]
    cd: List[float]
    kc: List[float]
    evtransp: List[float] = [0.0]
    evtransp[0] = (kc[0] * _refet(ud, rn, g, pcon, cn, tavg, w2m, es, ea, cd))
    return max(evtransp[0], 0.0001)