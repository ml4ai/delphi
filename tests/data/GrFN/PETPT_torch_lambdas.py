# import numpy as np
# from numba import jit
import torch


# @jit
def petpt__assign__td_1(tmax, tmin):
    return ((0.6*tmax)+(0.4*tmin))


# @jit
def petpt__decision__albedo_1(xhlai, msalb):
    return torch.where(xhlai <= 0.0, 0.23-((0.23-msalb)*torch.exp(-((0.75*xhlai)))), msalb)


# @jit
def petpt__assign__slang_1(srad):
    return srad*23.923


# @jit
def petpt__assign__eeq_1(slang, albedo, td):
    return (slang*(0.000204-(0.000183*albedo)))*(td+29.0)


# @jit
def petpt__assign__eo_0(eeq):
    return eeq*1.1


# @jit
def petpt__decision__eo_1(tmax, eeq, eo_0):
    return torch.where(tmax > 35.0, eeq*(((tmax-35.0)*0.05)+1.1), eo_0)


# @jit
def petpt__decision__eo_2(tmax, eeq, eo_1):
    return torch.where(tmax < 5.0, (eeq*0.01)*torch.exp((0.18*(tmax+20.0))), eo_1)


# @jit
def petpt__assign__eo_3(eo):
    eo = eo.reshape((len(eo), 1))
    res = torch.max(torch.cat((eo, torch.full_like(eo, 0.0001)), dim=1), dim=1)
    return res[0]
