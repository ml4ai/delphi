from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
import delphi.translators.for2py.math_ext as math

def PETPT__petpt__assign__td__0(tmax: Real, tmin: Real):
    return ((0.6*tmax)+(0.4*tmin))

def PETPT__petpt__condition__IF_0__0(xhlai: Real):
    return (xhlai <= 0.0)

def PETPT__petpt__assign__albedo__0(msalb: Real):
    return msalb

def PETPT__petpt__assign__albedo__1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*np.exp(-((0.75*xhlai)))))

def PETPT__petpt__decision__albedo__2(albedo_0: Array, albedo_1: Array, IF_0_0: bool):
    return np.where(IF_0_0, albedo_1, albedo_0)

def PETPT__petpt__assign__slang__0(srad: Real):
    return (srad*23.923)

def PETPT__petpt__assign__eeq__0(slang: Real, albedo: Real, td: Real):
    return ((slang*(0.000204-(0.000183*albedo)))*(td+29.0))

def PETPT__petpt__assign__eo__0(eeq: Real):
    return (eeq*1.1)

def PETPT__petpt__condition__IF_1__0(tmax: Real):
    return (tmax > 35.0)

def PETPT__petpt__assign__eo__1(eeq: Real, tmax: Real):
    return (eeq*(((tmax-35.0)*0.05)+1.1))

def PETPT__petpt__decision__eo__2(eo_0: Real, eo_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, eo_1, eo_0)

def PETPT__petpt__condition__IF_1__1(tmax: Real):
    return (tmax < 5.0)

def PETPT__petpt__assign__eo__3(eeq: Real, tmax: Real):
    return ((eeq*0.01)*np.exp((0.18*(tmax+20.0))))

def PETPT__petpt__decision__eo__4(eo_0: Real, eo_1: Real, IF_1_1: bool):
    return np.where(IF_1_1, eo_1, eo_0)

def PETPT__petpt__assign__eo__5(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

