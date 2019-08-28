from numbers import Real
import delphi.translators.for2py.math_ext as math

def petpt__assign__td_0(tmax: Real, tmin: Real):
    return ((0.6*tmax)+(0.4*tmin))

def petpt__condition__IF_1_0(xhlai: Real):
    return (xhlai <= 0.0)

def petpt__assign__albedo_0(msalb: Real):
    return msalb

def petpt__assign__albedo_1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*math.exp(-((0.75*xhlai)))))

def petpt__decision__albedo_2(IF_1_0: bool, albedo_1: Real, albedo_0: Real):
    return albedo_0 if IF_1_0 else albedo_1

def petpt__assign__slang_0(srad: Real):
    return (srad*23.923)

def petpt__assign__eeq_0(slang: Real, albedo: Real, td: Real):
    return ((slang*(0.000204-(0.000183*albedo)))*(td+29.0))

def petpt__assign__eo_0(eeq: Real):
    return (eeq*1.1)

def petpt__condition__IF_2_0(tmax: Real):
    return (tmax > 35.0)

def petpt__assign__eo_1(eeq: Real, tmax: Real):
    return (eeq*(((tmax-35.0)*0.05)+1.1))

def petpt__assign__eo_2(eeq: Real, tmax: Real):
    return ((eeq*0.01)*math.exp((0.18*(tmax+20.0))))

def petpt__decision__eo_3(IF_2_0: bool, eo_0: Real, eo_1: Real):
    return eo_1 if IF_2_0 else eo_0

def petpt__condition__IF_2_1(tmax: Real):
    return (tmax < 5.0)

def petpt__decision__eo_4(IF_2_1: bool, eo_3: Real, eo_2: Real):
    return eo_2 if IF_2_1 else eo_3

def petpt__assign__eo_5(eo: Real):
    return max(eo, 0.0001)

