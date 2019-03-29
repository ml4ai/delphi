import math

def petpt__assign__td_1(tmax, tmin):
    return ((0.6*tmax)+(0.4*tmin))

def petpt__condition__IF_1_0(xhlai):
    return (xhlai <= 0.0)

def petpt__assign__albedo_1(msalb):
    return msalb

def petpt__assign__albedo_2(msalb, xhlai):
    return (0.23-((0.23-msalb)*math.exp(-((0.75*xhlai)))))

def petpt__decision__albedo_3(IF_1_0, albedo_2, albedo_1):
    return albedo_1 if IF_1_0 else albedo_2

def petpt__assign__slang_1(srad):
    return (srad*23.923)

def petpt__assign__eeq_1(slang, albedo, td):
    return ((slang*(0.000204-(0.000183*albedo)))*(td+29.0))

def petpt__assign__eo_0(eeq):
    return (eeq*1.1)

def petpt__condition__IF_1_1(tmax):
    return (tmax > 35.0)

def petpt__assign__eo_1(eeq, tmax):
    return (eeq*(((tmax-35.0)*0.05)+1.1))

def petpt__assign__eo_2(eeq, tmax):
    return ((eeq*0.01)*math.exp((0.18*(tmax+20.0))))

def petpt__decision__eo_3(IF_1_1, eo_0, eo_1):
    return eo_1 if IF_1_1 else eo_0

def petpt__condition__IF_2_0(tmax):
    return (tmax < 5.0)

def petpt__decision__eo_4(IF_2_0, eo_3, eo_2):
    return eo_2 if IF_2_0 else eo_3

def petpt__assign__eo_5(eo):
    return max(eo, 0.0001)

