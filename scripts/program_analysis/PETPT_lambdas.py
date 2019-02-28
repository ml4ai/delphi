import math

def petpt__lambda__td_0(tmax, tmin):
    return ((0.6*tmax)+(0.4*tmin))

def petpt__lambda__IF_1_0(xhlai):
    return (xhlai<=0.0)

def petpt__lambda__albedo_0(msalb):
    return msalb

def petpt__lambda__albedo_1(msalb):
    return (0.23-((0.23-msalb)*math.exp(-((0.75*xhlai)))))

def petpt__lambda__slang_0(srad):
    return (srad*23.923)

def petpt__lambda__eeq_0(slang, albedo, td):
    return ((slang*(0.000204-(0.000183*albedo)))*(td+29.0))

def petpt__lambda__eo_0(eeq):
    return (eeq*1.1)

def petpt__lambda__IF_2_0(tmax):
    return (tmax>35.0)

def petpt__lambda__eo_1(eeq, tmax):
    return (eeq*(((tmax-35.0)*0.05)+1.1))

def petpt__lambda__IF_3_0(tmax):
    return (tmax<5.0)

def petpt__lambda__eo_2(eeq):
    return ((eeq*0.01)*math.exp((0.18*(tmax+20.0))))

def petpt__lambda__eo_3(eo):
    return max(eo, 0.0001)
