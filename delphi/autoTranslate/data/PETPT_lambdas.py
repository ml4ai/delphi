import math

def PETPT__lambda__TD_0(TMAX, TMIN):
    TD = ((0.6*TMAX)+(0.4*TMIN))
    return TD

def PETPT__lambda__IF_1_0(XHLAI):
    return (XHLAI<=0.0)

def PETPT__lambda__ALBEDO_0(MSALB):
    ALBEDO = MSALB
    return ALBEDO

def PETPT__lambda__ALBEDO_1(MSALB, XHLAI):
    ALBEDO = (0.23-((0.23-MSALB)*math.exp(-((0.75*XHLAI)))))
    return ALBEDO

def PETPT__lambda__SLANG_0(SRAD):
    SLANG = (SRAD*23.923)
    return SLANG

def PETPT__lambda__EEQ_0(SLANG, ALBEDO, TD):
    EEQ = ((SLANG*(0.000204-(0.000183*ALBEDO)))*(TD+29.0))
    return EEQ

def PETPT__lambda__EO_0(EEQ):
    EO = (EEQ*1.1)
    return EO

def PETPT__lambda__IF_2_0(TMAX):
    return (TMAX>35.0)

def PETPT__lambda__EO_1(EEQ, TMAX):
    EO = (EEQ*(((TMAX-35.0)*0.05)+1.1))
    return EO

def PETPT__lambda__IF_3_0(TMAX):
    return (TMAX<5.0)

def PETPT__lambda__EO_2(EEQ, TMAX):
    EO = ((EEQ*0.01)*math.exp((0.18*(TMAX+20.0))))
    return EO

def PETPT__lambda__EO_3(max, EO):
    EO = max(EO, 0.0001)
    return EO

