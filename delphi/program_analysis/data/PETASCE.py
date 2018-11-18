from typing import List
import math


def PETASCE(CANHT: list[float], DOY: list[int], MSALB, MEEVP: list[str],
            SRAD: list[float], TDEW: list[float], TMAX: list[float],
            TMIN: list[float], WINDHT: list[float], WINDRUN: list[float],
            XHLAI: list[float], XLAT: list[float], XELEV: list[float],
            EO: list[float]):

    DOY: list[int] = [0]

    MEEVP: list[str] = [""]

    CANHT: list[float] = [0.0]
    MSALB: list[float] = [0.0]
    SRAD: list[float] = [0.0]
    TDEW: list[float] = [0.0]
    TMAX: list[float] = [0.0]
    TMIN: list[float] = [0.0]
    WINDHT: list[float] = [0.0]
    WINDRUN: list[float] = [0.0]
    XHLAI: list[float] = [0.0]
    XLAT: list[float] = [0.0]
    XELEV: list[float] = [0.0]
    EO: list[float] = [0.0]
    TAVG: list[float] = [0.0]
    PATM: list[float] = [0.0]
    PSYCON: list[float] = [0.0]
    UDELTA: list[float] = [0.0]
    EMAX: list[float] = [0.0]
    EMIN: list[float] = [0.0]
    ES: list[float] = [0.0]
    EA: list[float] = [0.0]
    FC: list[float] = [0.0]
    FEW: list[float] = [0.0]
    FW: list[float] = [0.0]
    ALBEDO: list[float] = [0.0]
    RNS: list[float] = [0.0]
    PIE: list[float] = [0.0]
    DR: list[float] = [0.0]
    LDELTA: list[float] = [0.0]
    WS: list[float] = [0.0]
    RA1: list[float] = [0.0]
    RA2: list[float] = [0.0]
    RA: list[float] = [0.0]
    RSO: list[float] = [0.0]
    RATIO: list[float] = [0.0]
    FCD: list[float] = [0.0]
    TK4: list[float] = [0.0]
    RNL: list[float] = [0.0]
    RN: list[float] = [0.0]
    G: list[float] = [0.0]
    WINDSP: list[float] = [0.0]
    WIND2m: list[float] = [0.0]
    Cn: list[float] = [0.0]
    Cd: list[float] = [0.0]
    KCMAX: list[float] = [0.0]
    RHMIN: list[float] = [0.0]
    WND: list[float] = [0.0]
    CHT: list[float] = [0.0]
    REFET: list[float] = [0.0]
    SKC: list[float] = [0.0]
    KCBMIN: list[float] = [0.0]
    KCBMAX: list[float] = [0.0]
    KCB: list[float] = [0.0]
    KE: list[float] = [0.0]
    KC: list[float] = [0.0]

    TAVG = (TMAX + TMIN) / 2.0

    PATM = 101.3 * ((293.0 - 0.0065 * XELEV) / 293.0) ** 5.26

    PSYCON = 0.000665 * PATM

    UDELTA = 2503.0 * math.exp(17.27 * TAVG / (TAVG + 237.3)) / (TAVG + 237.3) ** 2.0

    EMAX = 0.6108 * math.exp((17.27 * TMAX) / (TMAX + 237.3))
    EMIN = 0.6108 * math.exp((17.27 * TMIN) / (TMIN + 237.3))
    ES = (EMAX + EMIN) / 2.0

    EA = 0.6108 * math.exp((17.27 * TDEW) / (TDEW + 237.3))

    RHMIN = max(20.0, math.min(80.0, EA / EMAX * 100.0))

    if XHLAI <= 0.0:
        ALBEDO = MSALB
    else:
        ALBEDO = 0.23

    RNS = (1.0 - ALBEDO) * SRAD

    PIE = 3.14159265359
    DR = 1.0 + 0.033 * math.cos(2.0 * PIE / 365.0 * DOY)
    LDELTA = 0.409 * math.sin(2.0 * PIE / 365.0 * DOY - 1.39)
    WS = math.acos(-1.0 * math.tan(XLAT * PIE / 180.0) * math.tan(LDELTA))
    RA1 = WS * math.sin(XLAT * PIE / 180.0) * math.sin(LDELTA)
    RA2 = math.cos(XLAT * PIE / 180.0) * math.cos(LDELTA) * math.sin(WS)
    RA = 24.0 / PIE * 4.92 * DR * (RA1 + RA2)

    RSO = (0.75+2E-5*XELEV)*RA

    RATIO = SRAD/RSO
    if (RATIO < 0.3):
        RATIO = 0.3
    elif (RATIO > 1.0):
        RATIO = 1.0
    FCD = 1.35*RATIO-0.35
    TK4 = ((TMAX+273.16)**4.0+(TMIN+273.16)**4.0)/2.0
    RNL = 4.901E-9*FCD*(0.34-0.14*math.sqrt(EA))*TK4

    RN = RNS - RNL

    G = 0.0

    WINDSP = WINDRUN * 1000.0 / 24.0 / 60.0 / 60.0
    WIND2m = WINDSP * (4.87/math.log(67.8*WINDHT-5.42))

    if MEEVP == "A":
        Cn = 1600.0
        Cd = 0.38
    elif MEEVP == "G":
        Cn = 900.0
        Cd = 0.34

    REFET = 0.408*UDELTA*(RN-G)+PSYCON*(Cn/(TAVG+273.0))*WIND2m*(ES-EA)
    REFET = REFET/(UDELTA+PSYCON*(1.0+Cd*WIND2m))
    REFET = max(0.0001, REFET)

    # need to figure out what to do about these GET Call
    # They seem to represent loading a value from a global variable defined
    # at some other point in program execution
    CALL GET('SPAM', 'SKC', SKC)
    CALL GET('SPAM', 'KCBMIN', KCBMIN)
    CALL GET('SPAM', 'KCBMAX', KCBMAX)
    if (XHLAI <= 0.0):
        KCB = 0.0
    else:
        KCB = max(0.0, KCBMIN+(KCBMAX-KCBMIN)*(1.0-math.exp(-1.0*SKC*XHLAI)))

    WND = max(1.0, min(WIND2m, 6.0))
    CHT = max(0.001, CANHT)
    if MEEVP == "A":
        KCMAX = max(1.0, KCB+0.05)
    elif MEEVP == "G":
        KCMAX = max((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))*(CHT / 3.0)**(0.3)), KCB+0.05)

    if KCB <= KCBMIN:
        FC = 0.0
    else:
        FC = ((KCB-KCBMIN)/(KCMAX-KCBMIN))**(1.0+0.5*CANHT)

    FW = 1.0
    FEW = min(1.0-FC, FW)

    KE = max(0.0, min(1.0*(KCMAX-KCB), FEW*KCMAX))

    KC = KCB + KE

    EO = (KCB + KE) * REFET

    EO = max(EO, 0.0001)

    # Need to figure out how to handle these. Seems akin to setting global variables
    # CALL PUT('SPAM', 'REFET', REFET)
    # CALL PUT('SPAM', 'KCB', KCB)
    # CALL PUT('SPAM', 'KE', KE)
    # CALL PUT('SPAM', 'KC', KC)

    return EO
