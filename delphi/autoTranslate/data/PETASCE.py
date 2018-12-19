from typing import List
import math


def PETASCE(CANHT: List[float], DOY: List[int], MSALB: List[float],
            MEEVP: List[str], SRAD: List[float], TDEW: List[float],
            TMAX: List[float], TMIN: List[float], WINDHT: List[float],
            WINDRUN: List[float], XHLAI: List[float], XLAT: List[float],
            XELEV: List[float]):

    # DOY: List[int] = [0]

    # MEEVP: List[str] = [""]

    # CANHT: List[float] = [0.0]
    # MSALB: List[float] = [0.0]
    # SRAD: List[float] = [0.0]
    # TDEW: List[float] = [0.0]
    # TMAX: List[float] = [0.0]
    # TMIN: List[float] = [0.0]
    # WINDHT: List[float] = [0.0]
    # WINDRUN: List[float] = [0.0]
    # XHLAI: List[float] = [0.0]
    # XLAT: List[float] = [0.0]
    # XELEV: List[float] = [0.0]
    # EO: List[float] = [0.0]
    TAVG: List[float] = [0.0]
    PATM: List[float] = [0.0]
    PSYCON: List[float] = [0.0]
    UDELTA: List[float] = [0.0]
    EMAX: List[float] = [0.0]
    EMIN: List[float] = [0.0]
    ES: List[float] = [0.0]
    EA: List[float] = [0.0]
    FC: List[float] = [0.0]
    FEW: List[float] = [0.0]
    FW: List[float] = [0.0]
    ALBEDO: List[float] = [0.0]
    RNS: List[float] = [0.0]
    PIE: List[float] = [0.0]
    DR: List[float] = [0.0]
    LDELTA: List[float] = [0.0]
    WS: List[float] = [0.0]
    RA1: List[float] = [0.0]
    RA2: List[float] = [0.0]
    RA: List[float] = [0.0]
    RSO: List[float] = [0.0]
    RATIO: List[float] = [0.0]
    FCD: List[float] = [0.0]
    TK4: List[float] = [0.0]
    RNL: List[float] = [0.0]
    RN: List[float] = [0.0]
    G: List[float] = [0.0]
    WINDSP: List[float] = [0.0]
    WIND2m: List[float] = [0.0]
    Cn: List[float] = [0.0]
    Cd: List[float] = [0.0]
    KCMAX: List[float] = [0.0]
    RHMIN: List[float] = [0.0]
    WND: List[float] = [0.0]
    CHT: List[float] = [0.0]
    REFET: List[float] = [0.0]
    SKC: List[float] = [0.0]
    KCBMIN: List[float] = [0.0]
    KCBMAX: List[float] = [0.0]
    KCB: List[float] = [0.0]
    KE: List[float] = [0.0]
    KC: List[float] = [0.0]

    TAVG[0] = (TMAX[0] + TMIN[0]) / 2.0

    PATM[0] = 101.3 * ((293.0 - 0.0065 * XELEV[0]) / 293.0) ** 5.26

    PSYCON[0] = 0.000665 * PATM[0]

    UDELTA[0] = 2503.0 * math.exp(17.27 * TAVG[0] / (TAVG[0] + 237.3)) / (TAVG[0] + 237.3) ** 2.0

    EMAX[0] = 0.6108 * math.exp((17.27 * TMAX[0]) / (TMAX[0] + 237.3))
    EMIN[0] = 0.6108 * math.exp((17.27 * TMIN[0]) / (TMIN[0] + 237.3))
    ES[0] = (EMAX[0] + EMIN[0]) / 2.0

    EA[0] = 0.6108 * math.exp((17.27 * TDEW[0]) / (TDEW[0] + 237.3))

    RHMIN[0] = max(20.0, min(80.0, EA[0] / EMAX[0] * 100.0))

    if XHLAI[0] <= 0.0:
        ALBEDO[0] = MSALB[0]
    else:
        ALBEDO[0] = 0.23

    RNS[0] = (1.0 - ALBEDO[0]) * SRAD[0]

    PIE = 3.14159265359
    DR[0] = 1.0 + 0.033 * math.cos(2.0 * PIE / 365.0 * DOY[0])
    LDELTA[0] = 0.409 * math.sin(2.0 * PIE / 365.0 * DOY[0] - 1.39)
    WS[0] = math.acos(-1.0 * math.tan(XLAT[0] * PIE / 180.0) * math.tan(LDELTA[0]))
    RA1[0] = WS[0] * math.sin(XLAT[0] * PIE / 180.0) * math.sin(LDELTA[0])
    RA2[0] = math.cos(XLAT[0] * PIE / 180.0) * math.cos(LDELTA[0]) * math.sin(WS[0])
    RA[0] = 24.0 / PIE * 4.92 * DR[0] * (RA1[0] + RA2[0])

    RSO[0] = (0.75+2E-5*XELEV[0])*RA[0]

    RATIO[0] = SRAD[0]/RSO[0]
    if (RATIO[0] < 0.3):
        RATIO[0] = 0.3
    elif (RATIO[0] > 1.0):
        RATIO[0] = 1.0
    FCD[0] = 1.35*RATIO[0]-0.35
    TK4[0] = ((TMAX[0]+273.16)**4.0+(TMIN[0]+273.16)**4.0)/2.0
    RNL[0] = 4.901E-9*FCD[0]*(0.34-0.14*math.sqrt(EA[0]))*TK4[0]

    RN[0] = RNS[0] - RNL[0]

    G[0] = 0.0

    WINDSP[0] = WINDRUN[0] * 1000.0 / 24.0 / 60.0 / 60.0
    WIND2m[0] = WINDSP[0] * (4.87/math.log(67.8*WINDHT[0]-5.42))

    if MEEVP[0] == "A":
        Cn[0] = 1600.0
        Cd[0] = 0.38
    elif MEEVP[0] == "G":
        Cn[0] = 900.0
        Cd[0] = 0.34

    REFET[0] = 0.408*UDELTA[0]*(RN[0]-G[0])+PSYCON[0]*(Cn[0]/(TAVG[0]+273.0))*WIND2m[0]*(ES[0]-EA[0])
    REFET[0] = REFET[0]/(UDELTA[0]+PSYCON[0]*(1.0+Cd[0]*WIND2m[0]))
    REFET[0] = max(0.0001, REFET[0])

    # need to figure out what to do about these GET Call
    # They seem to represent loading a value from a global variable defined
    # at some other point in program execution
    # CALL GET('SPAM', 'SKC', SKC)
    # CALL GET('SPAM', 'KCBMIN', KCBMIN)
    # CALL GET('SPAM', 'KCBMAX', KCBMAX)

    SKC: List[float] = [1.0]
    KCBMIN: List[float] = [1.0]
    KCBMAX: List[float] = [1.0]

    if (XHLAI[0] <= 0.0):
        KCB[0] = 0.0
    else:
        KCB[0] = max(0.0, KCBMIN[0]+(KCBMAX[0]-KCBMIN[0])*(1.0-math.exp(-1.0*SKC[0]*XHLAI[0])))

    WND[0] = max(1.0, min(WIND2m[0], 6.0))
    CHT[0] = max(0.001, CANHT[0])
    if MEEVP[0] == "A":
        KCMAX[0] = max(1.0, KCB[0]+0.05)
    elif MEEVP[0] == "G":
        KCMAX[0] = max((1.2+(0.04*(WND[0]-2.0)-0.004*(RHMIN[0]-45.0))*(CHT[0] / 3.0)**(0.3)), KCB[0]+0.05)

    if KCB[0] <= KCBMIN[0]:
        FC[0] = 0.0
    else:
        FC[0] = ((KCB[0]-KCBMIN[0])/(KCMAX[0]-KCBMIN[0]))**(1.0+0.5*CANHT[0])

    FW[0] = 1.0
    FEW[0] = min(1.0-FC[0], FW[0])

    KE[0] = max(0.0, min(1.0*(KCMAX[0]-KCB[0]), FEW[0]*KCMAX[0]))

    KC[0] = KCB[0] + KE[0]

    EO: List[float] = [0.0]
    EO[0] = (KCB[0] + KE[0]) * REFET[0]

    EO[0] = max(EO[0], 0.0001)

    # Need to figure out how to handle these. Seems akin to setting global variables
    # CALL PUT('SPAM', 'REFET', REFET)
    # CALL PUT('SPAM', 'KCB', KCB)
    # CALL PUT('SPAM', 'KE', KE)
    # CALL PUT('SPAM', 'KC', KC)

    return EO[0]
