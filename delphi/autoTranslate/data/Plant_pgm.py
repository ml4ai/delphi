from typing import List
import math


def MAIN(DOY: List[int], TMAX: List[float], TMIN: List[float], SWFAC1: List[float],
         SWFAC2: List[float], PD: List[float], EMP1: List[float], EMP2: List[float],
         PT: List[float], sla: List[float], di: List[float], N: List[float],
         nb: List[float], dN: List[float], p1: List[float]):
    E: List[float] = [0.0]
    Fc: List[float] = [0.0]
    Lai: List[float] = [0.0]
    # nb: List[float] = [0.0]
    # N: List[float] = [0.0]
    # PT: List[float] = [0.0]
    Pg: List[float] = [0.0]
    # di: List[float] = [0.0]
    PAR: List[float] = [0.0]
    rm: List[float] = [0.0]
    dwf: List[float] = [0.0]
    ints: List[float] = [0.0]
    # TMAX: List[float] = [0.0]
    # TMIN: List[float] = [0.0]
    # p1: List[float] = [0.0]
    # sla: List[float] = [0.0]
    # PD: List[float] = [0.0]
    # EMP1: List[float] = [0.0]
    # EMP2: List[float] = [0.0]
    Lfmax: List[float] = [0.0]
    dwc: List[float] = [0.0]
    TMN: List[float] = [0.0]
    dwr: List[float] = [0.0]
    dw: List[float] = [0.0]
    # dN: List[float] = [0.0]
    w: List[float] = [0.0]
    wc: List[float] = [0.0]
    wr: List[float] = [0.0]
    wf: List[float] = [0.0]
    tb: List[float] = [0.0]
    intot: List[float] = [0.0]
    dLAI: List[float] = [0.0]
    FL: List[float] = [0.0]
    # DOY: List[int] = [0]
    endsim: List[int] = [0]
    COUNT: List[int] = [0]
    DYN: List[str] = ['']
    # SWFAC1: List[float] = [0.0]
    # SWFAC2: List[float] = [0.0]
    # DOY[0] = 234
    endsim[0] = 0
    # TMAX[0] = 33.9000015
    # TMIN[0] = 22.7999992
    PAR[0] = 11.2500000
    # SWFAC1[0] = 1.00000000
    # SWFAC2[0] = 1.00000000
    # PD[0] = 5.00000000
    # EMP1[0] = 0.104000002
    # EMP2[0] = 0.639999986
    # PT[0] = 0.974399984
    # sla[0] = 2.80000009E-02
    # di[0] = 0.00000000
    # N[0] = 12.0639181
    # nb[0] = 5.30000019
    # dN[0] = 0.00000000
    # p1[0] = 2.99999993E-02

    TMN[0] = (0.5 * (TMAX[0] + TMIN[0]))
    PTS(TMAX, TMIN, PT)
    PGS(SWFAC1, SWFAC2, PAR, PD, PT, Lai, Pg)
    if (N[0] < Lfmax[0]):
        FL[0] = 1.0
        E[0] = 1.0
        dN[0] = (rm[0] * PT[0])
        LAIS(FL, di, PD, EMP1, EMP2, N, nb, SWFAC1, SWFAC2, PT, dN, p1, sla, dLAI)
        dw[0] = ((E[0] * Pg[0]) * PD[0])
        dwc[0] = (Fc[0] * dw[0])
        dwr[0] = ((1 - Fc[0]) * dw[0])
        dwf[0] = 0.0
    else:
        FL[0] = 2.00000000
        if ((TMN[0] >= tb[0]) and (TMN[0] <= 25)):
            di[0] = (TMN[0] - tb[0])
        else:
            di[0] = 0.0
        ints[0] = (ints[0] + di[0])
        E[0] = 1.0
        LAIS(FL, di, PD, EMP1, EMP2, N, nb, SWFAC1, SWFAC2, PT, dN, p1, sla, dLAI)
        dw[0] = ((E[0] * Pg[0]) * PD[0])
        dwf[0] = dw[0]
        dwc[0] = 0.0
        dwr[0] = 0.0
        dN[0] = 0.0
    return dLAI[0]


def LAIS(FL: List[float], di: List[float], PD: List[float], EMP1: List[float], EMP2: List[float], N: List[float], nb: List[float], SWFAC1: List[float], SWFAC2: List[float], PT: List[float], dN: List[float], p1: List[float], sla: List[float], dLAI: List[float]):
    SWFAC: List[float] = [0.0]
    a: List[float] = [0.0]
    SWFAC[0] = min(SWFAC1[0], SWFAC2[0])
    if (FL[0] == 1.0):
        a[0] = math.exp((EMP2[0] * (N[0] - nb[0])))
        dLAI[0] = (((((SWFAC[0] * PD[0]) * EMP1[0]) * PT[0]) * (a[0] / (1 + a[0]))) * dN[0])
    else:
        if (FL[0] == 2.0):
            dLAI[0] = -((((PD[0] * di[0]) * p1[0]) * sla[0]))
    return True


def PGS(SWFAC1: List[float], SWFAC2: List[float], PAR: List[float], PD: List[float], PT: List[float], Lai: List[float], Pg: List[float]):
    Y1: List[float] = [0.0]
    SWFAC: List[float] = [0.0]
    ROWSPC: List[float] = [0.0]
    SWFAC[0] = min(SWFAC1[0], SWFAC2[0])
    ROWSPC[0] = 60.0
    Y1[0] = (1.5 - (0.768 * ((((ROWSPC[0] * 0.01) ** 2) * PD[0]) ** 0.1)))
    Pg[0] = (((((PT[0] * SWFAC[0]) * 2.1) * PAR[0]) / PD[0]) * (1.0 - math.exp(-((Y1[0] * Lai[0])))))
    return True

def PTS(TMAX: List[float], TMIN: List[float], PT: List[float]):
    PT[0] = (1.0 - (0.0025 * ((((0.25 * TMIN[0]) + (0.75 * TMAX[0])) - 26.0) ** 2)))
    return True

# MAIN()
