import sys
from typing import List
from m_mod_evapotransp import *
from m_petasce_init import *
import math
from fortran_format import *
from for2py_arrays import *

def petasce(canht: List[float], doy: List[int], msalb: List[float], meevp: List[str], srad: List[float], tdew: List[float], tmax: List[float], tmin: List[float], windht: List[float], windrun: List[float], xhlai: List[float], xlat: List[float], xelev: List[float], eo: List[float]):
    
    tavg: List[float] = [0.0]
    
    patm: List[float] = [0.0]
    
    psycon: List[float] = [0.0]
    
    udelta: List[float] = [0.0]
    
    emax: List[float] = [0.0]
    
    emin: List[float] = [0.0]
    
    es: List[float] = [0.0]
    
    ea: List[float] = [0.0]
    
    fc: List[float] = [0.0]
    
    few: List[float] = [0.0]
    
    fw: List[float] = [0.0]
    
    albedo: List[float] = [0.0]
    
    rns: List[float] = [0.0]
    
    pie: List[float] = [0.0]
    
    dr: List[float] = [0.0]
    
    ldelta: List[float] = [0.0]
    
    ws: List[float] = [0.0]
    
    ra1: List[float] = [0.0]
    
    ra2: List[float] = [0.0]
    
    ra: List[float] = [0.0]
    
    rso: List[float] = [0.0]
    
    ratio: List[float] = [0.0]
    
    fcd: List[float] = [0.0]
    
    tk4: List[float] = [0.0]
    
    rnl: List[float] = [0.0]
    
    rn: List[float] = [0.0]
    
    g: List[float] = [0.0]
    
    windsp: List[float] = [0.0]
    
    wind2m: List[float] = [0.0]
    
    cn: List[float] = [0.0]
    
    cd: List[float] = [0.0]
    
    kcmax: List[float] = [0.0]
    
    rhmin: List[float] = [0.0]
    
    wnd: List[float] = [0.0]
    
    cht: List[float] = [0.0]
    
    skc: List[float] = [0.0]
    
    kcbmin: List[float] = [0.0]
    
    kcbmax: List[float] = [0.0]
    
    kcb: List[float] = [0.0]
    
    ke: List[float] = [0.0]
    
    kc: List[float] = [0.0]
    
    tavg[0] = ((tmax[0] + tmin[0]) / 2.0)
    patm[0] = (101.3 * (((293.0 - (0.0065 * xelev[0])) / 293.0) ** 5.26))
    psycon[0] = (0.000665 * patm[0])
    udelta[0] = ((2503.0 * math.exp([((17.27 * tavg[0]) / (tavg[0] + 237.3))])) / ((tavg[0] + 237.3) ** 2.0))
    emax[0] = (0.6108 * math.exp([((17.27 * tmax[0]) / (tmax[0] + 237.3))]))
    emin[0] = (0.6108 * math.exp([((17.27 * tmin[0]) / (tmin[0] + 237.3))]))
    es[0] = ((emax[0] + emin[0]) / 2.0)
    ea[0] = (0.6108 * math.exp([((17.27 * tdew[0]) / (tdew[0] + 237.3))]))
    rhmin[0] = max([20.0], [min([80.0], [((ea[0] / emax[0]) * 100.0)])])
    if (xhlai[0] <= 0.0):
        albedo[0] = msalb[0]
    else:
        albedo[0] = 0.23
    rns[0] = ((1.0 - albedo[0]) * srad[0])
    pie[0] = 3.14159265359
    dr[0] = (1.0 + (0.033 * math.cos([(((2.0 * pie[0]) / 365.0) * doy[0])])))
    ldelta[0] = (0.409 * math.sin([((((2.0 * pie[0]) / 365.0) * doy[0]) - 1.39)]))
    ws[0] = math.acos([-(((1.0 * math.tan([((xlat[0] * pie[0]) / 180.0)])) * math.tan([ldelta[0][0]])))])
    ra1[0] = ((ws[0] * math.sin([((xlat[0] * pie[0]) / 180.0)])) * math.sin([ldelta[0][0]]))
    ra2[0] = ((math.cos([((xlat[0] * pie[0]) / 180.0)]) * math.cos([ldelta[0][0]])) * math.sin([ws[0][0]]))
    ra[0] = ((((24.0 / pie[0]) * 4.92) * dr[0]) * (ra1[0] + ra2[0]))
    rso[0] = ((0.75 + (2E-5 * xelev[0])) * ra[0])
    ratio[0] = (srad[0] / rso[0])
    if (ratio[0] < 0.3):
        ratio[0] = 0.3
    else:
        if (ratio[0] > 1.0):
            ratio[0] = 1.0
    fcd[0] = ((1.35 * ratio[0]) - 0.35)
    tk4[0] = ((((tmax[0] + 273.16) ** 4.0) + ((tmin[0] + 273.16) ** 4.0)) / 2.0)
    rnl[0] = (((4.901E-9 * fcd[0]) * (0.34 - (0.14 * math.sqrt([ea[0][0]])))) * tk4[0])
    rn[0] = (rns[0] - rnl[0])
    g[0] = 0.0
    windsp[0] = ((((windrun[0] * 1000.0) / 24.0) / 60.0) / 60.0)
    wind2m[0] = (windsp[0] * (4.87 / math.log([((67.8 * windht[0]) - 5.42)])))
    if (meevp[0] == 'A'):
        cn[0] = 1600.0
        cd[0] = 0.38
    else:
        if (meevp[0] == 'G'):
            cn[0] = 900.0
            cd[0] = 0.34
    skc[0] = 1.0
    kcbmin[0] = 1.0
    kcbmax[0] = 1.0
    if (xhlai[0] <= 0.0):
        kcb[0] = 0.0
    else:
        kcb[0] = max([0.0], [(kcbmin[0] + ((kcbmax[0] - kcbmin[0]) * (1.0 - math.exp([-(((1.0 * skc[0]) * xhlai[0]))]))))])
    wnd[0] = max([1.0], [min([wind2m[0][0]], [6.0])])
    cht[0] = max([0.001], [canht[0][0]])
    if (meevp[0] == 'A'):
        kcmax[0] = max([1.0], [(kcb[0] + 0.05)])
    else:
        if (meevp[0] == 'G'):
            kcmax[0] = max([(1.2 + (((0.04 * (wnd[0] - 2.0)) - (0.004 * (rhmin[0] - 45.0))) * ((cht[0] / 3.0) ** 0.3)))], [(kcb[0] + 0.05)])
    if (kcb[0] <= kcbmin[0]):
        fc[0] = 0.0
    else:
        fc[0] = (((kcb[0] - kcbmin[0]) / (kcmax[0] - kcbmin[0])) ** (1.0 + (0.5 * canht[0])))
    fw[0] = 1.0
    few[0] = min([(1.0 - fc[0])], [fw[0][0]])
    ke[0] = max([0.0], [min([(1.0 * (kcmax[0] - kcb[0]))], [(few[0] * kcmax[0])])])
    kc[0] = (kcb[0] + ke[0])
    eo[0] = ev_transp([udelta[0]], [rn[0]], [g[0]], [psycon[0]], [cn[0]], [tavg[0]], [wind2m[0]], [es[0]], [ea[0]], [cd[0]], [kc[0]])
    

def main():
    
    doy: List[int] = [0]
    
    eo: List[float] = [0.0]
    
    for doy[0] in range(0, 500+1, 50):
        petasce([canht[0]], [doy[0]], [msalb[0]], [meevp[0]], [srad[0]], [tdew[0]], [tmax[0]], [tmin[0]], [windht[0]], [windrun[0]], [xhlai[0]], [xlat[0]], [xelev[0]], [eo[0]])
        write_list_stream = [eo[0]]
        output_fmt = list_output_formats(["REAL",])
        write_stream_obj = Format(output_fmt)
        write_line = write_stream_obj.write_line(write_list_stream)
        sys.stdout.write(write_line)
        update_vars()
    return

main()
