import numpy as np
from math import *


def PETPT(**kw):

    tmax = kw.get('tmax')
    tmin = kw.get('tmin')
    srad = kw.get('srad')
    xhlai = kw.get('xhlai')
    msalb = kw.get('msalb')

    # print(tmax)
    # return tmax

    td = 0.6*tmax + 0.4*tmin

    if xhlai < 0.0:
        albedo = msalb
    else:
        albedo = 0.23 - (0.23 - msalb)*np.exp(-0.75*xhlai)
        

    slang = srad*23.923
    eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
    eo = eeq*1.1

    if tmax > 35.0:
        eo = eeq*((tmax-35.0)*0.05 + 1.1)
    elif tmax < 5.0:
        eo = eeq*0.01*np.exp(0.18*(tmax+20.0))
    eo = max(eo, 0.0001)

    
    return eo*pow(10,4)




def PETASCE(**kw):

    tmax = kw.get('tmax')
    tmin = kw.get('tmin')
    srad = kw.get('srad')
    xhlai = kw.get('xhlai')
    msalb = kw.get('msalb')
    canht = kw.get('canht')
    doy = kw.get('doy')
    meevp =  kw.get('meevp')
    tdew =  kw.get('tdew')
    windht = kw.get('windht')
    windrun = kw.get('windrun')
    xlat = kw.get('xlat')
    xelev = kw.get('xelev')



    tavg = (tmax + tmin)/2.0

    patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26
    #print(patm)

    psycon = 0.00066*patm
    #print(psycon)


    udelta = 2503.0*np.exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0
    #print(udelta)

    emax = 0.6108*np.exp((17.27*tmax)/(tmax+237.3))
    emin = 0.6108*np.exp((17.27*tmin)/(tmin+237.3))
    es = (emax + emin) / 2.0

    ea = 0.6108*np.exp((17.27*tdew)/(tdew+237.3))

    rhmin = max(20.0, min(80.0, ea/emax*100.0))


    if xhlai < 0:
        albedo = msalb
    else:
        albedo = 0.23

    rns = (1.0-albedo)*srad
    #print(albedo)

    pie = 4*atan(1)
    dr = 1.0+0.033*np.cos(2.0*pi/365.0*doy)
    ldelta = 0.409*np.sin(2.0*pi/365.0*doy-1.39)
    ws = np.arccos(-1.0*np.tan(xlat*pi/180.0)*np.tan(ldelta))
    #print("tan with xlat:", np.tan(pi))
    #print("tan with ldelta:", np.tan(ldelta))
    #print("value inside arcos :", -1.0*np.tan(xlat*pie/180.0)*np.tan(ldelta))
    ra1 = ws*np.sin(xlat*pi/180.0)*np.sin(ldelta)
    ra2 = np.cos(xlat*pi/180.0)*np.cos(ldelta)*np.sin(ws)
    ra = 24.0/pi*4.92*dr*(ra1+ra2)
    
    rso = (0.75+2*pow(10,-5)*xelev)*ra

    ratio = srad/rso

    if ratio < 0.3:
        ratio = 0.3
    elif ratio > 1.0:
        ratio = 1.0

    fcd = 1.35*ratio-0.35
    #print(fcd)
    tk4 = ((tmax+273.16)**4.0+(tmin+273.16)**4.0)/2.0
    #print(tk4)
    rnl = 4.901*pow(10,-9)*fcd*(0.34-0.14*np.sqrt(ea))*tk4
    #print(rnl)

    rn = rns - rnl
    #print(rn)

    g = 0.0

    windsp = windrun*1000.0 / 24.0 / 60.0 / 60.0
    #print("value inside log :", 67.8*windht - 5.42)
    wind2m = windsp*(4.87/np.log(67.8*windht-5.42))

    if meevp == 'A':
        cn = 1600.0
        cd = 0.38
    elif meevp == 'G':
        cn = 900.0
        cd = 0.34
        
    refet = 0.408*udelta*(rn-g)+psycon*(cn/(tavg+273.0))*wind2m*(es-ea)
    #print(rn)
    refet = refet/(udelta+psycon*(1.0+cd*wind2m))

    kcbmax = 1.2
    kcbmin = 0.3
    skc = 0.8

    refet = max(0.0001, refet)

    if xhlai < 0:
        kcb = 0.0
    else:
        kcb = max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-np.exp(-1.0*skc*xhlai)))

    wnd = max(1.0, min(wind2m,6.0))
    cht = max(0.001, canht)

    if meevp =='A':
        kcmax = max(1.0, kcb+0.05)
    elif meevp == 'G':
        kcmax = max((1.2+(0.04*(wnd-2.0)-0.004*(rhmin-45.0))
                        *(cht/3.0)**(0.3)),kcb+0.05)

    if kcb < kcbmin:
        fc = 0.0
    else:
        fc = ((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+0.5*canht)

    fw = 1.0
    few = min(1.0-fc,fw)
    ke = max(0.0, min(1.0*(kcmax-kcb), few*kcmax))


    kc = kcb + ke
    eo = (kcb + ke)*refet


    eo = max(eo, 0.0001)


    return eo*pow(10,4)

# val = {'tmax': 16.0, 'tmin': 10.0, 'srad': 10.0, 'xhlai': 2.45, 'msalb': 0.18}

# y = PETPT(**val)
# print(y)
