import numpy as np
from math import *
from itertools import product
import matplotlib.pyplot as plt


def PETPT(msalb, srad, tmax, tmin, xhlai):

    td = 0.6*tmax + 0.4*tmin

    if xhlai < 0.0:
        albedo = msalb
    else:
        albedo = 0.23 - (0.23 -
                msalb)*np.exp(-0.75*xhlai)

    slang = srad*23.923
    eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
    eo = eeq*1.1

    if tmax > 35.0:
        eo = eeq*((tmax-35.0)*0.05 + 1.1)
    elif tmax < 5.0:
        eo = eeq*0.01*np.exp(0.18*(tmax+20.0))

    eo = max(eo, 0.0001)

    return eo*pow(10,4)

#def PETASCE(canht, doy, msalb, meevp, srad,
#        tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev):
def PETASCE(msalb, srad, tmax, tmin,  xhlai, 
       canht, doy, meevp, tdew, windht, windrun, xlat, xelev):

    tavg = (tmax + tmin)/2.0

    patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26


    psycon = 0.00066*patm


    udelta = 2503.0*np.exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0


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

    #pie = 4*atan(1)
    dr = 1.0+0.033*np.cos(2.0*pi/365.0*doy)
    ldelta = 0.409*np.sin(2.0*pi/365.0*doy-1.39)
    #ws = pi/4
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
    tk4 = ((tmax+273.16)**4.0+(tmin+273.16)**4.0)/2.0
    rnl = 4.901*pow(10,-9)*fcd*(0.34-0.14*np.sqrt(ea))*tk4

    rn = rns - rnl

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

size = 2

#Parameters for PETPT and PETASSCE
tmax = [-30 + x*60/size for x in range(size)]

tmin = [-30 + x*60/size for x in range(size)]

msalb = [0 + x*1/size for x in range(size)]

srad = [1 + x*35/size for x in range(size)]

xhlai = [0 + x*20/size for x in range(size)]

#Parameters for only PETASCE
tdew = [-30]*size
#tdew = [-30 + x*60/size for x in range(size)]

windht = [0.1]*size
windht = [0.1 + x*10/size for x in range(size)]

windrun = []
windrun = [0 + x*900/size for x in range(size)]

xlat = [-90 + x*90/size for x in range(size)]

xelev = [0 + x*6000/size for x in range(size)]

doy = [1 + x*365/size for x in range(size)]

canht = [0 + x*3/size for x in range(size)]

#Preset Values of Variables in PETASCE
meevp = 'G'

#print(PETPT(msalb, srad, tmax, tmin, xhlai))
#print(PETASCE(canht, doy, msalb, meevp, srad,
#            tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev))

#y1 = PETPT(msalb, srad, tmax, tmin, xhlai)

y1 = [PETPT(*args) for args in product(msalb, srad, tmax, tmin, xhlai)]
#y2 = [PETASCE(*args) for args in product(canht, doy, msalb, meevp, srad, tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev)]
x1 = np.arange(len(y1))
#x2 = np.arange(len(y2))


for canht_val in canht:
    for doy_val in doy:
        for tdew_val in tdew:
            for windht_val in windht:
                for windrun_val in windrun:
                    for xlat_val in xlat:
                        for xelev_val in xelev:
                            #print(canht_val, doy_val,  meevp, tdew_val, windht_val, windrun_val, xlat_val, xelev_val)
                            y2 = [PETASCE(*args, canht_val, doy_val,  meevp, tdew_val, windht_val, windrun_val, xlat_val, xelev_val) for args in product(msalb,
                                srad, tmax, tmin, xhlai)]

                            break
                            plt.figure()
                            plt.scatter(x1+1, y1, label = 'PETPT', c = 'r')
                            plt.scatter(x1+1, y2, label = 'PETASCE', c = 'b')
                            plt.legend()
                            plt.show()

#plt.figure()
#plt.scatter(x1+1, y1, label = 'PETPT', c = 'r')
#plt.scatter(x2+1, y2, label = 'PETASCE', c = 'b')
#plt.legend()
#plt.show()

precision = 1
i = 0
#print(len(y1), len(y2), len(y1)*len(y2))
for y1_val in y1:
    for y2_val in y2:
        i +=1
        if abs(y1_val - y2_val) < precision:
            break
            #print(i, y1_val, y2_val)

