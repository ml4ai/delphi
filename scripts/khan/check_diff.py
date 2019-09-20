import numpy as np
from math import *
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def PETPT(msalb, srad, tmax, tmin, xhlai):

    td = 0.6*tmax + 0.4*tmin

    albedo = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        if xhlai[i] < 0.0:
            albedo[i] = msalb[i]
        else:
            albedo[i] = 0.23 - (0.23 - msalb[i])*np.exp(-0.75*xhlai[i])

    slang = srad*23.923
    eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
    eo = eeq*1.1

    for i in range(len(tmax)):
        if tmax[i] > 35.0:
            eo[i] = eeq[i]*((tmax[i]-35.0)*0.05 + 1.1)
        elif tmax[i] < 5.0:
            eo[i] = eeq[i]*0.01*np.exp(0.18*(tmax[i]+20.0))
        eo[i] = max(eo[i], 0.0001)

    return eo*pow(10,4)

def PETASCE(canht, doy, msalb, meevp, srad,
            tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev):

    tavg = (tmax + tmin)/2.0

    patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26


    psycon = 0.00066*patm


    udelta = 2503.0*np.exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0


    emax = 0.6108*np.exp((17.27*tmax)/(tmax+237.3))
    emin = 0.6108*np.exp((17.27*tmin)/(tmin+237.3))
    es = (emax + emin) / 2.0

    ea = 0.6108*np.exp((17.27*tdew)/(tdew+237.3))

    rhmin = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        rhmin[i] = max(20.0, min(80.0, ea[i]/emax[i]*100.0))

    albedo = np.zeros(len(xhlai))

    for i in range(len(xhlai)):
        if xhlai[i] < 0:
            albedo[i] = msalb[i]
        else:
            albedo[i] = 0.23

    rns = (1.0-albedo)*srad

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

    for i in range(len(ratio)):
        if ratio[i] < 0.3:
            ratio[i] = 0.3
        elif ratio[i] > 1.0:
            ratio[i] = 1.0

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

    for i in range(len(refet)):
        refet[i] = max(0.0001, refet[i])

    kcb = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        if xhlai[i] < 0:
            kcb[i] = 0.0
        else:
            kcb[i] = max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-np.exp(-1.0*skc*xhlai[i])))

    wnd = np.zeros(len(xhlai))
    cht = np.zeros(len(xhlai))
    kcmax = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        wnd[i] = max(1.0, min(wind2m[i],6.0))
        cht[i] = max(0.001, canht[i])

        if meevp =='A':
            kcmax[i] = max(1.0, kcb[i]+0.05)
        elif meevp == 'G':
            kcmax[i] = max((1.2+(0.04*(wnd[i]-2.0)-0.004*(rhmin[i]-45.0))
                           *(cht[i]/3.0)**(0.3)),kcb[i]+0.05)

    fc = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        if kcb[i] < kcbmin:
            fc[i] = 0.0
        else:
            fc[i] = ((kcb[i]-kcbmin)/(kcmax[i]-kcbmin))**(1.0+0.5*canht[i])

    fw = 1.0
    few = np.zeros(len(xhlai))
    ke = np.zeros(len(xhlai))
    for i in range(len(xhlai)):
        few[i] = min(1.0-fc[i],fw)
        ke[i] = max(0.0, min(1.0*(kcmax[i]-kcb[i]), few[i]*kcmax[i]))


    kc = kcb + ke
    eo = (kcb + ke)*refet

    for i in range(len(xhlai)):
        eo[i] = max(eo[i], 0.0001)

    return eo*pow(10,4)

size = 1000

#Parameters for PETPT and PETASSCE
tmax = np.linspace(-30, 40, size)
tmin = np.linspace(-40, 30, size)
msalb = np.linspace(0, 1, size)
srad = np.linspace(1, 35, size)
xhlai = np.linspace(0, 20, size)

#Parameters for only PETASCE
tdew = np.linspace(-40, 40, size)
windht = np.linspace(0.1, 10, size)
windrun = np.linspace(0, 900, size)
xlat = np.linspace(-90, 90, size)
xelev = np.linspace(0, 6000, size)
doy = np.linspace(1, 365, size)
canht = np.linspace(0, 3, size)

#Preset Values of Variables in PETASCE
meevp = 'G'

#print("PETPT Output values :",PETPT(msalb, srad, tmax, tmin, xhlai))
#print("PETASCE Output values:", PETASCE(canht, doy, msalb, meevp, srad,
#            tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev))


y1 = PETPT(msalb, srad, tmax, tmin, xhlai)
y2 = PETASCE(canht, doy, msalb, meevp, srad, tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev)
x = np.arange(len(y1))

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.scatter(x+1, np.log(y1), label = 'PETPT', c = 'r')
#ax1.scatter(x+1, np.log(y2), label = 'PETASCE', c = 'b')
ax1.scatter(x+1, y1, label = 'PETPT', c = 'r')
ax1.scatter(x+1, y2, label = 'PETASCE', c = 'b')
plt.xlabel('Parameter set')
plt.ylabel(r'Log (EO ($\times 10^4$))')
plt.xticks(x+1)
#for i_x, i_y in zip(x+1, y1):
#    plt.text(i_x, i_y, '({},{})'.format(i_x, round(i_y,2)))
#for i_x, i_y in zip(x+1, y2):
#    plt.text(i_x, i_y, '({},{})'.format(i_x, round(i_y,2)))
plt.legend()
plt.show()




