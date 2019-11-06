import cvxpy as cp
import numpy as np
from math import *

def PETPT(msalb, srad, tmax, tmin, xhlai):

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

    #print(msalb, srad, tmax, tmin, xhlai)
    return eo



def PETASCE(msalb, canht, doy, meevp, tdew, windht, windrun, xlat, xelev, srad, tmax, tmin, xhlai):  
 
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

    #print(msalb, canht, doy, meevp, tdew, windht, windrun, xlat, xelev, srad, tmax, tmin, xhlai)

    return eo

size = 20

#Parameters for PETPT and PETASSCE
tmax = np.linspace(16.1, 36.7, size)  #UFGA7801.WTH and ET.OUT
tmin = np.linspace(0.0, 23.9, size)   #UFGA7801.WTH and ET.OUT
#msalb = np.linspace(0, 1, size)
srad = np.linspace(2.45, 27.8, size)   #UFGA7801.WTH and ET.OUT
xhlai = np.linspace(0.00, 4.77, size)  #PlantGro.OUT file

#Preset Values of Variables in PETPT and PETASCE
msalb = 0.18 # SOIL.SOL file

#Parameters for only PETASCE
#tdew = 16.0
tdew = np.linspace(0.0, 36.7, size)  # Taken from tmax and tmin

windrun = 400
#windrun = np.linspace(0, 900, size) 

doy = 1
#doy = np.linspace(1, 365, size)

canht = 1.0
#canht = np.linspace(0, 3, size)

#Preset Values of Variables in PETASCE
meevp = 'G'    #Soybean 
windht = 3.00                   #PlantGro.OUT
xlat = 26.63                    #PlantGro.OUT
xelev = 10                      #PlantGro.OUT

vfunc1 = np.vectorize(PETPT)
y1 = vfunc1(msalb, srad, tmax, tmin, xhlai)
x1 = np.arange(y1.size)
#print(len(x1))
#print(len(y1))
#print(y1)

vfunc2 = np.vectorize(PETASCE)
y2 = vfunc2(msalb, canht, doy, meevp, tdew, windht, windrun, xlat, xelev, srad, tmax, tmin, xhlai)
x2 = np.arange(y2.size)
#print(len(x2))
#print(len(y2))
#print(y2)


# Construct the problem.
x = cp.Variable(srad)
objective = cp.Minimize((y1 - y2)**2)
constraints = [0 <= y1, y2 <= 10, 0 <= y2, y2 <= 10]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)


#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(x1+1, y1, label = 'PETPT', c = 'r')
#ax1.scatter(x2+1, y2, label = 'PETASCE', c = 'b')
#plt.xlabel('Parameter set')
#plt.ylabel(r'EO ($\times 10^4$)')
#plt.legend()
#plt.show()




