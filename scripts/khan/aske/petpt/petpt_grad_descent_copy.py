import numpy as np
import random
#from math import *
from sympy import *


msalb = Symbol('msalb')
srad = Symbol('srad')
tmax = Symbol('tmax')
tmin = Symbol('tmin')
xhlai = Symbol('xhlai')
canht = Symbol('canht')
doy = Symbol('doy')
tdew = Symbol('tdew')
windht = Symbol('windht')
windrun = Symbol('windrun')
xlat = Symbol('xlat')
xelev = Symbol('xelev')


###################### PETPT Model  #################################

td = 0.6*tmax + 0.4*tmin

#if xhlai < 0.0:
albedo_0 = msalb
#else:
albedo_1 = 0.23 - (0.23 - msalb)*exp(-0.75*xhlai)

albedo = [albedo_0, albedo_1]

slang = srad*23.923

eeq = []
eo_1 = []
for i in range(len(albedo)):
    eeq.append(slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo[i])*(td+29.0))
    eo_1.append(eeq[i]*1.1)
    #if tmax > 35.0:
    eo_1.append(eeq[i]*((tmax-35.0)*0.05 + 1.1))
    #elif tmax < 5.0:
    eo_1.append(eeq[i]*0.01*exp(0.18*(tmax+20.0)))


#####################################################


##################### PETASCE Model #############################

tavg = (tmax + tmin)/2.0

patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26


psycon = 0.00066*patm


udelta = 2503.0*exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0


emax = 0.6108*exp((17.27*tmax)/(tmax+237.3))
emin = 0.6108*exp((17.27*tmin)/(tmin+237.3))
es = (emax + emin) / 2.0

ea = 0.6108*exp((17.27*tdew)/(tdew+237.3))

rhmin = Max(20.0, Min(80.0, ea/emax*100.0))



#if xhlai < 0:
albedo_0 = msalb
#else:
albedo_1 = 0.23

albedo = [albedo_0, albedo_1]

rns = []
for i in range(len(albedo)):
    rns.append((1.0-albedo[i])*srad)

dr = 1.0+0.033*cos(2.0*pi/365.0*doy)
ldelta = 0.409*sin(2.0*pi/365.0*doy-1.39)
ws = acos(-1.0*tan(xlat*pi/180.0)*tan(ldelta))
ra1 = ws*sin(xlat*pi/180.0)*sin(ldelta)
ra2 = cos(xlat*pi/180.0)*cos(ldelta)*sin(ws)
ra = 24.0/pi*4.92*dr*(ra1+ra2)

rso = (0.75+2*pow(10,-5)*xelev)*ra

ratio_0 = srad/rso

#if ratio < 0.3:
ratio_1 = 0.3
#elif ratio > 1.0:
ratio_2 = 1.0

ratio = [ratio_0, ratio_1, ratio_2]

fcd = []
rnl = []
tk4 = ((tmax+273.16)**4.0+(tmin+273.16)**4.0)/2.0
for i in range(len(ratio)):
    fcd.append(1.35*ratio[i]-0.35)
    rnl.append(4.901*pow(10,-9)*fcd[i]*(0.34-0.14*sqrt(ea))*tk4)

rn = []
for i in range(len(rns)):
    for j  in range(len(rnl)):
        rn.append(rns[i] - rnl[j])

g = 0.0

windsp = windrun*1000.0 / 24.0 / 60.0 / 60.0
wind2m = windsp*(4.87/log(67.8*windht-5.42))

#if meevp == 'A':
cn_0 = 1600.0
cd_0 = 0.38
#elif meevp == 'G':
cn_1 = 900.0
cd_1 = 0.34


refet = []
for i in range(len(rn)):
    refet.append(0.408*udelta*(rn[i]-g)+psycon*(cn_0/(tavg+273.0))*wind2m*(es-ea))
    refet[2*i] = Max(0.0001,refet[i]/(udelta+psycon*(1.0+cd_0*wind2m)))
    refet.append(0.408*udelta*(rn[i]-g)+psycon*(cn_1/(tavg+273.0))*wind2m*(es-ea))
    refet[2*i+1] = Max(0.0001,refet[2*i+1]/(udelta+psycon*(1.0+cd_1*wind2m)))

kcbmax = 1.2
kcbmin = 0.3
skc = 0.8


#if xhlai < 0:
kcb_0 = 0.0
#else:
kcb_1 = Max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-exp(-1.0*skc*xhlai)))

kcb = [kcb_0, kcb_1]


wnd = Max(1.0, Min(wind2m,6.0))
cht = Max(0.001, canht)

kcmax = []
for i in range(len(kcb)):
    #if meevp =='A':
    kcmax.append(Max(1.0, kcb[i]+0.05))
    #elif meevp == 'G':
    kcmax.append(Max((1.2+(0.04*(wnd-2.0)-0.004*(rhmin-45.0))
            *(cht/3.0)**(0.3)),kcb[i]+0.05))

fc =[]

#if kcb < kcbmin:
fc.append(0.0)
#else:
for i in range(len(kcmax)):
    for j in range(len(kcb)):
        fc.append(((kcb[j]-kcbmin)/(kcmax[i]-kcbmin))**(1.0+0.5*canht))

fw = 1.0
few = []
for i in range(len(fc)):        
        few.append(Min(1.0-fc[i],fw))

ke = []
for i in range(len(few)):
    for j in range(len(kcmax)):
        for k in range(len(kcb)):
            ke.append(Max(0.0, Min(1.0*(kcmax[j]-kcb[k]), few[i]*kcmax[j])))

eo_2 = []
for i in range(len(kcb)):
    for j in range(len(ke)):
        for k in range(len(refet)):
            eo_2.append((kcb[i] + ke[j])*refet[k])
            #eo_2.append(Max(0.0001,(kcb[i] + ke[j])*refet[k]))


################################################################

Loss = [(x - y)**2 for x in eo_2 for y in eo_1]
#print(len(Loss))
#print(len(eo_1))
#print(len(eo_2))

Diff_Matrix = Matrix(Loss).jacobian([msalb, srad, tmax, tmin, xhlai])
#print(Diff_Matrix[0,0])
#print(Diff_Matrix.shape[0])
#print(Diff_Matrix.shape[1])





alpha = 0.1
iterations = 0
check = 0
precision = 0.01
printData = True
maxIterations = 1000

size = 2

msalb_val = np.linspace(0, 1, size)
srad_val = np.linspace(1, 35, size)
tmax_val = np.linspace(-30, 60, size)
tmin_val = np.linspace(-30, 60, size)
#tmin_val = np.zeros(len(tmax_val))
#for i in range(len(tmax_val)):
#    tmin_val[i] = random.uniform(-30, tmax_val[i])
xhlai_val = np.linspace(1, 20, size)

tdew = random.uniform(-30, 60)
windht = random.uniform(0.1, 10)
windrun = random.uniform(0, 900)
xlat = random.uniform(-90, 90)
xelev = random.uniform(0, 6000)
doy = random.uniform(1, 365)
canht = random.uniform(0, 3)

for i in range(Diff_Matrix.shape[0]):
    for theta in msalb_val:
        for theta1 in srad_val:
            for theta2 in tmax_val:
                for theta3 in tmin_val:
                    for theta4 in xhlai_val:
                        while True:
                            temptheta = theta - alpha*(Diff_Matrix[i,0].subs(msalb, theta)).subs(srad, theta1).subs(tmax, theta2).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                            temptheta1 = theta1 - alpha*(Diff_Matrix[i,1].subs(srad, theta1)).subs(msalb, theta).subs(tmax, theta2).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                            temptheta2 = theta2 - alpha*(Diff_Matrix[i,2].subs(tmax, theta2)).subs(msalb, theta).subs(srad, theta1).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                            temptheta3 = theta3 - alpha*(Diff_Matrix[i,3].subs(tmin, theta3)).subs(msalb, theta).subs(srad, theta1).subs(tmax, theta2).subs(xhlai, theta4).evalf()
                            temptheta4 = theta4 - alpha*(Diff_Matrix[i,4].subs(xhlai, theta4)).subs(msalb, theta).subs(srad, theta1).subs(tmax, theta2).subs(tmax, theta3).evalf()
#                            print(temptheta); print(temptheta1);
#                            print(temptheta2); print(temptheta3);
#                            print(temptheta4)

                            iterations += 1
                            if iterations > maxIterations:
                                print("Adjust alpha values")
                                printData = False
                                break

                            if abs(temptheta - theta) < precision and abs(temptheta1 - theta1) < precision and abs(temptheta2 - theta2) < precision and abs(temptheta3 - theta3) <           precision and abs(temptheta4 - theta4) < precision:
                                break

                            theta = temptheta;
                            theta1 = temptheta1;
                            theta2 = temptheta2;
                            theta3 = temptheta3;
                            theta4 = temptheta4;

    iterations = 0;
    if printData:
        print("The function "+str(eo[i])+"converges to a minimum")
        print("Number of iterations:",iterations, sep='')
        print("theta (msalb) =",theta,sep=" ")
        print("theta1 (srad) =",theta1,sep=" ")
        print("theta2 (tmax) =",theta2,sep=" ")
        print("theta3 (tmin) =",theta3,sep=" ")
        print("theta4 (xhlai) =",theta4,sep=" ")
