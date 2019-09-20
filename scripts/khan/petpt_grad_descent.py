import numpy as np
import random
#from math import *
from sympy import *


msalb = Symbol('msalb')
srad = Symbol('srad')
tmax = Symbol('tmax')
tmin = Symbol('tmin')
xhlai = Symbol('xhlai')




# PETPT Model

td = 0.6*tmax + 0.4*tmin

#if xhlai < 0.0:
albedo_0 = msalb
#else:
albedo_1 = 0.23 - (0.23 - msalb)*exp(-0.75*xhlai)

albedo = [albedo_0, albedo_1]

slang = srad*23.923
eeq_0 = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo_0)*(td+29.0)
eeq_1 = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo_1)*(td+29.0)
eo_0 = eeq_0*1.1
eo_1 = eeq_1*1.1

#if tmax > 35.0:
eo_2 = eeq_0*((tmax-35.0)*0.05 + 1.1)
eo_3 = eeq_1*((tmax-35.0)*0.05 + 1.1)
#elif tmax < 5.0:
eo_4 = eeq_0*0.01*exp(0.18*(tmax+20.0))
eo_5 = eeq_1*0.01*exp(0.18*(tmax+20.0))

eo = [eo_0, eo_1, eo_2, eo_3, eo_4, eo_5]

#eo = msalb**2 + srad**2 + tmax**2 + tmin**2 + tmax**2 + xhlai**2

#################################################################

# PETASCE Model

tavg = (tmax + tmin)/2.0

patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26


psycon = 0.00066*patm


udelta = 2503.0*np.exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0


emax = 0.6108*np.exp((17.27*tmax)/(tmax+237.3))
emin = 0.6108*np.exp((17.27*tmin)/(tmin+237.3))
es = (emax + emin) / 2.0

ea = 0.6108*np.exp((17.27*tdew)/(tdew+237.3))

rhmin = max(20.0, min(80.0, ea/emax*100.0))


#if xhlai < 0:
albedo_0 = msalb
#else:
albedo_1 = 0.23

albedo = [albedo_0, albedo_1]

rns = (1.0-albedo)*srad
#rns_0 = (1.0-albedo_0)*srad
#rns_1 = (1.0-albedo_1)*srad

dr = 1.0+0.033*np.cos(2.0*pi/365.0*doy)
ldelta = 0.409*np.sin(2.0*pi/365.0*doy-1.39)
ws = np.arccos(-1.0*np.tan(xlat*pi/180.0)*np.tan(ldelta))
ra1 = ws*np.sin(xlat*pi/180.0)*np.sin(ldelta)
ra2 = np.cos(xlat*pi/180.0)*np.cos(ldelta)*np.sin(ws)
ra = 24.0/pi*4.92*dr*(ra1+ra2)


rso = (0.75+2*pow(10,-5)*xelev)*ra

ratio = srad/rso

#if ratio < 0.3:
ratio_0 = 0.3
#elif ratio > 1.0:
ratio_1 = 1.0

ratio = [ratio_0, ratio_1]

fcd = 1.35*ratio-0.35
#fcd_0 = 1.35*ratio_0-0.35
#fcd_1 = 1.35*ratio_1-0.35
tk4 = ((tmax+273.16)**4.0+(tmin+273.16)**4.0)/2.0
rnl = 4.901*pow(10,-9)*fcd*(0.34-0.14*np.sqrt(ea))*tk4
#rnl_0 = 4.901*pow(10,-9)*fcd_0*(0.34-0.14*np.sqrt(ea))*tk4
#rnl_1 = 4.901*pow(10,-9)*fcd_1*(0.34-0.14*np.sqrt(ea))*tk4

rn = rns - rnl
#rn_0 = rns_0 - rnl_0
#rn_1 = rns_0 - rnl_1
#rn_2 = rns_1 - rnl_0
#rn_3 = rns_1 - rnl_1

g = 0.0

windsp = windrun*1000.0 / 24.0 / 60.0 / 60.0
wind2m = windsp*(4.87/np.log(67.8*windht-5.42))

#if meevp == 'A':
cn_0 = 1600.0
cd_0 = 0.38
#elif meevp == 'G':
cn_1 = 900.0
cd_1 = 0.34


refet_0 = 0.408*udelta*(rn_0-g)+psycon*(cn_0/(tavg+273.0))*wind2m*(es-ea)
refet_0 = refet_0/(udelta+psycon*(1.0+cd_0*wind2m))
refet_1 = 0.408*udelta*(rn_0-g)+psycon*(cn_1/(tavg+273.0))*wind2m*(es-ea)
refet_1 = refet_1/(udelta+psycon*(1.0+cd_1*wind2m))
refet_2 = 0.408*udelta*(rn_1-g)+psycon*(cn_0/(tavg+273.0))*wind2m*(es-ea)
refet_2 = refet_2/(udelta+psycon*(1.0+cd_0*wind2m))
refet_3 = 0.408*udelta*(rn_1-g)+psycon*(cn_1/(tavg+273.0))*wind2m*(es-ea)
refet_3 = refet_3/(udelta+psycon*(1.0+cd_1*wind2m))

refet = [refet_0, refet_1, refet_2, refet_3]
kcbmax = 1.2
kcbmin = 0.3
skc = 0.8

refet = max(0.0001, refet)
#refet_0 = max(0.0001, refet_0)
#refet_1 = max(0.0001, refet_1)
#refet_2 = max(0.0001, refet_2)
#refet_3 = max(0.0001, refet_3)


#if xhlai < 0:
kcb_0 = 0.0
#else:
kcb_1 = max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-np.exp(-1.0*skc*xhlai)))

kcb = [kcb_0, kcb_1]

wnd = max(1.0, min(wind2m,6.0))
cht = max(0.001, canht)

#if meevp =='A':
kcmax_0 = max(1.0, kcb_0+0.05)
#kcmax_1 = max(1.0, kcb_1+0.05)
#elif meevp == 'G':
kcmax_1 = max((1.2+(0.04*(wnd-2.0)-0.004*(rhmin-45.0))
        *(cht/3.0)**(0.3)),kcb+0.05)
#kcmax_2 = max((1.2+(0.04*(wnd-2.0)-0.004*(rhmin-45.0))
#        *(cht/3.0)**(0.3)),kcb_0+0.05)
#kcmax_3 = max((1.2+(0.04*(wnd-2.0)-0.004*(rhmin-45.0))
#        *(cht/3.0)**(0.3)),kcb_1+0.05)


kcmax = [kcmax_0, kcmax_1]

#if kcb < kcbmin:
fc_0 = 0.0
#else:
fc_1 = ((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+0.5*canht)
#fc_1 = ((kcb_0-kcbmin)/(kcmax_0-kcbmin))**(1.0+0.5*canht)
#fc_2 = ((kcb_1-kcbmin)/(kcmax_0-kcbmin))**(1.0+0.5*canht)
#fc_3 = ((kcb_0-kcbmin)/(kcmax_1-kcbmin))**(1.0+0.5*canht)
#fc_4 = ((kcb_1-kcbmin)/(kcmax_1-kcbmin))**(1.0+0.5*canht)
#fc_5 = ((kcb_0-kcbmin)/(kcmax_2-kcbmin))**(1.0+0.5*canht)
#fc_6 = ((kcb_1-kcbmin)/(kcmax_2-kcbmin))**(1.0+0.5*canht)
#fc_7 = ((kcb_0-kcbmin)/(kcmax_3-kcbmin))**(1.0+0.5*canht)
#fc_8 = ((kcb_1-kcbmin)/(kcmax_3-kcbmin))**(1.0+0.5*canht)

fc = [fc_0, fc_1]


fw = 1.0
few = min(1.0-fc,fw)
#few_0 = min(1.0-fc_0,fw)
#few_1 = min(1.0-fc_1,fw)
#few_2 = min(1.0-fc_2,fw)
#few_3 = min(1.0-fc_3,fw)
#few_4 = min(1.0-fc_4,fw)
#few_5 = min(1.0-fc_5,fw)
#few_6 = min(1.0-fc_6,fw)
#few_7 = min(1.0-fc_7,fw)
#few_8 = min(1.0-fc_8,fw)




#ke_0 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_0*kcmax_0))
#ke_1 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_1*kcmax_0))
#ke_2 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_2*kcmax_0))
#ke_3 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_3*kcmax_0))
#ke_4 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_4*kcmax_0))
#ke_5 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_5*kcmax_0))
#ke_6 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_6*kcmax_0))
#ke_7 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_7*kcmax_0))
#ke_8 = max(0.0, min(1.0*(kcmax_0-kcb_0), few_8*kcmax_0))


#   ke_9 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_0*kcmax_0))
#   ke_10 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_1*kcmax_0))
#   ke_11 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_2*kcmax_0))
#   ke_12 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_3*kcmax_0))
#   ke_13 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_4*kcmax_0))
#   ke_14 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_5*kcmax_0))
#   ke_15 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_6*kcmax_0))
#   ke_16 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_7*kcmax_0))
#   ke_17 = max(0.0, min(1.0*(kcmax_0-kcb_1), few_8*kcmax_0))


 #   ke_18 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_0*kcmax_1))
 #   ke_19 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_1*kcmax_1))
 #   ke_20 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_2*kcmax_1))
 #   ke_21 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_3*kcmax_1))
 #   ke_22 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_4*kcmax_1))
 #   ke_23 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_5*kcmax_1))
 #   ke_24 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_6*kcmax_1))
 #   ke_25 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_7*kcmax_1))
 #   ke_26 = max(0.0, min(1.0*(kcmax_1-kcb_0), few_8*kcmax_1))



  #  ke_27 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_0*kcmax_1))
  #  ke_28 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_1*kcmax_1))
  #  ke_29 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_2*kcmax_1))
  #  ke_30 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_3*kcmax_1))
  #  ke_31 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_4*kcmax_1))
  #  ke_32 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_5*kcmax_1))
  #  ke_33 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_6*kcmax_1))
  #  ke_34 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_7*kcmax_1))
  #  ke_35 = max(0.0, min(1.0*(kcmax_1-kcb_1), few_8*kcmax_1))


   # ke_36 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_0*kcmax_2))
   # ke_37 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_1*kcmax_2))
   # ke_38 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_2*kcmax_2))
   # ke_39 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_3*kcmax_2))
   # ke_40 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_4*kcmax_2))
   # ke_41 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_5*kcmax_2))
   # ke_42 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_6*kcmax_2))
   # ke_43 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_7*kcmax_2))
   # ke_44 = max(0.0, min(1.0*(kcmax_2-kcb_0), few_8*kcmax_2))

 #   ke_45 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_0*kcmax_2))
 #   ke_46 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_1*kcmax_2))
 #   ke_47 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_2*kcmax_2))
 #   ke_48 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_3*kcmax_2))
 #   ke_49 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_4*kcmax_2))
 #   ke_50 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_5*kcmax_2))
 #   ke_51 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_6*kcmax_2))
 #   ke_52 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_7*kcmax_2))
 #   ke_53 = max(0.0, min(1.0*(kcmax_2-kcb_1), few_8*kcmax_2))



 #   ke_54 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_0*kcmax_3))
 #   ke_55 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_1*kcmax_3))
 #   ke_56 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_2*kcmax_3))
 #   ke_57 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_3*kcmax_3))
 #   ke_58 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_4*kcmax_3))
 #   ke_59 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_5*kcmax_3))
 #   ke_60 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_6*kcmax_3))
 #   ke_61 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_7*kcmax_3))
 #   ke_62 = max(0.0, min(1.0*(kcmax_3-kcb_0), few_8*kcmax_3))


 #   ke_63 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_0*kcmax_3))
 #   ke_64 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_1*kcmax_3))
 #   ke_65 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_2*kcmax_3))
 #   ke_66 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_3*kcmax_3))
 #   ke_67 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_4*kcmax_3))
 #   ke_68 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_5*kcmax_3))
 #   ke_69 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_6*kcmax_3))
 #   ke_70 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_7*kcmax_3))
 #   ke_71 = max(0.0, min(1.0*(kcmax_3-kcb_1), few_8*kcmax_3))


 #   kc_0 = kcb_0 + ke_0
 #   kc_1 = kcb_0 + ke_1
 #   kc_2 = kcb_0 + ke_2
 #   kc_3 = kcb_0 + ke_3
 #   kc_4 = kcb_0 + ke_4
 #   kc_5 = kcb_0 + ke_5
 #   kc_6 = kcb_0 + ke_6
 #   kc_7 = kcb_0 + ke_7
 #   kc_8 = kcb_0 + ke_8

 #   kc_9 = kcb_0 + ke_9
 #   kc_10 = kcb_0 + ke_10
 #   kc_11 = kcb_0 + ke_11
 #   kc_12 = kcb_0 + ke_12
 #   kc_13 = kcb_0 + ke_13
 #   kc_14 = kcb_0 + ke_14
 #   kc_15 = kcb_0 + ke_15
 #   kc_16 = kcb_0 + ke_16
 #   kc_17 = kcb_0 + ke_17


 #   kc_18 = kcb_0 + ke_18
 #   kc_19 = kcb_0 + ke_19
 #   kc_20 = kcb_0 + ke_20
 #   kc_21 = kcb_0 + ke_21
 #   kc_22 = kcb_0 + ke_22
 #   kc_23 = kcb_0 + ke_23
 #   kc_24 = kcb_0 + ke_24
 #   kc_25 = kcb_0 + ke_25
 #   kc_26 = kcb_0 + ke_26

 #   kc_27 = kcb_0 + ke_27
 #   kc_28 = kcb_0 + ke_28
 #   kc_29 = kcb_0 + ke_29
 #   kc_30 = kcb_0 + ke_30
 #   kc_31 = kcb_0 + ke_31
 #   kc_32 = kcb_0 + ke_32
 #   kc_33 = kcb_0 + ke_33
 #   kc_34 = kcb_0 + ke_34
 #   kc_35 = kcb_0 + ke_35

 #   kc_36 = kcb_0 + ke_36
 #   kc_37 = kcb_0 + ke_37
 #   kc_38 = kcb_0 + ke_38
 #   kc_39 = kcb_0 + ke_39
 #   kc_40 = kcb_0 + ke_40
 #   kc_41 = kcb_0 + ke_41
 #   kc_42 = kcb_0 + ke_42
 #   kc_43 = kcb_0 + ke_43
 #   kc_44 = kcb_0 + ke_44

 #   kc_45 = kcb_0 + ke_45
 #   kc_46 = kcb_0 + ke_46
 #   kc_47 = kcb_0 + ke_47
 #   kc_48 = kcb_0 + ke_48
 #   kc_49 = kcb_0 + ke_49
 #   kc_50 = kcb_0 + ke_50
 #   kc_51 = kcb_0 + ke_51
 #   kc_52 = kcb_0 + ke_52
 #   kc_53 = kcb_0 + ke_53

 #   kc_54 = kcb_0 + ke_54
 #   kc_55 = kcb_0 + ke_55
 #   kc_56 = kcb_0 + ke_56
 #   kc_57 = kcb_0 + ke_57
 #   kc_58 = kcb_0 + ke_58
 #   kc_59 = kcb_0 + ke_59
 #   kc_60 = kcb_0 + ke_60
 #   kc_61 = kcb_0 + ke_61
 #   kc_62 = kcb_0 + ke_62

 #   kc_63 = kcb_0 + ke_63
 #   kc_64 = kcb_0 + ke_64
 #   kc_65 = kcb_0 + ke_65
 #   kc_66 = kcb_0 + ke_66
 #   kc_67 = kcb_0 + ke_67
 #   kc_68 = kcb_0 + ke_68
 #   kc_69 = kcb_0 + ke_69
 #   kc_70 = kcb_0 + ke_70
 #   kc_71 = kcb_0 + ke_71

 #   kc_72 = kcb_1 + ke_0
 #   kc_73 = kcb_1 + ke_1
 #   kc_74 = kcb_1 + ke_2
 #   kc_75 = kcb_1 + ke_3
 #   kc_76 = kcb_1 + ke_4
 #   kc_77 = kcb_1 + ke_5
 #   kc_78 = kcb_1 + ke_6
 #   kc_79 = kcb_1 + ke_7
 #   kc_80 = kcb_1 + ke_8

 #   kc_81 = kcb_1 + ke_9
 #   kc_82 = kcb_1 + ke_10
 #   kc_83 = kcb_1 + ke_11
 #   kc_84 = kcb_1 + ke_12
 #   kc_85 = kcb_1 + ke_13
 #   kc_86 = kcb_1 + ke_14
 #   kc_87 = kcb_1 + ke_15
 #   kc_88 = kcb_1 + ke_16
 #   kc_89 = kcb_1 + ke_17


 #   kc_90 = kcb_1 + ke_18
 #   kc_91 = kcb_1 + ke_19
 #   kc_92 = kcb_1 + ke_20
 #   kc_93 = kcb_1 + ke_21
 #   kc_94 = kcb_1 + ke_22
 #   kc_95 = kcb_1 + ke_23
 #   kc_96 = kcb_1 + ke_24
 #   kc_97 = kcb_1 + ke_25
 #   kc_98 = kcb_1 + ke_26

 #   kc_99 = kcb_1 + ke_27
 #   kc_100 = kcb_1 + ke_28
 #   kc_101= kcb_1 + ke_29
 #   kc_102 = kcb_1 + ke_30
 #   kc_103 = kcb_1 + ke_31
 #   kc_104 = kcb_1 + ke_32
 #   kc_105 = kcb_1 + ke_33
 #   kc_106 = kcb_1 + ke_34
 #   kc_107 = kcb_1 + ke_35

 #   kc_108 = kcb_1 + ke_36
 #   kc_109 = kcb_1 + ke_37
 #   kc_110 = kcb_1 + ke_38
 #   kc_111 = kcb_1 + ke_39
 #   kc_112 = kcb_1 + ke_40
 #   kc_113 = kcb_1 + ke_41
 #   kc_114 = kcb_1 + ke_42
 #   kc_115 = kcb_1 + ke_43
 #   kc_116 = kcb_1 + ke_44

 #   kc_117 = kcb_1 + ke_45
 #   kc_118 = kcb_1 + ke_46
 #   kc_119 = kcb_1 + ke_47
 #   kc_120 = kcb_1 + ke_48
 #   kc_121 = kcb_1 + ke_49
 #   kc_122 = kcb_1 + ke_50
 #   kc_123 = kcb_1 + ke_51
 #   kc_124 = kcb_1 + ke_52
 #   kc_125 = kcb_1 + ke_53

 #   kc_126 = kcb_1 + ke_54
 #   kc_127 = kcb_1 + ke_55
 #   kc_128 = kcb_1 + ke_56
 #   kc_129 = kcb_1 + ke_57
 #   kc_130 = kcb_1 + ke_58
 #   kc_131 = kcb_1 + ke_59
 #   kc_132 = kcb_1 + ke_60
 #   kc_133 = kcb_1 + ke_61
 #   kc_134 = kcb_1 + ke_62

 #   kc_135 = kcb_1 + ke_63
 #   kc_136 = kcb_1 + ke_64
 #   kc_137 = kcb_1 + ke_65
 #   kc_138 = kcb_1 + ke_66
 #   kc_139 = kcb_1 + ke_67
 #   kc_140 = kcb_1 + ke_68
 #   kc_141 = kcb_1 + ke_69
 #   kc_142 = kcb_1 + ke_70
 #   kc_143 = kcb_1 + ke_71

kc = kcb + ke

eo = (kcb + ke)*refet

eo = max(eo, 0.0001)


fp_msalb = eo.diff(msalb)
fp_srad = eo.diff(srad)
fp_tmax = eo.diff(tmax)
fp_tmin = eo.diff(tmin)
fp_xhlai = eo.diff(xhlai)

grad = [fp_msalb, fp_srad, fp_tmax, fp_tmin, fp_xhlai]

alpha = 0.1
iterations = 0
check = 0
precision = 0.000001
printData = True
maxIterations = 1000

#theta = random.uniform(0, 1); print(theta)
#theta1 = random.uniform(1, 35); print(theta1)
#theta2 = random.uniform(-30, 60); print(theta2)
#theta3 = random.uniform(-30, theta2); print(theta3)
#theta4 = random.randint(1, 20); print(theta4)

msalb_val = np.linspace(0, 1, 100)
srad_val = np.linspace(1, 35, 100)
tmax_val = np.linspace(-30, 60, 100)
tmin_val = np.zeros(len(tmax_val))
for i in range(len(tmax_val)):
    tmin_val[i] = random.uniform(-30, tmax_val[i])
xhlai_val = np.linspace(1, 20, 100)


for theta in msalb_val:
    for theta1 in srad_val:
        for theta2 in tmax_val:
            for theta3 in tmin_val:
                for theta4 in xhlai_val:
                    while True:
                        temptheta = theta - alpha*(fp_msalb.subs(msalb, theta)).subs(srad, theta1).subs(tmax, theta2).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                        temptheta1 = theta1 - alpha*(fp_srad.subs(srad, theta1)).subs(msalb, theta).subs(tmax, theta2).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                        temptheta2 = theta2 - alpha*(fp_tmax.subs(tmax, theta2)).subs(msalb, theta).subs(srad, theta1).subs(tmin, theta3).subs(xhlai, theta4).evalf()
                        temptheta3 = theta3 - alpha*(fp_tmin.subs(tmin, theta3)).subs(msalb, theta).subs(srad, theta1).subs(tmax, theta2).subs(xhlai, theta4).evalf()
                        temptheta4 = theta4 - alpha*(fp_xhlai.subs(xhlai, theta4)).subs(msalb, theta).subs(srad, theta1).subs(tmax, theta2).subs(tmax, theta3).evalf()

                        iterations += 1
                        if iterations > maxIterations:
                            print("Adjust alpha values and check if function is convex")
                            printData = False
                            break

                        if abs(temptheta - theta) < precision and abs(temptheta1 - theta1) < precision and abs(temptheta2 - theta2) < precision and abs(temptheta3 - theta3) < precision and abs(temptheta4 - theta4) < precision:
                            break

                        theta = temptheta; 
                        theta1 = temptheta1; 
                        theta2 = temptheta2; 
                        theta3 = temptheta3; 
                        theta4 = temptheta4; 

                iterations = 0;
                if printData:
                    print("The function converges to a minimum")
                    print("Number of iterations:",iterations, sep='')
                    print("theta (msalb) =",theta,sep=" ")
                    print("theta1 (srad) =",theta1,sep=" ")
                    print("theta2 (tmax) =",theta2,sep=" ")
                    print("theta3 (tmin) =",theta3,sep=" ")
                    print("theta4 (xhlai) =",theta4,sep=" ")
