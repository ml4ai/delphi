from numbers import Real
from random import random
from delphi.translators.for2py.strings import *
import numpy as np
import delphi.translators.for2py.math_ext as math

def mini_PET__yr_doy__assign__yr__0(yrdoy: int):
    return int(int((yrdoy/1000)))

def mini_PET__yr_doy__assign__doy__0(yrdoy: int, yr: int):
    return int((yrdoy-(yr*1000)))

def mini_PET__vpslop__assign__vpslop_return__0(t: Real, vpsat_f8087: Real):
    return (((18.0*(2501.0-(2.373*t)))*vpsat_f8087)/(8.314*((t+273.0)**2)))

def mini_PET__vpsat__assign__vpsat_return__0(t: Real):
    return (610.78*np.exp(((17.269*t)/(t+237.3))))

def mini_PET__pet__assign__clouds__0(clouds: Real):
    return clouds

def mini_PET__pet__assign__srad__0(srad: Real):
    return srad

def mini_PET__pet__assign__tavg__0(tavg: Real):
    return tavg

def mini_PET__pet__assign__tdew__0(tdew: Real):
    return tdew

def mini_PET__pet__assign__tmax__0(tmax: Real):
    return tmax

def mini_PET__pet__assign__tmin__0(tmin: Real):
    return tmin

def mini_PET__pet__assign__vapr__0(vapr: Real):
    return vapr

def mini_PET__pet__assign__windht__0(windht: Real):
    return windht

def mini_PET__pet__assign__windsp__0(windsp: Real):
    return windsp

def mini_PET__pet__assign__windrun__0(windrun: Real):
    return windrun

def mini_PET__pet__assign__xlat__0(xlat: Real):
    return xlat

def mini_PET__pet__assign__xelev__0(xelev: Real):
    return xelev

def mini_PET__pet__assign__yrdoy__0(yrdoy: int):
    return yrdoy

def mini_PET__pet__condition__IF_0__0(meevp):
    return ((meevp == "a") or (meevp == "g"))

def mini_PET__pet__condition__IF_0__1(meevp):
    return (meevp == "f")

def mini_PET__pet__condition__IF_0__2(meevp):
    return (meevp == "d")

def mini_PET__pet__condition__IF_0__3(meevp):
    return (meevp == "p")

def mini_PET__pet__condition__IF_0__4(meevp):
    return (meevp == "m")

def mini_PET__petasce__assign__tavg__0(tmax: Real, tmin: Real):
    return ((tmax+tmin)/2.0)

def mini_PET__petasce__assign__patm__0(xelev: Real):
    return (101.3*(((293.0-(0.0065*xelev))/293.0)**5.26))

def mini_PET__petasce__assign__psycon__0(patm: Real):
    return (0.000665*patm)

def mini_PET__petasce__assign__udelta__0(tavg: Real):
    return ((2503.0*np.exp(((17.27*tavg)/(tavg+237.3))))/((tavg+237.3)**2.0))

def mini_PET__petasce__assign__emax__0(tmax: Real):
    return (0.6108*np.exp(((17.27*tmax)/(tmax+237.3))))

def mini_PET__petasce__assign__emin__0(tmin: Real):
    return (0.6108*np.exp(((17.27*tmin)/(tmin+237.3))))

def mini_PET__petasce__assign__es__0(emax: Real, emin: Real):
    return ((emax+emin)/2.0)

def mini_PET__petasce__assign__ea__0(tdew: Real):
    return (0.6108*np.exp(((17.27*tdew)/(tdew+237.3))))

def mini_PET__petasce__assign__rhmin__0(ea: Real, emax: Real):
    return np.maximum(np.full_like(ea, 20.0), np.minimum(np.full_like(ea, 80.0), ((ea/emax)*100.0)))

def mini_PET__petasce__condition__IF_0__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petasce__assign__albedo__0(msalb: Real):
    return msalb

def mini_PET__petasce__assign__albedo__1():
    return 0.23

def mini_PET__petasce__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, albedo_1, albedo_0)

def mini_PET__petasce__assign__rns__0(albedo: Real, srad: Real):
    return ((1.0-albedo)*srad)

def mini_PET__petasce__assign__pie__0():
    return 3.14159265359

def mini_PET__petasce__assign__dr__0(pie: Real, doy: int):
    return (1.0+(0.033*np.cos((((2.0*pie)/365.0)*doy))))

def mini_PET__petasce__assign__ldelta__0(pie: Real, doy: int):
    return (0.409*np.sin(((((2.0*pie)/365.0)*doy)-1.39)))

def mini_PET__petasce__assign__ws__0(xlat: Real, pie: Real, ldelta: Real):
    return np.arccos(-(((1.0*np.tan(((xlat*pie)/180.0)))*np.tan(ldelta))))

def mini_PET__petasce__assign__ra1__0(ws: Real, xlat: Real, pie: Real, ldelta: Real):
    return ((ws*np.sin(((xlat*pie)/180.0)))*np.sin(ldelta))

def mini_PET__petasce__assign__ra2__0(xlat: Real, pie: Real, ldelta: Real, ws: Real):
    return ((np.cos(((xlat*pie)/180.0))*np.cos(ldelta))*np.sin(ws))

def mini_PET__petasce__assign__ra__0(pie: Real, dr: Real, ra1: Real, ra2: Real):
    return ((((24.0/pie)*4.92)*dr)*(ra1+ra2))

def mini_PET__petasce__assign__rso__0(xelev: Real, ra: Real):
    return ((0.75+(2e-05*xelev))*ra)

def mini_PET__petasce__assign__ratio__0(srad: Real, rso: Real):
    return (srad/rso)

def mini_PET__petasce__condition__IF_1__0(ratio: Real):
    return (ratio < 0.3)

def mini_PET__petasce__assign__ratio__1():
    return 0.3

def mini_PET__petasce__decision__ratio__2(ratio_0: Real, ratio_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, ratio_1, ratio_0)

def mini_PET__petasce__condition__IF_1__1(ratio: Real):
    return (ratio > 1.0)

def mini_PET__petasce__assign__ratio__3():
    return 1.0

def mini_PET__petasce__decision__ratio__4(ratio_0: Real, ratio_1: Real, IF_1_1: bool):
    return np.where(IF_1_1, ratio_1, ratio_0)

def mini_PET__petasce__assign__fcd__0(ratio: Real):
    return ((1.35*ratio)-0.35)

def mini_PET__petasce__assign__tk4__0(tmax: Real, tmin: Real):
    return ((((tmax+273.16)**4.0)+((tmin+273.16)**4.0))/2.0)

def mini_PET__petasce__assign__rnl__0(fcd: Real, ea: Real, tk4: Real):
    return (((4.901e-09*fcd)*(0.34-(0.14*np.sqrt(ea))))*tk4)

def mini_PET__petasce__assign__rn__0(rns: Real, rnl: Real):
    return (rns-rnl)

def mini_PET__petasce__assign__g__0():
    return 0.0

def mini_PET__petasce__assign__windsp__0(windrun: Real):
    return ((((windrun*1000.0)/24.0)/60.0)/60.0)

def mini_PET__petasce__assign__wind2m__0(windsp: Real, windht: Real):
    return (windsp*(4.87/np.log(((67.8*windht)-5.42))))

def mini_PET__petasce__condition__IF_2__0(meevp):
    return (meevp == "a")

def mini_PET__petasce__assign__cn__0():
    return 1600.0

def mini_PET__petasce__assign__cd__0():
    return 0.38

def mini_PET__petasce__decision__cn__1(cn_0: Real, cn_1: Real, IF_2_0: bool):
    return np.where(IF_2_0, cn_1, cn_0)

def mini_PET__petasce__decision__cd__1(cd_0: Real, cd_1: Real, IF_2_0: bool):
    return np.where(IF_2_0, cd_1, cd_0)

def mini_PET__petasce__condition__IF_2__1(meevp):
    return (meevp == "g")

def mini_PET__petasce__assign__cn__2():
    return 900.0

def mini_PET__petasce__assign__cd__2():
    return 0.34

def mini_PET__petasce__decision__cn__3(cn_0: Real, cn_1: Real, IF_2_1: bool):
    return np.where(IF_2_1, cn_1, cn_0)

def mini_PET__petasce__decision__cd__3(cd_0: Real, cd_1: Real, IF_2_1: bool):
    return np.where(IF_2_1, cd_1, cd_0)

def mini_PET__petasce__assign__refet__0(udelta: Real, rn: Real, g: Real, psycon: Real, cn: Real, tavg: Real, wind2m: Real, es: Real, ea: Real):
    return (((0.408*udelta)*(rn-g))+(((psycon*(cn/(tavg+273.0)))*wind2m)*(es-ea)))

def mini_PET__petasce__assign__refet__1(refet: Real, udelta: Real, psycon: Real, cd: Real, wind2m: Real):
    return (refet/(udelta+(psycon*(1.0+(cd*wind2m)))))

def mini_PET__petasce__assign__refet__2(refet: Real):
    return np.maximum(np.full_like(refet, 0.0001), refet)

def mini_PET__petasce__condition__IF_3__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petasce__assign__kcb__0():
    return 0.0

def mini_PET__petasce__assign__kcb__1(kcbmin: Real, kcbmax: Real, skc: Real, xhlai: Real):
    return np.maximum(np.full_like(kcbmin, 0.0), (kcbmin+((kcbmax-kcbmin)*(1.0-np.exp(-(((1.0*skc)*xhlai)))))))

def mini_PET__petasce__decision__kcb__2(kcb_0: Real, kcb_1: Real, IF_3_0: bool):
    return np.where(IF_3_0, kcb_1, kcb_0)

def mini_PET__petasce__assign__wnd__0(wind2m: Real):
    return np.maximum(np.full_like(wind2m, 1.0), np.minimum(wind2m, np.full_like(wind2m, 6.0)))

def mini_PET__petasce__assign__cht__0(canht: Real):
    return np.maximum(np.full_like(canht, 0.001), canht)

def mini_PET__petasce__condition__IF_4__0(meevp):
    return (meevp == "a")

def mini_PET__petasce__assign__kcmax__0(kcb: Real):
    return np.maximum(np.full_like(kcb, 1.0), (kcb+0.05))

def mini_PET__petasce__decision__kcmax__1(kcmax_0: Real, kcmax_1: Real, IF_4_0: bool):
    return np.where(IF_4_0, kcmax_1, kcmax_0)

def mini_PET__petasce__condition__IF_4__1(meevp):
    return (meevp == "g")

def mini_PET__petasce__assign__kcmax__2(wnd: Real, rhmin: Real, cht: Real, kcb: Real):
    return np.maximum((1.2+(((0.04*(wnd-2.0))-(0.004*(rhmin-45.0)))*((cht/3.0)**0.3))), (kcb+0.05))

def mini_PET__petasce__decision__kcmax__3(kcmax_0: Real, kcmax_1: Real, IF_4_1: bool):
    return np.where(IF_4_1, kcmax_1, kcmax_0)

def mini_PET__petasce__condition__IF_5__0(kcb: Real, kcbmin: Real):
    return (kcb <= kcbmin)

def mini_PET__petasce__assign__fc__0():
    return 0.0

def mini_PET__petasce__assign__fc__1(kcb: Real, kcbmin: Real, kcmax: Real, canht: Real):
    return (((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+(0.5*canht)))

def mini_PET__petasce__decision__fc__2(fc_0: Real, fc_1: Real, IF_5_0: bool):
    return np.where(IF_5_0, fc_1, fc_0)

def mini_PET__petasce__assign__fw__0():
    return 1.0

def mini_PET__petasce__assign__few__0(fc: Real, fw: Real):
    return np.minimum((1.0-fc), fw)

def mini_PET__petasce__assign__ke__0(kcmax: Real, kcb: Real, few: Real):
    return np.maximum(np.full_like(kcmax, 0.0), np.minimum((1.0*(kcmax-kcb)), (few*kcmax)))

def mini_PET__petasce__assign__kc__0(kcb: Real, ke: Real):
    return (kcb+ke)

def mini_PET__petasce__assign__eo__0(kcb: Real, ke: Real, refet: Real):
    return ((kcb+ke)*refet)

def mini_PET__petasce__assign__eo__1(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

def mini_PET__petpen__assign__shair__0():
    return 0.001005

def mini_PET__petpen__assign__patm__0():
    return 101300.0

def mini_PET__petpen__assign__sbzcon__0():
    return 4.903e-09

def mini_PET__petpen__assign__lhvap__0(tavg: Real):
    return ((2501.0-(2.373*tavg))*1000.0)

def mini_PET__petpen__assign__psycon__0(shair: Real, patm: Real, lhvap: Real):
    return (((shair*patm)/(0.622*lhvap))*1000000)

def mini_PET__petpen__assign__esat__0(vpsat_60883: Real, vpsat_1ab8c: Real):
    return ((vpsat_60883+vpsat_1ab8c)/2.0)

def mini_PET__petpen__condition__IF_0__0(vapr: Real):
    return (vapr > 1e-06)

def mini_PET__petpen__assign__eair__1(vapr: Real):
    return (vapr*1000.0)

def mini_PET__petpen__decision__eair__2(eair_0: Real, eair_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, eair_1, eair_0)

def mini_PET__petpen__assign__vpd__0(esat: Real, eair: Real):
    return np.maximum(np.full_like(esat, 0.0), (esat-eair))

def mini_PET__petpen__assign__s__0(vpslop_db18b: Real, vpslop_b0968: Real):
    return ((vpslop_db18b+vpslop_b0968)/2.0)

def mini_PET__petpen__assign__rt__0(tavg: Real):
    return (8.314*(tavg+273.0))

def mini_PET__petpen__assign__dair__0(patm: Real, eair: Real, rt: Real):
    return ((0.028966*(patm-(0.387*eair)))/rt)

def mini_PET__petpen__assign__vhcair__0(dair: Real, shair: Real):
    return (dair*shair)

def mini_PET__petpen__assign__refht__0():
    return 0.12

def mini_PET__petpen__assign__windsp_m__0(windsp: Real):
    return (windsp*1000.0)

def mini_PET__petpen__assign__k__0():
    return 0.41

def mini_PET__petpen__assign__d__0(refht: Real):
    return ((2.0/3.0)*refht)

def mini_PET__petpen__assign__zom__0(refht: Real):
    return (0.123*refht)

def mini_PET__petpen__assign__zoh__0(zom: Real):
    return (0.1*zom)

def mini_PET__petpen__assign__ra__0(windht: Real, d: Real, zom: Real, zoh: Real, k: Real, windsp_m: Real):
    return ((np.log(((windht-d)/zom))*np.log(((windht-d)/zoh)))/((k**2)*windsp_m))

def mini_PET__petpen__assign__rl__0():
    return 100

def mini_PET__petpen__assign__rs__0(rl: Real):
    return (rl/(0.5*2.88))

def mini_PET__petpen__assign__rs__1(rs: Real):
    return (rs/86400)

def mini_PET__petpen__assign__g__0():
    return 0.0

def mini_PET__petpen__condition__IF_1__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petpen__assign__albedo__0(msalb: Real):
    return msalb

def mini_PET__petpen__assign__albedo__1():
    return 0.23

def mini_PET__petpen__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, albedo_1, albedo_0)

def mini_PET__petpen__assign__tk4__0(tmax: Real, tmin: Real):
    return ((((tmax+273.0)**4)+((tmin+273.0)**4))/2.0)

def mini_PET__petpen__assign__radb__0(sbzcon: Real, tk4: Real, eair: Real, clouds: Real):
    return (((sbzcon*tk4)*(0.34-(0.14*np.sqrt((eair/1000)))))*((1.35*(1.0-clouds))-0.35))

def mini_PET__petpen__assign__rnet__0(albedo: Real, srad: Real, radb: Real):
    return (((1.0-albedo)*srad)-radb)

def mini_PET__petpen__assign__rnetmg__0(rnet: Real, g: Real):
    return (rnet-g)

def mini_PET__petpen__assign__et0__0(s: Real, rnetmg: Real, dair: Real, shair: Real, vpd: Real, ra: Real, psycon: Real, rs: Real):
    return (((s*rnetmg)+(((dair*shair)*vpd)/ra))/(s+(psycon*(1+(rs/ra)))))

def mini_PET__petpen__assign__et0__1(et0: Real, lhvap: Real):
    return (et0/(lhvap/1000000.0))

def mini_PET__petpen__condition__IF_2__0(xhlai: Real):
    return (xhlai <= 6.0)

def mini_PET__petpen__assign__xhlai__0(xhlai: Real):
    return xhlai

def mini_PET__petpen__assign__xhlai__1():
    return 6.0

def mini_PET__petpen__decision__xhlai__2(xhlai_0: Real, xhlai_1: Real, IF_2_0: bool):
    return np.where(IF_2_0, xhlai_1, xhlai_0)

def mini_PET__petpen__assign__kc__0(eoratio: Real, xhlai: Real):
    return (1.0+(((eoratio-1.0)*xhlai)/6.0))

def mini_PET__petpen__assign__eo__0(et0: Real, kc: Real):
    return (et0*kc)

def mini_PET__petpen__assign__eo__1(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0))

def mini_PET__petpen__assign__eo__2(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

def mini_PET__petdyn__assign__shair__0():
    return 0.001005

def mini_PET__petdyn__assign__patm__0():
    return 101300.0

def mini_PET__petdyn__assign__sbzcon__0():
    return 4.903e-09

def mini_PET__petdyn__assign__lhvap__0(tavg: Real):
    return ((2501.0-(2.373*tavg))*1000.0)

def mini_PET__petdyn__assign__psycon__0(shair: Real, patm: Real, lhvap: Real):
    return (((shair*patm)/(0.622*lhvap))*1000000)

def mini_PET__petdyn__assign__esat__0(vpsat_00ba9: Real, vpsat_c8f50: Real):
    return ((vpsat_00ba9+vpsat_c8f50)/2.0)

def mini_PET__petdyn__assign__vpd__0(esat: Real, eair: Real):
    return (esat-eair)

def mini_PET__petdyn__assign__s__0(vpslop_2749b: Real, vpslop_0af7a: Real):
    return ((vpslop_2749b+vpslop_0af7a)/2.0)

def mini_PET__petdyn__assign__rt__0(tavg: Real):
    return (8.314*(tavg+273.0))

def mini_PET__petdyn__assign__dair__0(patm: Real, eair: Real, rt: Real):
    return ((0.028966*(patm-(0.387*eair)))/rt)

def mini_PET__petdyn__assign__vhcair__0(dair: Real, shair: Real):
    return (dair*shair)

def mini_PET__petdyn__assign__windsp_m__0(windsp: Real):
    return ((windsp*1000.0)/86400.0)

def mini_PET__petdyn__assign__k__0():
    return 0.41

def mini_PET__petdyn__condition__IF_0__0(canht: Real):
    return (canht <= 0.1)

def mini_PET__petdyn__assign__zcrop__0():
    return (2.0+0.1)

def mini_PET__petdyn__assign__dcrop__0():
    return (0.75*0.1)

def mini_PET__petdyn__assign__zomc__0(dcrop: Real):
    return (0.25*(0.1-dcrop))

def mini_PET__petdyn__assign__zovc__0(zomc: Real):
    return (0.1*zomc)

def mini_PET__petdyn__assign__dfao__0():
    return ((2.0*0.1)/3.0)

def mini_PET__petdyn__assign__zomf__0():
    return (0.123*0.1)

def mini_PET__petdyn__assign__zohf__0(zomf: Real):
    return (0.1*zomf)

def mini_PET__petdyn__assign__zcrop__1(canht: Real):
    return (2.0+canht)

def mini_PET__petdyn__assign__dcrop__1(canht: Real):
    return (0.75*canht)

def mini_PET__petdyn__assign__zomc__1(canht: Real, dcrop: Real):
    return (0.25*(canht-dcrop))

def mini_PET__petdyn__assign__zovc__1(zomc: Real):
    return (0.1*zomc)

def mini_PET__petdyn__assign__dfao__1(canht: Real):
    return ((2.0*canht)/3.0)

def mini_PET__petdyn__assign__zomf__1(canht: Real):
    return (0.123*canht)

def mini_PET__petdyn__assign__zohf__1(zomf: Real):
    return (0.1*zomf)

def mini_PET__petdyn__decision__zcrop__2(zcrop_0: Real, zcrop_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, zcrop_1, zcrop_0)

def mini_PET__petdyn__decision__dfao__2(dfao_0: Real, dfao_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, dfao_1, dfao_0)

def mini_PET__petdyn__decision__zohf__2(zohf_0: Real, zohf_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, zohf_1, zohf_0)

def mini_PET__petdyn__decision__zomf__2(zomf_0: Real, zomf_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, zomf_1, zomf_0)

def mini_PET__petdyn__decision__dcrop__2(dcrop_0: Real, dcrop_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, dcrop_1, dcrop_0)

def mini_PET__petdyn__decision__zomc__2(zomc_0: Real, zomc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, zomc_1, zomc_0)

def mini_PET__petdyn__decision__zovc__2(zovc_0: Real, zovc_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, zovc_1, zovc_0)

def mini_PET__petdyn__assign__dlh__0(canht: Real, xhlai: Real):
    return ((1.1*np.maximum(np.full_like(canht, 0.1), canht))*np.log((1.0+((0.2*xhlai)**0.25))))

def mini_PET__petdyn__condition__IF_1__0(xhlai: Real):
    return (xhlai < 1.0)

def mini_PET__petdyn__assign__zolh__0(canht: Real, xhlai: Real):
    return (0.01+((0.3*np.maximum(np.full_like(canht, 0.1), canht))*((0.2*xhlai)**0.5)))

def mini_PET__petdyn__assign__zolh__1(canht: Real, dlh: Real):
    return ((0.3*np.maximum(np.full_like(canht, 0.1), canht))*(1.0-(dlh/np.maximum(np.full_like(canht, 0.1), canht))))

def mini_PET__petdyn__decision__zolh__2(zolh_0: Real, zolh_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, zolh_1, zolh_0)

def mini_PET__petdyn__assign__wind2c__0(windsp_m: Real, zcrop: Real, dcrop: Real, zomc: Real):
    return (((windsp_m*np.log(((10.0-0.075)/0.00625)))*np.log(((zcrop-dcrop)/zomc)))/(np.log(((10.0-dcrop)/zomc))*np.log(((2.0-0.075)/0.00625))))

def mini_PET__petdyn__assign__ra__0(zcrop: Real, dcrop: Real, zomc: Real, zovc: Real, k: Real, wind2c: Real):
    return (((np.log(((zcrop-dcrop)/zomc))*np.log(((zcrop-dcrop)/zovc)))/((k**2)*wind2c))/86400)

def mini_PET__petdyn__assign__hts__0():
    return 0.13

def mini_PET__petdyn__assign__rasoil__0(hts: Real, k: Real, windsp_m: Real):
    return (((np.log(((2.0-((2*hts)/3.0))/(0.123*hts)))*np.log(((2.0-((2*hts)/3.0))/((0.1*0.123)*hts))))/((k**2)*windsp_m))/86400)

def mini_PET__petdyn__assign__zos__0():
    return 0.01

def mini_PET__petdyn__assign__maxht__0():
    return 1.0

def mini_PET__petdyn__assign__rb__0(zos: Real, maxht: Real, k: Real, windsp_m: Real):
    return (((np.log((2.0/zos))*np.log(((0.83*maxht)/zos)))/((k**2)*windsp_m))/86400)

def mini_PET__petdyn__assign__ac__0(xhlai: Real):
    return (1-np.exp(-((0.5*xhlai))))

def mini_PET__petdyn__assign__asvar__0(ac: Real):
    return (1-ac)

def mini_PET__petdyn__assign__raero__0(ac: Real, ra: Real, asvar: Real, rasoil: Real):
    return ((ac*ra)+(asvar*rasoil))

def mini_PET__petdyn__assign__rl__0():
    return 100

def mini_PET__petdyn__condition__IF_2__0(xhlai: Real):
    return (xhlai >= 0.1)

def mini_PET__petdyn__assign__rs__0(rl: Real, xhlai: Real):
    return (rl/((1/0.5)*(1.0-np.exp(-((0.5*xhlai))))))

def mini_PET__petdyn__assign__rs__1(rl: Real):
    return (rl/(0.5*0.1))

def mini_PET__petdyn__decision__rs__2(rs_0: Real, rs_1: Real, IF_2_0: bool):
    return np.where(IF_2_0, rs_1, rs_0)

def mini_PET__petdyn__assign__rs__3(rs: Real):
    return (rs/86400)

def mini_PET__petdyn__assign__rtot__0(ac: Real, rs: Real, asvar: Real, rb: Real):
    return ((ac*rs)+(asvar*rb))

def mini_PET__petdyn__assign__g__0():
    return 0.0

def mini_PET__petdyn__condition__IF_3__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petdyn__assign__albedo__0(msalb: Real):
    return msalb

def mini_PET__petdyn__assign__albedo__1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*np.exp(-((0.75*xhlai)))))

def mini_PET__petdyn__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_3_0: bool):
    return np.where(IF_3_0, albedo_1, albedo_0)

def mini_PET__petdyn__assign__tk4__0(tmax: Real, tmin: Real):
    return ((((tmax+273.0)**4)+((tmin+273.0)**4))/2.0)

def mini_PET__petdyn__assign__radb__0(sbzcon: Real, tk4: Real, eair: Real, clouds: Real):
    return (((sbzcon*tk4)*(0.34-(0.14*np.sqrt((eair/1000)))))*((1.35*(1.0-clouds))-0.35))

def mini_PET__petdyn__assign__rnet__0(albedo: Real, srad: Real, radb: Real):
    return (((1.0-albedo)*srad)-radb)

def mini_PET__petdyn__assign__rnetmg__0(rnet: Real, g: Real):
    return (rnet-g)

def mini_PET__petdyn__assign__eo__0(s: Real, rnetmg: Real, dair: Real, shair: Real, vpd: Real, raero: Real, psycon: Real, rtot: Real):
    return (((s*rnetmg)+(((dair*shair)*vpd)/raero))/(s+(psycon*(1+(rtot/raero)))))

def mini_PET__petdyn__assign__eo__1(eo: Real, lhvap: Real):
    return (eo/(lhvap/1000000.0))

def mini_PET__petdyn__assign__eo__2(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

def mini_PET__petpt__assign__td__0(tmax: Real, tmin: Real):
    return ((0.6*tmax)+(0.4*tmin))

def mini_PET__petpt__condition__IF_0__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petpt__assign__albedo__0(msalb: Real):
    return msalb

def mini_PET__petpt__assign__albedo__1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*np.exp(-((0.75*xhlai)))))

def mini_PET__petpt__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, albedo_1, albedo_0)

def mini_PET__petpt__assign__slang__0(srad: Real):
    return (srad*23.923)

def mini_PET__petpt__assign__eeq__0(slang: Real, albedo: Real, td: Real):
    return ((slang*(0.000204-(0.000183*albedo)))*(td+29.0))

def mini_PET__petpt__assign__eo__0(eeq: Real):
    return (eeq*1.1)

def mini_PET__petpt__condition__IF_1__0(tmax: Real):
    return (tmax > 35.0)

def mini_PET__petpt__assign__eo__1(eeq: Real, tmax: Real):
    return (eeq*(((tmax-35.0)*0.05)+1.1))

def mini_PET__petpt__decision__eo__2(eo_0: Real, eo_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, eo_1, eo_0)

def mini_PET__petpt__condition__IF_1__1(tmax: Real):
    return (tmax < 5.0)

def mini_PET__petpt__assign__eo__3(eeq: Real, tmax: Real):
    return ((eeq*0.01)*np.exp((0.18*(tmax+20.0))))

def mini_PET__petpt__decision__eo__4(eo_0: Real, eo_1: Real, IF_1_1: bool):
    return np.where(IF_1_1, eo_1, eo_0)

def mini_PET__petpt__assign__eo__5(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

def mini_PET__petpno__assign__shair__0():
    return 1005.0

def mini_PET__petpno__assign__patm__0():
    return 101300.0

def mini_PET__petpno__assign__sbzcon__0():
    return 4.903e-09

def mini_PET__petpno__assign__lhvap__0(tavg: Real):
    return ((2501.0-(2.373*tavg))*1000.0)

def mini_PET__petpno__assign__psycon__0(shair: Real, patm: Real, lhvap: Real):
    return ((shair*patm)/(0.622*lhvap))

def mini_PET__petpno__assign__esat__0(vpsat_4cc77: Real, vpsat_0247c: Real):
    return ((vpsat_4cc77+vpsat_0247c)/2.0)

def mini_PET__petpno__assign__vpd__0(esat: Real, eair: Real):
    return (esat-eair)

def mini_PET__petpno__assign__s__0(vpslop_0d762: Real, vpslop_7a69c: Real):
    return ((vpslop_0d762+vpslop_7a69c)/2.0)

def mini_PET__petpno__assign__rt__0(tavg: Real):
    return (8.314*(tavg+273.0))

def mini_PET__petpno__assign__dair__0(rt: Real, patm: Real, eair: Real):
    return (((0.1*18.0)/rt)*(((patm-eair)/0.622)+eair))

def mini_PET__petpno__assign__vhcair__0(dair: Real, shair: Real):
    return (dair*shair)

def mini_PET__petpno__assign__g__0():
    return 0.0

def mini_PET__petpno__condition__IF_0__0(xhlai: Real):
    return (xhlai <= 0.0)

def mini_PET__petpno__assign__albedo__0(msalb: Real):
    return msalb

def mini_PET__petpno__assign__albedo__1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*np.exp(-((0.75*xhlai)))))

def mini_PET__petpno__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, albedo_1, albedo_0)

def mini_PET__petpno__assign__tk4__0(tmax: Real, tmin: Real):
    return ((((tmax+273.0)**4)+((tmin+273.0)**4))/2.0)

def mini_PET__petpno__assign__radb__0(sbzcon: Real, tk4: Real, eair: Real, clouds: Real):
    return (((sbzcon*tk4)*(0.4-(0.005*np.sqrt(eair))))*((1.1*(1.0-clouds))-0.1))

def mini_PET__petpno__assign__rnet__0(albedo: Real, srad: Real, radb: Real):
    return (((1.0-albedo)*srad)-radb)

def mini_PET__petpno__assign__wfnfao__0(windsp: Real):
    return (0.0027*(1.0+(0.01*windsp)))

def mini_PET__petpno__assign__rnetmg__0(rnet: Real, g: Real, lhvap: Real):
    return (((rnet-g)/lhvap)*1000000.0)

def mini_PET__petpno__assign__eo__0(s: Real, rnetmg: Real, psycon: Real, wfnfao: Real, vpd: Real):
    return (((s*rnetmg)+((psycon*wfnfao)*vpd))/(s+psycon))

def mini_PET__petpno__assign__eo__1(eo: Real):
    return np.maximum(eo, np.full_like(eo, 0.0001))

def mini_PET__petmey__assign__yrdoy__0(yrdoy: int):
    return yrdoy

def mini_PET__petmey__assign__yrsim__0(yrsim: int):
    return yrsim

def mini_PET__petmey__assign__crop__0(crop):
    return str(crop)[0:2].ljust(2, " ")

def mini_PET__petmey__assign__tav__0(meantemp: Real):
    return meantemp

def mini_PET__petmey__condition__IF_0__0(yrdoy: int, yrsim: int):
    return (yrdoy <= (yrsim+3))

def mini_PET__petmey__assign__tavt__0(tav: Real):
    return tav

def mini_PET__petmey__assign__tavy2__0(tav: Real):
    return tav

def mini_PET__petmey__assign__tavy1__0(tav: Real):
    return tav

def mini_PET__petmey__assign__tavy2__1(tavy1: Real):
    return tavy1

def mini_PET__petmey__assign__tavy1__1(tavt: Real):
    return tavt

def mini_PET__petmey__assign__tavt__1(tav: Real):
    return tav

def mini_PET__petmey__decision__tavy2__2(tavy2_0: Real, tavy2_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, tavy2_1, tavy2_0)

def mini_PET__petmey__decision__tavt__2(tavt_0: Real, tavt_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, tavt_1, tavt_0)

def mini_PET__petmey__decision__tavy1__2(tavy1_0: Real, tavy1_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, tavy1_1, tavy1_0)

def mini_PET__petmey__assign__t3day__0(tavy2: Real, tavy1: Real, tav: Real):
    return (((tavy2+tavy1)+tav)/3.0)

def mini_PET__petmey__condition__IF_1__0(crop):
    return (crop == "ri")

def mini_PET__petmey__assign__albedo__0(xhlai: Real):
    return (0.23-((0.23-0.05)*np.exp(-((0.75*xhlai)))))

def mini_PET__petmey__assign__albedo__1(msalb: Real, xhlai: Real):
    return (0.23-((0.23-msalb)*np.exp(-((0.75*xhlai)))))

def mini_PET__petmey__decision__albedo__2(albedo_0: Real, albedo_1: Real, IF_1_0: bool):
    return np.where(IF_1_0, albedo_1, albedo_0)

def mini_PET__petmey__assign__coeff_a__0():
    return 0.92

def mini_PET__petmey__assign__coeff_b__0():
    return 0.08

def mini_PET__petmey__assign__coeff_c__0():
    return 0.34

def mini_PET__petmey__assign__coeff_d__0():
    return -(0.139)

def mini_PET__petmey__assign__coeff_winda__0():
    return 17.8636

def mini_PET__petmey__assign__coeff_windb__0():
    return 0.044

def mini_PET__petmey__assign__stefboltz__0():
    return 4.896e-09

def mini_PET__petmey__assign__pi__0():
    return (22.0/7.0)

def mini_PET__petmey__assign__latheatvap__0(meantemp: Real):
    return (2.50025-(0.002365*meantemp))

def mini_PET__petmey__assign__delta__0(meantemp: Real):
    return ((0.1*np.exp((21.255-(5304/(meantemp+273.1)))))*(5304/((meantemp+273.1)**2)))

def mini_PET__petmey__assign__dodpg__0(delta: Real):
    return (delta/(delta+0.066))

def mini_PET__petmey__assign__radj__0(jday: int, pi: Real):
    return (((float(jday)/365.25)*pi)*2.0)

def mini_PET__petmey__assign__maxirradiance__0(radj: Real):
    return ((22.357+(11.0947*np.cos(radj)))-(2.3594*np.sin(radj)))

def mini_PET__petmey__assign__vpsatvar__0(meantemp: Real):
    return (0.611*np.exp(((17.27*meantemp)/(meantemp+237.3))))

def mini_PET__petmey__assign__vpdew__0(meandewpt: Real):
    return (0.611*np.exp(((17.27*meandewpt)/(meandewpt+237.3))))

def mini_PET__petmey__assign__vpdew__1(vpdew: Real, vpsatvar: Real):
    return np.minimum(vpdew, vpsatvar)

def mini_PET__petmey__assign__vpd__0(vpsatvar: Real, vpdew: Real):
    return (vpsatvar-vpdew)

def mini_PET__petmey__assign__netemissivity__0(coeff_c: Real, coeff_d: Real, vpdew: Real):
    return (coeff_c+(coeff_d*np.sqrt(vpdew)))

def mini_PET__petmey__assign__fac1__0(coeff_a: Real, solarirradiance: Real, maxirradiance: Real, coeff_b: Real):
    return ((coeff_a*(solarirradiance/maxirradiance))+coeff_b)

def mini_PET__petmey__assign__radlon__0(fac1: Real, netemissivity: Real, stefboltz: Real, meantemp: Real):
    return (((fac1*netemissivity)*stefboltz)*((meantemp+273.0)**4.0))

def mini_PET__petmey__assign__netrad__0(albedo: Real, solarirradiance: Real, radlon: Real):
    return (((1.0-albedo)*solarirradiance)-radlon)

def mini_PET__petmey__assign__gflux__0(meantemp: Real, t3day: Real):
    return ((meantemp-t3day)*0.12)

def mini_PET__petmey__assign__windfunc__0(coeff_winda: Real, coeff_windb: Real, dailywindrun: Real):
    return (coeff_winda+(coeff_windb*dailywindrun))

def mini_PET__petmey__assign__eo__0(dodpg: Real, netrad: Real, gflux: Real, windfunc: Real, vpd: Real, latheatvap: Real):
    return (((dodpg*(netrad-gflux))+(((1.0-dodpg)*windfunc)*vpd))/latheatvap)

def mini_PET__petmey__condition__IF_2__0(eo: Real):
    return (eo < 0.0)

def mini_PET__petmey__assign__eo__1():
    return 0.0

def mini_PET__petmey__decision__eo__2(eo_0: Real, eo_1: Real, IF_2_0: bool):
    return np.where(IF_2_0, eo_1, eo_0)

def mini_PET__pse__condition__IF_0__0(ke: Real):
    return (ke >= 0.0)

def mini_PET__pse__assign__eos__0(ke: Real, refet: Real):
    return (ke*refet)

def mini_PET__pse__decision__eos__1(eos_0: Real, eos_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, eos_1, eos_0)

def mini_PET__pse__condition__IF_0__1(ksevap: Real):
    return (ksevap <= 0.0)

def mini_PET__pse__condition__IF_0__2(xlai: Real):
    return (xlai <= 1.0)

def mini_PET__pse__assign__eos__2(eo: Real, xlai: Real):
    return (eo*(1.0-(0.39*xlai)))

def mini_PET__pse__assign__eos__3(eo: Real, xlai: Real):
    return ((eo/1.1)*np.exp(-((0.4*xlai))))

def mini_PET__pse__decision__eos__4(eos_0: Real, eos_1: Real, IF_0_2: bool):
    return np.where(IF_0_2, eos_1, eos_0)

def mini_PET__pse__assign__eos__5(eo: Real, ksevap: Real, xlai: Real):
    return (eo*np.exp(-((ksevap*xlai))))

def mini_PET__pse__decision__IF_0__3(IF_0_0: bool, IF_0_1: bool):
    return np.where(IF_0_1, IF_0_1, IF_0_0)

def mini_PET__pse__decision__eos__6(eos_0: Real, eos_1: Real, IF_0_1: bool):
    return np.where(IF_0_1, eos_1, eos_0)

def mini_PET__pse__assign__eos__7(eos: Real):
    return np.maximum(eos, np.full_like(eos, 0.0))

def mini_PET__flood_evap__assign__ef__0(eo: Real, xlai: Real):
    return (eo*(1.0-(0.53*xlai)))

def mini_PET__flood_evap__condition__IF_0__0(xlai: Real):
    return (xlai > 0.85)

def mini_PET__flood_evap__assign__ef__1(eo: Real, xlai: Real):
    return ((eo/1.1)*np.exp(-((0.6*xlai))))

def mini_PET__flood_evap__decision__ef__2(ef_0: Real, ef_1: Real, IF_0_0: bool):
    return np.where(IF_0_0, ef_1, ef_0)

