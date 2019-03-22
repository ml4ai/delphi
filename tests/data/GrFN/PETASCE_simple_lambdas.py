import math

def petasce__assign__tavg_1(tmax, tmin):
    return ((tmax+tmin)/2.0)

def petasce__assign__patm_1(xelev):
    return (101.3*(((293.0-(0.0065*xelev))/293.0)**5.26))

def petasce__assign__psycon_1(patm):
    return (0.000665*patm)

def petasce__assign__udelta_1(tavg):
    return ((2503.0*math.exp(((17.27*tavg)/(tavg+237.3))))/((tavg+237.3)**2.0))

def petasce__assign__emax_1(tmax):
    return (0.6108*math.exp(((17.27*tmax)/(tmax+237.3))))

def petasce__assign__emin_1(tmin):
    return (0.6108*math.exp(((17.27*tmin)/(tmin+237.3))))

def petasce__assign__es_1(emax, emin):
    return ((emax+emin)/2.0)

def petasce__assign__ea_1(tdew):
    return (0.6108*math.exp(((17.27*tdew)/(tdew+237.3))))

def petasce__assign__rhmin_1():
    return max(20.0, min(80.0, ((ea/emax)*100.0)))

def petasce__condition__IF_1_0(xhlai):
    return (xhlai<=0.0)

def petasce__assign__albedo_1(msalb):
    return msalb

def petasce__assign__albedo_2():
    return 0.23

def petasce__decision__albedo_3(IF_1_0, albedo_2, albedo_1):
    return albedo_1 if IF_1_0 else albedo_2

def petasce__assign__rns_1(albedo, srad):
    return ((1.0-albedo)*srad)

def petasce__assign__pie_1():
    return 3.14159265359

def petasce__assign__dr_1(pie, doy):
    return (1.0+(0.033*math.cos((((2.0*pie)/365.0)*doy))))

def petasce__assign__ldelta_1(pie, doy):
    return (0.409*math.sin(((((2.0*pie)/365.0)*doy)-1.39)))

def petasce__assign__ws_1():
    return math.acos(-(((1.0*math.tan(((xlat*pie)/180.0)))*math.tan(ldelta))))

def petasce__assign__ra1_1(ws, xlat, pie, ldelta):
    return ((ws*math.sin(((xlat*pie)/180.0)))*math.sin(ldelta))

def petasce__assign__ra2_1(xlat, pie, ldelta, ws):
    return ((math.cos(((xlat*pie)/180.0))*math.cos(ldelta))*math.sin(ws))

def petasce__assign__ra_1(pie, dr, ra1, ra2):
    return ((((24.0/pie)*4.92)*dr)*(ra1+ra2))

def petasce__assign__rso_1(xelev, ra):
    return ((0.75+(2e-05*xelev))*ra)

def petasce__assign__ratio_1(srad, rso):
    return (srad/rso)

def petasce__condition__IF_1_1(ratio):
    return (ratio<0.3)

def petasce__assign__ratio_2():
    return 0.3

def petasce__assign__ratio_3():
    return 1.0

def petasce__decision__ratio_3(IF_1_0, ratio_1, ratio_2):
    return ratio_2 if IF_1_0 else ratio_1

def petasce__condition__IF_2_0(ratio):
    return (ratio>1.0)

def petasce__decision__ratio_4(IF_2_0, ratio_3):
    return ratio_3 if IF_2_0 else ratio_3

def petasce__assign__fcd_1(ratio):
    return ((1.35*ratio)-0.35)

def petasce__assign__tk4_1(tmax, tmin):
    return ((((tmax+273.16)**4.0)+((tmin+273.16)**4.0))/2.0)

def petasce__assign__rnl_1(fcd, ea, tk4):
    return (((4.901e-09*fcd)*(0.34-(0.14*math.sqrt(ea))))*tk4)

def petasce__assign__rn_1(rns, rnl):
    return (rns-rnl)

def petasce__assign__g_1():
    return 0.0

def petasce__assign__windsp_1(windrun):
    return ((((windrun*1000.0)/24.0)/60.0)/60.0)

def petasce__assign__wind2m_1(windsp, windht):
    return (windsp*(4.87/math.log(((67.8*windht)-5.42))))

def petasce__condition__IF_1_2(meevp):
    return (meevp=="A")

def petasce__assign__cn_1():
    return 1600.0

def petasce__assign__cd_1():
    return 0.38

def petasce__assign__cn_2():
    return 900.0

def petasce__assign__cd_2():
    return 0.34

def petasce__decision__cn_2(IF_1_0, cn_0, cn_1):
    return cn_1 if IF_1_0 else cn_0

def petasce__condition__IF_2_1(meevp):
    return (meevp=="G")

def petasce__decision__cn_3(IF_2_0, cn_2):
    return cn_2 if IF_2_0 else cn_2

def petasce__decision__cd_2(IF_2_0, cd_0, cd_2):
    return cd_2 if IF_2_0 else cd_0

def petasce__decision__cd_3(IF_2_0, cd_0, cd_2):
    return cd_2 if IF_2_0 else cd_0

def petasce__assign__refet_1(udelta, rn, g, psycon, cn, tavg, wind2m, es, ea):
    return (((0.408*udelta)*(rn-g))+(((psycon*(cn/(tavg+273.0)))*wind2m)*(es-ea)))

def petasce__assign__refet_2(refet, udelta, psycon, cd, wind2m):
    return (refet/(udelta+(psycon*(1.0+(cd*wind2m)))))

def petasce__assign__refet_3():
    return max(0.0001, refet)

def petasce__assign__skc_1():
    return 0.8

def petasce__assign__kcbmin_1():
    return 0.3

def petasce__assign__kcbmax_1():
    return 1.2

def petasce__condition__IF_1_3(xhlai):
    return (xhlai<=0.0)

def petasce__assign__kcb_1():
    return 0.0

def petasce__assign__kcb_2():
    return max(0.0, (kcbmin+((kcbmax-kcbmin)*(1.0-math.exp(-(((1.0*skc)*xhlai)))))))

def petasce__decision__kcb_3(IF_1_0, kcb_2, kcb_1):
    return kcb_1 if IF_1_0 else kcb_2

def petasce__assign__wnd_1():
    return max(1.0, min(wind2m, 6.0))

def petasce__assign__cht_1():
    return max(0.001, canht)

def petasce__condition__IF_1_4(meevp):
    return (meevp=="A")

def petasce__assign__kcmax_1():
    return max(1.0, (kcb+0.05))

def petasce__assign__kcmax_2(wnd, rhmin, cht):
    return max((1.2+(((0.04*(wnd-2.0))-(0.004*(rhmin-45.0)))*((cht/3.0)**0.3))), (kcb+0.05))

def petasce__decision__kcmax_2(IF_1_0, kcmax_0, kcmax_1):
    return kcmax_1 if IF_1_0 else kcmax_0

def petasce__condition__IF_2_2(meevp):
    return (meevp=="G")

def petasce__decision__kcmax_3(IF_2_0, kcmax_2):
    return kcmax_2 if IF_2_0 else kcmax_2

def petasce__condition__IF_1_5(kcb, kcbmin):
    return (kcb<=kcbmin)

def petasce__assign__fc_1():
    return 0.0

def petasce__assign__fc_2(kcb, kcbmin, kcmax, canht):
    return (((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+(0.5*canht)))

def petasce__decision__fc_3(IF_1_0, fc_2, fc_1):
    return fc_1 if IF_1_0 else fc_2

def petasce__assign__fw_1():
    return 1.0

def petasce__assign__few_1(fc):
    return min((1.0-fc), fw)

def petasce__assign__ke_1():
    return max(0.0, min((1.0*(kcmax-kcb)), (few*kcmax)))

def petasce__assign__kc_1(kcb, ke):
    return (kcb+ke)

def petasce__assign__eo_0(kcb, ke, refet):
    return ((kcb+ke)*refet)

def petasce__assign__eo_1(eo):
    return max(eo, 0.0001)

