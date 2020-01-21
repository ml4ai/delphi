import numpy as np

def petasce__assign__tavg_1(tmax, tmin):
    return ((tmax+tmin)/2.0)

def petasce__assign__patm_1(xelev):
    return (101.3*(((293.0-(0.0065*xelev))/293.0)**5.26))

def petasce__assign__psycon_1(patm):
    return (0.000665*patm)

def petasce__assign__udelta_1(tavg):
    return ((2503.0*np.exp(((17.27*tavg)/(tavg+237.3))))/((tavg+237.3)**2.0))

def petasce__assign__emax_1(tmax):
    return (0.6108*np.exp(((17.27*tmax)/(tmax+237.3))))

def petasce__assign__emin_1(tmin):
    return (0.6108*np.exp(((17.27*tmin)/(tmin+237.3))))

def petasce__assign__es_1(emax, emin):
    return ((emax+emin)/2.0)

def petasce__assign__ea_1(tdew):
    return (0.6108*np.exp(((17.27*tdew)/(tdew+237.3))))

def petasce__assign__rhmin_1(ea, emax):
    return np.maximum(
        np.full_like(ea, 20.0),
        np.minimum(
            np.full_like(ea, 80.0),
            ((ea/emax)*100.0)
        )
    )

def petasce__condition__IF_1_0(xhlai):
    return (xhlai <= 0.0)

def petasce__assign__albedo_1(msalb):
    return msalb

def petasce__assign__albedo_2():
    return 0.23

def petasce__decision__albedo_3(IF_1_0, albedo_2, albedo_1):
    return np.where(IF_1_0, albedo_1, albedo_2)

def petasce__assign__rns_1(albedo, srad):
    return ((1.0-albedo)*srad)

def petasce__assign__pie_1():
    return 3.14159265359

def petasce__assign__dr_1(pie, doy):
    return (1.0+(0.033*np.cos((((2.0*pie)/365.0)*doy))))

def petasce__assign__ldelta_1(pie, doy):
    return (0.409*np.sin(((((2.0*pie)/365.0)*doy)-1.39)))

def petasce__assign__ws_1(xlat, pie, ldelta):
    return np.arccos(-(((1.0*np.tan(((xlat*pie)/180.0)))*np.tan(ldelta))))

def petasce__assign__ra1_1(ws, xlat, pie, ldelta):
    return ((ws*np.sin(((xlat*pie)/180.0)))*np.sin(ldelta))

def petasce__assign__ra2_1(xlat, pie, ldelta, ws):
    return ((np.cos(((xlat*pie)/180.0))*np.cos(ldelta))*np.sin(ws))

def petasce__assign__ra_1(pie, dr, ra1, ra2):
    return ((((24.0/pie)*4.92)*dr)*(ra1+ra2))

def petasce__assign__rso_1(xelev, ra):
    return ((0.75+(2e-05*xelev))*ra)

def petasce__assign__ratio_1(srad, rso):
    return (srad/rso)

def petasce__condition__IF_1_1(ratio):
    return (ratio < 0.3)

def petasce__assign__ratio_2():
    return 0.3

def petasce__assign__ratio_3():
    return 1.0

def petasce__decision__ratio_4(IF_1_1, ratio_1, ratio_2):
    return np.where(IF_1_1, ratio_2, ratio_1)

def petasce__condition__IF_2_0(ratio):
    return (ratio > 1.0)

def petasce__decision__ratio_5(IF_2_0, ratio_4, ratio_3):
    return np.where(IF_2_0, ratio_3, ratio_4)

def petasce__assign__fcd_1(ratio):
    return ((1.35*ratio)-0.35)

def petasce__assign__tk4_1(tmax, tmin):
    return ((((tmax+273.16)**4.0)+((tmin+273.16)**4.0))/2.0)

def petasce__assign__rnl_1(fcd, ea, tk4):
    return (((4.901e-09*fcd)*(0.34-(0.14*np.sqrt(ea))))*tk4)

def petasce__assign__rn_1(rns, rnl):
    return (rns-rnl)

def petasce__assign__g_1():
    return 0.0

def petasce__assign__windsp_1(windrun):
    return ((((windrun*1000.0)/24.0)/60.0)/60.0)

def petasce__assign__wind2m_1(windsp, windht):
    return (windsp*(4.87/np.log(((67.8*windht)-5.42))))

def petasce__assign__cn_1():
    return 0.0

def petasce__assign__cd_1():
    return 0.0

def petasce__condition__IF_1_2(meevp):
    return (meevp == "A")

def petasce__assign__cn_2():
    return 1600.0

def petasce__assign__cd_2():
    return 0.38

def petasce__assign__cn_3():
    return 900.0

def petasce__assign__cd_3():
    return 0.34

def petasce__decision__cn_4(IF_1_2, cn_1, cn_2):
    return np.where(IF_1_2, cn_2, cn_1)

def petasce__decision__cd_4(IF_1_2, cd_1, cd_2):
    return np.where(IF_1_2, cd_2, cd_1)

def petasce__condition__IF_2_1(meevp):
    return (meevp == "G")

def petasce__decision__cn_5(IF_2_1, cn_4, cn_3):
    return np.where(IF_2_1, cn_3, cn_4)

def petasce__decision__cd_5(IF_2_1, cd_4, cd_3):
    return np.where(IF_2_1, cd_3, cd_4)

def petasce__assign__refet_1(udelta, rn, g, psycon, cn, tavg, wind2m, es, ea):
    return (((0.408*udelta)*(rn-g))+(((psycon*(cn/(tavg+273.0)))*wind2m)*(es-ea)))

def petasce__assign__refet_2(refet, udelta, psycon, cd, wind2m):
    return (refet/(udelta+(psycon*(1.0+(cd*wind2m)))))

def petasce__assign__refet_3(refet):
    return np.maximum(np.full_like(0.0001, refet), refet)

def petasce__assign__skc_1():
    return 0.8

def petasce__assign__kcbmin_1():
    return 0.3

def petasce__assign__kcbmax_1():
    return 1.2

def petasce__condition__IF_1_3(xhlai):
    return (xhlai <= 0.0)

def petasce__assign__kcb_1():
    return 0.0

def petasce__assign__kcb_2(kcbmin, kcbmax, skc, xhlai):
    return np.maximum(
        np.full_like(kcbmin, 0.0),
        (kcbmin+((kcbmax-kcbmin)*(1.0-np.exp(-(((1.0*skc)*xhlai))))))
    )

def petasce__decision__kcb_3(IF_1_3, kcb_2, kcb_1):
    return np.where(IF_1_3, kcb_1, kcb_2)

def petasce__assign__wnd_1(wind2m):
    return np.maximum(
        np.full_like(wind2m, 1.0),
        np.minimum(
            wind2m,
            np.full_like(wind2m, 6.0)
        )
    )

def petasce__assign__cht_1(canht):
    return np.maximum(np.full_like(canht, 0.001), canht)

def petasce__assign__kcmax_1():
    return 0.5

def petasce__condition__IF_1_4(meevp):
    return (meevp == "A")

def petasce__assign__kcmax_2(kcb):
    return np.maximum(np.full_like(kcb, 1.0), (kcb+0.05))

def petasce__assign__kcmax_3(wnd, rhmin, cht, kcb):
    return np.maximum(
        (1.2+(((0.04*(wnd-2.0))-(0.004*(rhmin-45.0)))*((cht/3.0)**0.3))),
        (kcb+0.05)
    )

def petasce__decision__kcmax_4(IF_1_4, kcmax_1, kcmax_2):
    return np.where(IF_1_4, kcmax_2, kcmax_1)

def petasce__condition__IF_2_2(meevp):
    return (meevp == "G")

def petasce__decision__kcmax_5(IF_2_2, kcmax_4, kcmax_3):
    return np.where(IF_2_2, kcmax_3, kcmax_4)

def petasce__condition__IF_1_5(kcb, kcbmin):
    return (kcb <= kcbmin)

def petasce__assign__fc_1():
    return 0.0

def petasce__assign__fc_2(kcb, kcbmin, kcmax, canht):
    return (((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+(0.5*canht)))

def petasce__decision__fc_3(IF_1_5, fc_2, fc_1):
    return np.where(IF_1_5, fc_1, fc_2)

def petasce__assign__fw_1():
    return 1.0

def petasce__assign__few_1(fc, fw):
    return np.minimum((1.0-fc), fw)

def petasce__assign__ke_1(kcmax, kcb, few):
    return np.maximum(
        np.full_like(kcmax, 0.0),
        np.minimum((1.0*(kcmax-kcb)), (few*kcmax))
    )

def petasce__assign__eo_1(kcb, ke, refet):
    return ((kcb+ke)*refet)

def petasce__assign__eo_2(eo):
    return np.maximum(eo, np.full_like(eo, 0.0001))
