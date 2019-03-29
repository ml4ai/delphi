import torch
import numpy as np


def petasce__assign__tavg_1(tmax, tmin):
    return ((tmax+tmin)/2.0)


def petasce__assign__patm_1(xelev):
    return (101.3*(((293.0-(0.0065*xelev))/293.0)**5.26))


def petasce__assign__psycon_1(patm):
    return (0.000665*patm)


def petasce__assign__udelta_1(tavg):
    return ((2503.0*torch.exp(((17.27*tavg)/(tavg+237.3))))/((tavg+237.3)**2.0))


def petasce__assign__emax_1(tmax):
    return (0.6108*torch.exp(((17.27*tmax)/(tmax+237.3))))


def petasce__assign__emin_1(tmin):
    return (0.6108*torch.exp(((17.27*tmin)/(tmin+237.3))))


def petasce__assign__es_1(emax, emin):
    return ((emax+emin)/2.0)


def petasce__assign__ea_1(tdew):
    return (0.6108*torch.exp(((17.27*tdew)/(tdew+237.3))))


def petasce__assign__rhmin_1(ea, emax):
    return torch.max(
        torch.full_like(ea, 20.0),
        torch.min(torch.full_like(ea, 80.0), ((ea/emax)*100.0))
    )


# def petasce__condition__IF_1_0(xhlai):
#     return (xhlai <= 0.0)
#
#
# def petasce__assign__albedo_1(msalb):
#     return msalb
#
#
# def petasce__assign__albedo_2():
#     return 0.23

# petasce__decision__albedo_3
def petasce__decision__albedo_1(xhlai, msalb):
    return torch.where(xhlai <= 0.0, msalb, torch.full_like(xhlai, 0.23))


def petasce__assign__rns_1(albedo, srad):
    return ((1.0-albedo)*srad)

# TODO: Figure out how to handle return of numeric with torch tensor
def petasce__assign__pie_1():
    return 3.14159265359


def petasce__assign__dr_1(pie, doy):
    # doy = doy.float()       # NOTE: Allows us to handle integer as a float
    return (1.0+(0.033*torch.cos((((2.0*pie)/365.0)*doy))))


def petasce__assign__ldelta_1(pie, doy):
    return (0.409*torch.sin(((((2.0*pie)/365.0)*doy)-1.39)))


def petasce__assign__ws_1(xlat, pie, ldelta):
    return torch.acos(
        -(((1.0*torch.tan(((xlat*pie)/180.0)))*torch.tan(ldelta)))
    )


def petasce__assign__ra1_1(ws, xlat, pie, ldelta):
    return ((ws*torch.sin(((xlat*pie)/180.0)))*torch.sin(ldelta))


def petasce__assign__ra2_1(xlat, pie, ldelta, ws):
    return ((torch.cos(((xlat*pie)/180.0))*torch.cos(ldelta))*torch.sin(ws))


def petasce__assign__ra_1(pie, dr, ra1, ra2):
    return ((((24.0/pie)*4.92)*dr)*(ra1+ra2))


def petasce__assign__rso_1(xelev, ra):
    return ((0.75+(2e-05*xelev))*ra)


def petasce__assign__ratio_1(srad, rso):
    return (srad/rso)


# def petasce__condition__IF_1_1(ratio):
#     return (ratio < 0.3)
#
#
# def petasce__assign__ratio_2():
#     return 0.3
#
#
# def petasce__assign__ratio_3():
#     return 1.0


# petasce__decision__ratio_4
def petasce__decision__ratio_2(ratio):
    return torch.where(ratio < 0.3, torch.full_like(ratio, 0.3), ratio)


# def petasce__condition__IF_2_0(ratio):
#     return (ratio > 1.0)


# petasce__decision__ratio_5
def petasce__decision__ratio_3(ratio_1, ratio_2):
    return torch.where(ratio_1 > 1.0, torch.full_like(ratio_1, 1.0), ratio_2)


def petasce__assign__fcd_1(ratio):
    return ((1.35*ratio)-0.35)


def petasce__assign__tk4_1(tmax, tmin):
    return ((((tmax+273.16)**4.0)+((tmin+273.16)**4.0))/2.0)


def petasce__assign__rnl_1(fcd, ea, tk4):
    return (((4.901e-09*fcd)*(0.34-(0.14*torch.sqrt(ea))))*tk4)


def petasce__assign__rn_1(rns, rnl):
    return (rns-rnl)


def petasce__assign__g_1():
    return 0.0


def petasce__assign__windsp_1(windrun):
    return ((((windrun*1000.0)/24.0)/60.0)/60.0)


def petasce__assign__wind2m_1(windsp, windht):
    return (windsp*(4.87/torch.log(((67.8*windht)-5.42))))


# def petasce__assign__cn_1():
#     return 0.0


# def petasce__assign__cd_1():
#     return 0.0


# def petasce__condition__IF_1_2(meevp):
#     return (meevp == "A")


# def petasce__assign__cn_2():
#     return 1600.0
#
#
# def petasce__assign__cd_2():
#     return 0.38


# def petasce__assign__cn_3():
#     return 900.0


# def petasce__assign__cd_3():
#     return 0.34


# petasce__decision__cd_4
def petasce__decision__cd_1(meevp):
    return torch.tensor(np.where(meevp == "A", 0.38, 0.0))


# def petasce__condition__IF_2_1(meevp):
#     return (meevp == "G")

# petasce__decision__cd_5
def petasce__decision__cd_2(meevp, cd_1):
    return torch.tensor(np.where(meevp == "G", 0.34, cd_1))


# petasce__decision__cn_4
def petasce__decision__cn_1(meevp):
    return torch.tensor(np.where(meevp == "A", 1600.0, 0.0))


# petasce__decision__cn_5
def petasce__decision__cn_2(meevp, cn_1):
    return torch.tensor(np.where(meevp == "G", 900.0, cn_1))


def petasce__assign__refet_1(udelta, rn, g, psycon, cn, tavg, wind2m, es, ea):
    return (((0.408*udelta)*(rn-g))+(((psycon*(cn/(tavg+273.0)))*wind2m)*(es-ea)))


def petasce__assign__refet_2(refet, udelta, psycon, cd, wind2m):
    return (refet/(udelta+(psycon*(1.0+(cd*wind2m)))))


def petasce__assign__refet_3(refet):
    return torch.max(torch.full_like(refet, 0.0001), refet)


def petasce__assign__skc_1():
    return 0.8


def petasce__assign__kcbmin_1():
    return 0.3


def petasce__assign__kcbmax_1():
    return 1.2


# def petasce__condition__IF_1_3(xhlai):
#     return (xhlai <= 0.0)
#
#
# def petasce__assign__kcb_1():
#     return 0.0
#
#
# def petasce__assign__kcb_2(kcbmin, kcbmax, skc, xhlai):
#     return torch.max(
#         torch.full_like(kcbmin, 0.0),
#         (kcbmin+((kcbmax-kcbmin)*(1.0-torch.exp(-(((1.0*skc)*xhlai))))))
#     )

# petasce__decision__kcb_3
def petasce__decision__kcb_1(xhlai, kcbmin, kcbmax, skc):
    return torch.where(
        xhlai <= 0.0,
        torch.full_like(xhlai, 0.0),
        torch.max(
            torch.full_like(kcbmin, 0.0),
            (kcbmin+((kcbmax-kcbmin)*(1.0-torch.exp(-(((1.0*skc)*xhlai))))))
        )
    )


def petasce__assign__wnd_1(wind2m):
    return torch.max(
        torch.full_like(wind2m, 1.0),
        torch.min(wind2m, torch.full_like(wind2m, 6.0))
    )


def petasce__assign__cht_1(canht):
    return torch.max(torch.full_like(canht, 0.001), canht)


# def petasce__assign__kcmax_1():
#     return 0.0
#
#
# def petasce__condition__IF_1_4(meevp):
#     return (meevp == "A")
#
#
# def petasce__assign__kcmax_2(kcb):
#     return torch.max(torch.full_like(kcb, 1.0), (kcb+0.05))
#
#
# def petasce__assign__kcmax_3(wnd, rhmin, cht, kcb):
#     return torch.max((1.2+(((0.04*(wnd-2.0))-(0.004*(rhmin-45.0)))*((cht/3.0)**0.3))), (kcb+0.05))

# petasce__decision__kcmax_4
def petasce__decision__kcmax_1(meevp, kcb):
    return torch.tensor(np.where(
        meevp == "A",
        0.0,
        torch.max(torch.full_like(kcb, 1.0), (kcb+0.05)).numpy()
    ))


# def petasce__condition__IF_2_2(meevp):
#     return (meevp == "G")

# petasce__decision__kcmax_5
def petasce__decision__kcmax_2(meevp, kcmax_1, wnd, rhmin, cht, kcb):
    return torch.tensor(np.where(
        meevp == "G",
        torch.max(
            (1.2+(((0.04*(wnd-2.0))-(0.004*(rhmin-45.0)))*((cht/3.0)**0.3))),
            (kcb+0.05)
        ).numpy(),
        kcmax_1
    ))


# def petasce__condition__IF_1_5(kcb, kcbmin):
#     return (kcb <= kcbmin)
#
#
# def petasce__assign__fc_1():
#     return 0.0
#
#
# def petasce__assign__fc_2(kcb, kcbmin, kcmax, canht):
#     return (((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+(0.5*canht)))

# petasce__decision__fc_3
def petasce__decision__fc_1(kcb, kcbmin, kcmax, canht):
    return torch.where(
        kcb <= kcbmin,
        torch.full_like(kcb, 0.0),
        (((kcb-kcbmin)/(kcmax-kcbmin))**(1.0+(0.5*canht)))
    )


def petasce__assign__fw_1():
    return 1.0


def petasce__assign__few_1(fc, fw):
    return torch.min((1.0-fc), fw)


def petasce__assign__ke_1(kcmax, kcb, few):
    return torch.max(
        torch.full_like(kcmax, 0.0),
        torch.min((1.0*(kcmax-kcb)), (few*kcmax))
    )


def petasce__assign__eo_0(kcb, ke, refet):
    return ((kcb+ke)*refet)


def petasce__assign__eo_1(eo):
    return torch.max(eo, torch.full_like(eo, 0.0001))
