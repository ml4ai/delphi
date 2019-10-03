from dreal import *

tmax = Variable("tmax")
tmin = Variable("tmin")
xhlai = Variable("xhlai")
msalb = Variable("msalb")
srad = Variable("srad")

xelev = Variable("xelev")
tdew = Variable("tdew")
doy = Variable("doy")
xlat = Variable("xlat")
windht = Variable("windht")
windrun = Variable("windrun")
meevp = Variable("meevp")
canht = Variable("canht")

tavg = Variable("tavg")
patm = Variable("patm")
psycon = Variable("psycon")
udelta = Variable("udelta")
emax = Variable("emax")
emin = Variable("emin")
es = Variable("es")
ea = Variable("ea")
rhmin = Variable("rhmin")
albedo = Variable("albedo")
rns = Variable("rns")
pie = Variable("pie")
dr = Variable("dr")
ldelta = Variable("ldelta")
ws = Variable("ws")
ra1 = Variable("ra1")
ra2 = Variable("ra2")
ra = Variable("ra")
rso = Variable("rso")
ratio = Variable("ratio")
fcd = Variable("fcd") 
tk4 = Variable("tk4")
rnl = Variable("rnl")
rn = Variable("rn")
g = Variable("g")
windsp = Variable("windsp")
wind2m  = Variable("wind2m")
cn = Variable("cn")
cd = Variable("cd")
refet = Variable("refet")
kcbmax = Variable("kcbmax")
kcbmin = Variable("kcbmin")
skc = Variable("skc")
kcb = Variable("kcb")
wnd = Variable("wnd")
cht = Variable("cht")
kcmax = Variable("kcmax")
fc = Variable("fc")
fw = Variable("fw")
few = Variable("few")
ke = Variable("ke")
kc = Variable("kc")

eo = Variable("eo")



tavg = (tmax + tmin)/2.0

patm = 101.3 * ((293.0 - 0.0065*xelev)/293.0)**5.26

psycon = 0.00066*patm


udelta = 2503.0*exp(17.27*tavg/(tavg+237.3))/(tavg+237.3)**2.0

emax = 0.6108*exp((17.27*tmax)/(tmax+237.3))
emin = 0.6108*exp((17.27*tmin)/(tmin+237.3))
es = (emax + emin) / 2.0

ea = 0.6108*exp((17.27*tdew)/(tdew+237.3))

rhmin = max(20.0, min(80.0, ea/emax*100.0))


if xhlai < 0:
    albedo = msalb
else:
    albedo = 0.23

rns = (1.0-albedo)*srad

pie = 4*atan(1)
dr = 1.0+0.033*cos(2.0*pie/365.0*doy)
ldelta = 0.409*sin(2.0*pie/365.0*doy-1.39)
ws = acos(-1.0*tan(xlat*pie/180.0)*tan(ldelta))
ra1 = ws*sin(xlat*pie/180.0)*sin(ldelta)
ra2 = cos(xlat*pie/180.0)*cos(ldelta)*sin(ws)
ra = 24.0/pie*4.92*dr*(ra1+ra2)


rso = (0.75+2*pow(10,-5)*xelev)*ra

ratio = srad/rso

if ratio < 0.3:
    ratio = 0.3
elif ratio > 1.0:
    ratio = 1.0


fcd = 1.35*ratio-0.35
tk4 = ((tmax+273.16)**4.0+(tmin+273.16)**4.0)/2.0
rnl = 4.901*pow(10,-9)*fcd*(0.34-0.14*sqrt(ea))*tk4

rn = rns - rnl

g = 0.0

windsp = windrun*1000.0 / 24.0 / 60.0 / 60.0
wind2m = windsp*(4.87/log(67.8*windht-5.42))

if meevp == 0:
    cn = 1600.0
    cd = 0.38
elif meevp == 1:
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

if meevp == 0:
    kcmax = max(1.0, kcb+0.05)
elif meevp == 1:
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

f_sat = And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin, tmin <= 23.9, 
0 <= msalb, msalb <=
1, 0.0 <= xhlai, xhlai <= 4.77, 2.45 <= srad, srad <= 27.8, 
0.0 <= tdew, tdew <= 36.7,
xelev == 10.0,
windrun == 400,
doy == 1,
canht == 1.0,
meevp == 1,
windht == 3.0,
xlat == 26.63,
0.0 <= eo, eo <= 10.0)

result = CheckSatisfiability(f_sat, 0.0001)
print(result)

