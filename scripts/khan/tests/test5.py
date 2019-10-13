from dreal import *
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#sns.set_style('whitegrid')

#############################################

        ###   PETASCE MODEL    ###

############################################



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

eo_1 = Variable("eo_1")



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
eo_1 = (kcb + ke)*refet

eo_1 = max(eo_1, 0.0001)



#########################################################

        ####     PETPT MODEL   ####

########################################################


td = Variable("td")
albedo = Variable("albedo")
slang = Variable("slang")
eeq = Variable("eeq")
eo_2 = Variable("eo_2")
eo = Variable("eo")

count = Variable("count")
begin = Variable("begin")
end = Variable("end")
interval = Variable("interval")
step = Variable("step")

td = 0.6*tmax + 0.4*tmin

if xhlai < 0.0:
    albedo = msalb
else:
    albedo = 0.23 - (0.23 - msalb)*exp(-0.75*xhlai)

slang = srad*23.923
eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
eo_2 = eeq*1.1

if tmax > 35.0:
    eo_2 = eeq*((tmax-35.0)*0.05 + 1.1)
elif tmax < 5.0:
    eo_2 = eeq*0.01*exp(0.18*(tmax+20.0))

eo_2 = max(eo_2, 0.0001)

eo = (eo_2 - eo_1)**2

count = 0
begin = 16.1
end = 36.7
interval = 100 
step = (end - begin)/interval
tmax_lst1 = list()
tmin_lst1 = list()
msalb_lst1 = list()
xhlai_lst1 = list()
srad_lst1 = list()
tdew_lst1 = list()

print('Scanning over TMAX\n')

while count < interval:
    #print(count)
    #print(begin + step*count)
    #print(begin + step*(count+1))

    f_sat =  And(begin + step*count <= tmax, tmax <= begin + step*(count+1), 0.0 <= tmin, tmin <= 23.9, 
    #f_sat =  And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin, tmin <= 23.9, 
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
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
    #print(result)
    #print(result[0])
    
    tmax_lst1.append(result[0].lb())
    tmin_lst1.append(result[1].lb())
    xhlai_lst1.append(result[2].lb())
    msalb_lst1.append(result[3].lb())
    srad_lst1.append(result[4].lb())
    tdew_lst1.append(result[6].lb())

    count += 1
#    print('\n')


print('Scanning over TMIN\n')


count = 0
begin = 0.0
end = 23.9
interval = 100 
step = (end - begin)/interval
tmax_lst2 = list()
tmin_lst2 = list()
msalb_lst2 = list()
xhlai_lst2 = list()
srad_lst2 = list()
tdew_lst2 = list()

while count < interval:
    
    f_sat =  And(16.1 <= tmax, tmax <= 36.7, begin + step*count <= tmin, tmin <= begin + step*(count+1), 
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
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
#    print(result)
    
    tmax_lst2.append(result[0].lb())
    tmin_lst2.append(result[1].lb())
    xhlai_lst2.append(result[2].lb())
    msalb_lst2.append(result[3].lb())
    srad_lst2.append(result[4].lb())
    tdew_lst2.append(result[6].lb())

    count += 1
#    print('\n')


print('Scanning over MSALB\n')


count = 0
begin = 0.0
end = 1.0
interval = 100
step = (end - begin)/interval
tmax_lst3 = list()
tmin_lst3 = list()
msalb_lst3 = list()
xhlai_lst3 = list()
srad_lst3 = list()
tdew_lst3 = list()

while count < interval:
    
    f_sat =  And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin,
            tmin <= 23.9, 
    begin + step*count <= msalb, msalb <=
    begin + step*(count+1), 0.0 <= xhlai, xhlai <= 4.77, 2.45 <= srad, srad <= 27.8, 
    0.0 <= tdew, tdew <= 36.7,
    xelev == 10.0,
    windrun == 400,
    doy == 1,
    canht == 1.0,
    meevp == 1,
    windht == 3.0,
    xlat == 26.63,
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
#    print(result)
    
    tmax_lst3.append(result[0].lb())
    tmin_lst3.append(result[1].lb())
    xhlai_lst3.append(result[2].lb())
    msalb_lst3.append(result[3].lb())
    srad_lst3.append(result[4].lb())
    tdew_lst3.append(result[6].lb())

    count += 1
#    print('\n')


print('Scanning over XHLAI\n')

count = 0
begin = 0.0
end = 4.77
interval = 100 
step = (end - begin)/interval
tmax_lst4 = list()
tmin_lst4 = list()
msalb_lst4 = list()
xhlai_lst4 = list()
srad_lst4 = list()
tdew_lst4 = list()

while count < interval:
    
    f_sat =  And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin,
            tmin <= 23.9, 
    0.0 <= msalb, msalb <= 1.0,
    begin + step*count <= xhlai, xhlai <= begin + step*(count+1), 2.45 <= srad, srad <= 27.8, 
    0.0 <= tdew, tdew <= 36.7,
    xelev == 10.0,
    windrun == 400,
    doy == 1,
    canht == 1.0,
    meevp == 1,
    windht == 3.0,
    xlat == 26.63,
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
#    print(result)

    tmax_lst4.append(result[0].lb())
    tmin_lst4.append(result[1].lb())
    xhlai_lst4.append(result[2].lb())
    msalb_lst4.append(result[3].lb())
    srad_lst4.append(result[4].lb())
    tdew_lst4.append(result[6].lb())

    count += 1
#    print('\n')


print('Scanning over SRAD\n')

count = 0
begin = 2.45
end = 27.8
interval = 100
step = (end - begin)/interval
tmax_lst5 = list()
tmin_lst5 = list()
msalb_lst5 = list()
xhlai_lst5 = list()
srad_lst5 = list()
tdew_lst5 = list()

while count < interval:
    
    f_sat =  And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin,
            tmin <= 23.9, 
    0.0 <= msalb, msalb <= 1.0,
    0.0 <= xhlai, xhlai <= 4.77, begin + step*count <= srad, srad <= begin + step*(count+1), 
    0.0 <= tdew, tdew <= 36.7,
    xelev == 10.0,
    windrun == 400,
    doy == 1,
    canht == 1.0,
    meevp == 1,
    windht == 3.0,
    xlat == 26.63,
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
#    print(result)

    tmax_lst5.append(result[0].lb())
    tmin_lst5.append(result[1].lb())
    xhlai_lst5.append(result[2].lb())
    msalb_lst5.append(result[3].lb())
    srad_lst5.append(result[4].lb())
    tdew_lst5.append(result[6].lb())

    count += 1
#    print('\n')

print('Scanning over TDEW\n')

count = 0
begin = 0.0
end = 36.7
interval = 100
step = (end - begin)/interval
tmax_lst6 = list()
tmin_lst6 = list()
msalb_lst6 = list()
xhlai_lst6 = list()
srad_lst6 = list()
tdew_lst6 = list()

while count < interval:
    
    f_sat =  And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin,
            tmin <= 23.9, 
    0.0 <= msalb, msalb <= 1.0,
    0.0 <= xhlai, xhlai <= 4.77, 2.45 <= srad, srad <= 27.8, 
    begin + step*count <= tdew, tdew <= begin + step*(count+1),
    xelev == 10.0,
    windrun == 400,
    doy == 1,
    canht == 1.0,
    meevp == 1,
    windht == 3.0,
    xlat == 26.63,
    0.00 <= eo, eo <= 0.001)

    result = CheckSatisfiability(f_sat, 0.0001)
#    print(result)

    tmax_lst6.append(result[0].lb())
    tmin_lst6.append(result[1].lb())
    xhlai_lst6.append(result[2].lb())
    msalb_lst6.append(result[3].lb())
    srad_lst6.append(result[4].lb())
    tdew_lst6.append(result[6].lb())

    count += 1
#    print('\n')


#print(tmax_lst1, tmax_lst2, tmax_lst3, tmax_lst4, tmax_lst5)
#print(tmin_lst1, tmin_lst2, tmin_lst3, tmin_lst4, tmin_lst5)
#print(msalb_lst1, msalb_lst2, msalb_lst3, msalb_lst4, msalb_lst5)
#print(xhlai_lst1, xhlai_lst2, xhlai_lst3, xhlai_lst4, xhlai_lst5)
#print(srad_lst1, srad_lst2, srad_lst3, srad_lst4, srad_lst5)


# tmax_lst = tmax_lst1 + tmax_lst2 + tmax_lst3 + tmax_lst4 + tmax_lst5
# tmin_lst = tmin_lst1 + tmin_lst2 + tmin_lst3 + tmin_lst4 + tmin_lst5
# msalb_lst = msalb_lst1 + msalb_lst2 + msalb_lst3 + msalb_lst4 + msalb_lst5
# xhlai_lst = xhlai_lst1 + xhlai_lst2 + xhlai_lst3 + xhlai_lst4 + xhlai_lst5
# srad_lst = srad_lst1 + srad_lst2 + srad_lst3 + srad_lst4 + srad_lst5
# tdew_lst = tdew_lst1 + tdew_lst2 + tdew_lst3 + tdew_lst4 + tdew_lst5
tmax_lst = tmax_lst1 + tmax_lst2 + tmax_lst3 + tmax_lst4 + tmax_lst5 + tmax_lst6
tmin_lst = tmin_lst1 + tmin_lst2 + tmin_lst3 + tmin_lst4 + tmin_lst5 + tmin_lst6
msalb_lst = msalb_lst1 + msalb_lst2 + msalb_lst3 + msalb_lst4 + msalb_lst5 + msalb_lst6
xhlai_lst = xhlai_lst1 + xhlai_lst2 + xhlai_lst3 + xhlai_lst4 + xhlai_lst5 + xhlai_lst6
srad_lst = srad_lst1 + srad_lst2 + srad_lst3 + srad_lst4 + srad_lst5 + srad_lst6
tdew_lst = tdew_lst1 + tdew_lst2 + tdew_lst3 + tdew_lst4 + tdew_lst5 + tdew_lst6

for i in range(6):
    fig, axs = plt.subplots(nrows = 1, ncols = 6, sharex = True)
    ax = axs[0]
    ax.scatter(list(range(interval)), tmax_lst[i*interval:(i+1)*interval], color = 'r', label = 'TMAX', s =
            50)
    axs[i].axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
    ax.legend()
    ax = axs[1]
    ax.scatter(list(range(interval)), tmin_lst[i*interval:(i+1)*interval], color = 'b', label = 'TMIN', s =
            50)
    ax.legend()
    ax = axs[2]
    ax.scatter(list(range(interval)), msalb_lst[i*interval:(i+1)*interval], color = 'g', label = 'MSALB', s =
            50)
    ax.legend()
    ax = axs[3]
    ax.scatter(list(range(interval)), xhlai_lst[i*interval:(i+1)*interval], color = 'brown', label = 'XHLAI', s =
            50)
    ax.legend()
    ax = axs[4]
    ax.scatter(list(range(interval)), srad_lst[i*interval:(i+1)*interval], color = 'black', label = 'SRAD', s =
        50)
    ax.legend()
    ax = axs[5]
    ax.scatter(list(range(interval)), tdew_lst[i*interval:(i+1)*interval],
            color = 'purple', label = 'TDEW', s =
        50)
    ax.legend()
    plt.show()




# fig, axs = plt.subplots(nrows = 1, ncols = 5, sharex = True)
# ax = axs[0]
# ax.scatter(list(range(interval)), tmax_lst2, color = 'r', label = 'TMAX', s =
        # 50)
# ax.legend()
# ax = axs[1]
# ax.scatter(list(range(interval)), tmin_lst2, color = 'b', label = 'TMIN', s =
        # 50)
# ax.axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
# ax.legend()
# ax = axs[2]
# ax.scatter(list(range(interval)), msalb_lst2, color = 'g', label = 'MSALB', s =
        # 50)
# ax.legend()
# ax = axs[3]
# ax.scatter(list(range(interval)), xhlai_lst2, color = 'brown', label = 'XHLAI', s =
        # 50)
# ax.legend()
# ax = axs[4]
# ax.scatter(list(range(interval)), srad_lst2, color = 'black', label = 'SRAD', s =
        # 50)
# ax.legend()
# plt.show()



# fig, axs = plt.subplots(nrows = 1, ncols = 5, sharex = True)
# ax = axs[0]
# ax.scatter(list(range(interval)), tmax_lst3, color = 'r', label = 'TMAX', s =
        # 50)
# ax.legend()
# ax = axs[1]
# ax.scatter(list(range(interval)), tmin_lst3, color = 'b', label = 'TMIN', s =
        # 50)
# ax.legend()
# ax = axs[2]
# ax.scatter(list(range(interval)), msalb_lst3, color = 'g', label = 'MSALB', s =
        # 50)
# ax.axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
# ax.legend()
# ax = axs[3]
# ax.scatter(list(range(interval)), xhlai_lst3, color = 'brown', label = 'XHLAI', s =
        # 50)
# ax.legend()
# ax = axs[4]
# ax.scatter(list(range(interval)), srad_lst3, color = 'black', label = 'SRAD', s =
        # 50)
# ax.legend()
# plt.show()


# fig, axs = plt.subplots(nrows = 1, ncols = 5, sharex = True)
# ax = axs[0]
# ax.scatter(list(range(interval)), tmax_lst4, color = 'r', label = 'TMAX', s =
        # 50)
# ax.legend()
# ax = axs[1]
# ax.scatter(list(range(interval)), tmin_lst4, color = 'b', label = 'TMIN', s =
        # 50)
# ax.legend()
# ax = axs[2]
# ax.scatter(list(range(interval)), msalb_lst4, color = 'g', label = 'MSALB', s =
        # 50)
# ax.legend()
# ax = axs[3]
# ax.scatter(list(range(interval)), xhlai_lst4, color = 'brown', label = 'XHLAI', s =
        # 50)
# ax.axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
# ax.legend()
# ax = axs[4]
# ax.scatter(list(range(interval)), srad_lst4, color = 'black', label = 'SRAD', s =
        # 50)
# ax.legend()
# plt.show()



# fig, axs = plt.subplots(nrows = 1, ncols = 5, sharex = True)
# ax = axs[0]
# ax.scatter(list(range(interval)), tmax_lst5, color = 'r', label = 'TMAX', s =
        # 50)
# ax.legend()
# ax = axs[1]
# ax.scatter(list(range(interval)), tmin_lst5, color = 'b', label = 'TMIN', s =
        # 50)
# ax.legend()
# ax = axs[2]
# ax.scatter(list(range(interval)), msalb_lst5, color = 'g', label = 'MSALB', s =
        # 50)
# ax.legend()
# ax = axs[3]
# ax.scatter(list(range(interval)), xhlai_lst5, color = 'brown', label = 'XHLAI', s =
        # 50)
# ax.legend()
# ax = axs[4]
# ax.scatter(list(range(interval)), srad_lst5, color = 'black', label = 'SRAD', s =
        # 50)
# ax.axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
# ax.legend()
# #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
#                    wspace=0.35)
# plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(tmax_lst, msalb_lst, srad_lst, marker = 'o')
# ax.set_xlabel('TMAX')
# ax.set_ylabel('MSALB')
# ax.set_zlabel('SRAD')
#plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(list(range(5*interval)), tmax_lst, color = 'r',label = 'TMAX', s = 50)
# ax.set_xlabel('parameter set')
# ax.set_ylabel('tmax vals')
# ax.axvspan(0, interval, facecolor = 'green', alpha = 0.25)
# ax.axvspan(interval, 2*interval, facecolor = 'blue', alpha = 0.25)
# ax.axvspan(2*interval, 3*interval, facecolor = 'yellow', alpha = 0.25)
# ax.axvspan(3*interval, 4*interval, facecolor = 'brown', alpha = 0.25)
# ax.axvspan(4*interval, 5*interval, facecolor = 'orange', alpha = 0.25)
# plt.text(interval/2, 28, 'TMAX')
# plt.text(3*interval/2, 28, 'TMIN')
# plt.text(5*interval/2, 28, 'MSALB')
# plt.text(7*interval/2, 28, 'XHLAI')
# plt.text(9*interval/2, 28, 'SRAD')
# #ax.annotate('tuning TMAX', xy = (interval/2,16.5), xytext = (interval/2, 17), ha = 'center', va
# #        = 'bottom', arrowprops = dict(arrowstyle='-[, widthB=4.0, lengthB=0.5',
# #        lw = 2.0))
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(list(range(5*interval)), tmin_lst, color = 'r',label = 'TMIN', s = 50)
# ax.set_xlabel('parameter set')
# ax.set_ylabel('tmin vals')
# ax.axvspan(0, interval, facecolor = 'green', alpha = 0.25)
# ax.axvspan(interval, 2*interval, facecolor = 'blue', alpha = 0.25)
# ax.axvspan(2*interval, 3*interval, facecolor = 'yellow', alpha = 0.25)
# ax.axvspan(3*interval, 4*interval, facecolor = 'brown', alpha = 0.25)
# ax.axvspan(4*interval, 5*interval, facecolor = 'orange', alpha = 0.25)
# plt.text(interval/2, 20, 'TMAX')
# plt.text(3*interval/2, 20, 'TMIN')
# plt.text(5*interval/2, 20, 'MSALB')
# plt.text(7*interval/2, 20, 'XHLAI')
# plt.text(9*interval/2, 20, 'SRAD')
# #plt.legend()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(list(range(5*interval)), msalb_lst, color = 'r',label = 'MSALB',  s = 50)
# ax.set_xlabel('parameter set')
# ax.set_ylabel('msalb vals')
# ax.axvspan(0, interval, facecolor = 'green', alpha = 0.25)
# ax.axvspan(interval, 2*interval, facecolor = 'blue', alpha = 0.25)
# ax.axvspan(2*interval, 3*interval, facecolor = 'yellow', alpha = 0.25)
# ax.axvspan(3*interval, 4*interval, facecolor = 'brown', alpha = 0.25)
# ax.axvspan(4*interval, 5*interval, facecolor = 'orange', alpha = 0.25)
# plt.text(interval/2, 1.20, 'TMAX')
# plt.text(3*interval/2, 1.20, 'TMIN')
# plt.text(5*interval/2, 1.20, 'MSALB')
# plt.text(7*interval/2, 1.20, 'XHLAI')
# plt.text(9*interval/2, 1.20, 'SRAD')
# #plt.legend()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(list(range(5*interval)), xhlai_lst, color = 'r',label = 'XHLAI',  s = 50)
# ax.set_xlabel('parameter set')
# ax.set_ylabel('xhlai vals')
# ax.axvspan(0, interval, facecolor = 'green', alpha = 0.25)
# ax.axvspan(interval, 2*interval, facecolor = 'blue', alpha = 0.25)
# ax.axvspan(2*interval, 3*interval, facecolor = 'yellow', alpha = 0.25)
# ax.axvspan(3*interval, 4*interval, facecolor = 'brown', alpha = 0.25)
# ax.axvspan(4*interval, 5*interval, facecolor = 'orange', alpha = 0.25)
# plt.text(interval/2, 3.8, 'TMAX')
# plt.text(3*interval/2, 3.8, 'TMIN')
# plt.text(5*interval/2, 3.8, 'MSALB')
# plt.text(7*interval/2, 3.8, 'XHLAI')
# plt.text(9*interval/2, 3.8, 'SRAD')
# #plt.legend()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(list(range(5*interval)), srad_lst, color = 'r',label = 'SRAD',s = 50)
# ax.set_xlabel('parameter set')
# ax.set_ylabel('srad vals')
# ax.axvspan(0, interval, facecolor = 'green', alpha = 0.25)
# ax.axvspan(interval, 2*interval, facecolor = 'blue', alpha = 0.25)
# ax.axvspan(2*interval, 3*interval, facecolor = 'yellow', alpha = 0.25)
# ax.axvspan(3*interval, 4*interval, facecolor = 'brown', alpha = 0.25)
# ax.axvspan(4*interval, 5*interval, facecolor = 'orange', alpha = 0.25)
# plt.text(interval/2, 26, 'TMAX')
# plt.text(3*interval/2, 26, 'TMIN')
# plt.text(5*interval/2, 26, 'MSALB')
# plt.text(7*interval/2, 26, 'XHLAI')
# plt.text(9*interval/2, 26, 'SRAD')
# #plt.legend()
# plt.show()



#print(tmax_lst)
