import numpy as np
from math import *
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

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
    # elif tmax < 5.0:
        # eo = eeq*0.01*np.exp(0.18*(tmax+20.0))

    eo = max(eo, 0.0001)

    #print(msalb, srad, tmax, tmin, xhlai)
    return eo



size = 10

#Parameters for PETPT and PETASSCE
tmax = np.linspace(16.1, 36.7, size)  #UFGA7801.WTH and ET.OUT
tmin = np.linspace(0.0, 23.9, size)   #UFGA7801.WTH and ET.OUT
msalb = np.linspace(0, 1, size)
srad = np.linspace(2.45, 27.8, size)   #UFGA7801.WTH and ET.OUT
xhlai = np.linspace(0.00, 4.77, size)  #PlantGro.OUT file

#Preset Values of Variables in PETPT and PETASCE
# msalb = 0.18 # SOIL.SOL file

#Parameters for only PETASCE
tdew = 16.0
#tdew = np.linspace(0.0, 36.7, size)  # Taken from tmax and tmin

windrun = 400
#windrun = np.linspace(0, 900, size) 

doy = 180
#doy = np.linspace(1, 365, 365)

canht = 1.0
#canht = np.linspace(0, 3, size)

#Preset Values of Variables in PETASCE
meevp = 'G'    #Soybean 
windht = 3.00                   #PlantGro.OUT
xlat = 26.63                    #PlantGro.OUT
xelev = 10                      #PlantGro.OUT

#print("PETPT Output values :",PETPT(msalb, srad, tmax, tmin, xhlai))
#print("PETASCE Output values:", PETASCE(canht, doy, msalb, meevp, srad,
#            tdew, tmax, tmin, windht, windrun, xhlai, xlat, xelev))

# vfunc = np.vectorize(PETPT)
#y1 = [PETPT(msalb, *args) for args in product(srad, tmax, tmin, xhlai)]
y1 = [PETPT(*args) for args in product(msalb, srad, tmax, tmin, xhlai)]
#y2 = [PETASCE(msalb, *args, canht, doy, meevp, tdew, windht, windrun, xlat, xelev) for args in product(srad, tmax, tmin, xhlai)]
x = np.arange(len(y1))
print(len(y1))
print(max(y1))
print(min(y1))

fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax1.scatter(x+1, np.log(y1), label = 'PETPT', c = 'r')
#ax1.scatter(x+1, np.log(y2), label = 'PETASCE', c = 'b')
ax1.scatter(x+1, y1, label = 'PETPT', c = 'r')
plt.xlabel('Parameter set')
plt.ylabel(r'EO ($\times 10^4$)')
#plt.ylabel(r'Log (EO ($\times 10^4$))')
#plt.xticks(x+1)
#for i_x, i_y in zip(x+1, y1):
#    plt.text(i_x, i_y, '({},{})'.format(i_x, round(i_y,2)))
#for i_x, i_y in zip(x+1, y2):
#    plt.text(i_x, i_y, '({},{})'.format(i_x, round(i_y,2)))
plt.legend()
plt.show()




