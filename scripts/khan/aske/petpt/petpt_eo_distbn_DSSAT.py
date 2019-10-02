import numpy as np
import pandas as pd
import glob
from pathlib import Path
import fitter
from scipy.stats import exponnorm, genlogistic
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

data = list()


filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/UFGA7801.WTH')
for filename in filenames:
    
    data1 = pd.read_csv(filename, skiprows=4, header=0, delim_whitespace=True,
            error_bad_lines=False, warn_bad_lines=False)

    data1['DOY'] = data1.index + 1
    

    data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DOY']])
    del data1
# print(data[0])    


df_param = pd.DataFrame(data[0])
# print(df_param)

data2 = pd.read_csv('PlantGro.OUT', skiprows=12, header=0,
        delim_whitespace=True, error_bad_lines=False)

df2 = data2[['LAID']]
df2 = df2[~df2.isna().any(axis=1)]


df1_LAI = df2[:122]
df_LAI = df1_LAI.append([df1_LAI, df1_LAI])
df_LAI['DOY'] = list(range(len(df_LAI)))
# print(df_LAI.shape[0])
# print(df_LAI)

param = pd.merge(df_LAI[1:], df_param, how='inner',on='DOY')
param['MSALB'] = 0.18
param['LAID'] = np.float64(param['LAID'])
param = param.set_index('DOY')
print(param.describe(include='all'))
# print(param)

# param.hist(bins=50, grid=False,figsize=(5,5))
# plt.show()

fig, ax = plt.subplots(5, figsize=(10,10))
sns.distplot(param['TMAX'], bins=50, kde=True, ax=ax[0], label='TMAX')
ax[0].legend()
sns.distplot(param['TMIN'], bins=50, kde=True, ax=ax[1], label='TMIN')
ax[1].legend()
sns.distplot(param['MSALB'], bins=50, kde=True, ax=ax[2], label='MSALB')
ax[2].legend()
sns.distplot(param['SRAD'], bins=50, kde=True, ax=ax[3], label='SRAD')
ax[3].legend()
sns.distplot(param['LAID'], bins=50, kde=True, ax=ax[4], label='XHLAI')
ax[4].legend()
plt.show()

f_tmax = fitter.Fitter(param.TMAX, timeout=10)
f_tmin = fitter.Fitter(param.TMIN, timeout=10)
# f_msalb = fitter.Fitter(param.MSALB, timeout=10)
f_srad = fitter.Fitter(param.SRAD, timeout=10)
f_xhlai = fitter.Fitter(param.LAID, timeout=10)

f_tmax.fit()
f_tmin.fit()
# f_msalb.fit()
f_srad.fit()
f_xhlai.fit()

tmax_best = f_tmax.get_best()
tmin_best = f_tmin.get_best()
srad_best = f_srad.get_best()
xhlai_best = f_xhlai.get_best()

print('tmax best',tmax_best)
print('tmin best',tmin_best)
print('srad best',srad_best)
print('xhlai best',xhlai_best)

tmax_val = list(tmax_best.values())[0]
tmin_val = list(tmin_best.values())[0]
# msalb_val = list(f_msalb.get_best().values())[0]
srad_val = list(srad_best.values())[0]
xhlai_val = list(xhlai_best.values())[0]

# print('tmax summary', f_tmax.summary())
# print('tmin_val', f_tmin.summary())
# print('msalb_val', f_msalb.summary())
# print('srad_val', f_srad.summary())
# print('xhlai_val', f_xhlai.summary())

print('tmax_val', tmax_val)
print('tmin_val', tmin_val)
# print('msalb_val', msalb_val)
print('srad_val', srad_val)
print('xhlai_val', xhlai_val)

def PETPT(srad, tmax, tmin, xhlai, msalb):

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

    return eo

# print(exponnorm.rvs(*tmax_val))
# print(exponnorm.rvs(*tmin_val))
# print(exponnorm.rvs(*msalb_val))
# print(exponnorm.rvs(*srad_val[:3]))

# y1 = PETPT( exponnorm.rvs(*srad_val), exponnorm.rvs(*tmax_val), exponnorm.rvs(*tmin_val),exponnorm.rvs(*xhlai_val), msalb_val=0.18)
# print(y1)


# niter = 100000

# eo = []
# for i in range(niter):
    # y1 = PETPT(exponnorm.rvs(*msalb_val), exponnorm.rvs(*srad_val), exponnorm.rvs(*tmax_val), exponnorm.rvs(*tmin_val), exponnorm.rvs(*xhlai_val))
    # eo.append(y1)
    
# fig = plt.figure()
# sns.distplot(eo, hist=True, kde=True, color='black', label='PETPT')
# plt.xlabel('eo values')
# plt.ylabel('Freq.')
# plt.legend()
# plt.show()




