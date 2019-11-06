import numpy as np
from math import *
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm
sns.set_style('white')

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
    elif tmax < 5.0:
        eo = eeq*0.01*np.exp(0.18*(tmax+20.0))

    eo = max(eo, 0.0001)

    return eo

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

size = 10

#Parameters for PETPT and PETASSCE
tmax = get_truncated_normal(26.4, 2, 16.1, 36.7)
tmin = get_truncated_normal(11.95, 2, 0.0, 23.9)
msalb = get_truncated_normal(0.5, 0.2, 0.0, 1.0)
srad = get_truncated_normal(15.125, 2, 2.45, 27.8)
xhlai = get_truncated_normal(2.385, 0.8, 0.0, 4.77)

samples = 10000

tmax_val = tmax.rvs(samples)
tmin_val = tmin.rvs(samples)
msalb_val = msalb.rvs(samples)
srad_val = srad.rvs(samples)
xhlai_val = xhlai.rvs(samples)

fig, ax = plt.subplots(5, figsize=(10,10))
sns.distplot(tmax_val, kde=True, ax=ax[0], label='TMAX')
ax[0].legend()
sns.distplot(tmin_val, kde=True, ax=ax[1], label='TMIN')
ax[1].legend()
sns.distplot(msalb_val, kde=True, ax=ax[2], label='MSALB')
ax[2].legend()
sns.distplot(srad_val, kde=True, ax=ax[3], label='SRAD')
ax[3].legend()
sns.distplot(xhlai_val, kde=True, ax=ax[4], label='XHLAI')
ax[4].legend()
#ax[0].hist(tmax.rvs(samples), density=True, label='TMAX')
# ax[0].legend()
# ax[1].hist(tmin.rvs(samples), density=True, label='TMIN')
# ax[1].legend()
# ax[2].hist(msalb.rvs(samples), density=True, label='MSALB')
# ax[2].legend()
# ax[3].hist(srad.rvs(samples), density=True, label='SRAD')
# ax[3].legend()
# ax[4].hist(xhlai.rvs(samples), density=True, label='XHLAI')
# ax[4].legend()
plt.show()

eo_mean, eo_sq_mean, eo_var = [], [], []

sample_size = 100
niter = 10000

for j in range(sample_size):
    eo = []
    eo_sq = []
    for i in range(niter):
        y1 = PETPT(msalb.rvs(), srad.rvs(), tmax.rvs(), tmin.rvs(), xhlai.rvs())
        eo.append(y1)
        eo_sq.append(y1**2)
    eo_mean.append(sum(eo)/niter)
    eo_sq_mean.append(sum(eo_sq)/niter)
    eo_var.append(sum(eo_sq)/niter - (sum(eo)/niter)**2)
    
    
fig, ax = plt.subplots(2)
# plt.hist(eo, density=True, label='PETPT')
sns.distplot(eo_mean, hist=True, kde=True, color='black', label='eo (PETPT) mean', ax=ax[0])
ax[0].legend()
sns.distplot(eo_var, hist=True, kde=True, color='black', label='eo (PETPT) var', ax=ax[1])
ax[1].legend()
ax[1].set_xlabel('eo values')
ax[0].set_ylabel('Freq.')
ax[1].set_ylabel('Freq.')
plt.show()




