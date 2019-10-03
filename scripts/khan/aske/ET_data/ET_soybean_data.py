import numpy as np
import pandas as pd

data = pd.read_csv('ET.OUT', skiprows=12, header=0,
        delim_whitespace=True, error_bad_lines=False)


df = data[['SRAA','TMAXA', 'TMINA', 'ETAA']]
df = df[~df.isna().any(axis=1)]
#print(df.columns.values)

df['SRAA'] = pd.to_numeric(df['SRAA'], errors='coerce')
df['TMAXA'] = pd.to_numeric(df['TMAXA'], errors='coerce')
df['TMINA'] = pd.to_numeric(df['TMINA'], errors='coerce')
df['ETAA'] = pd.to_numeric(df['ETAA'], errors='coerce')
#print(df.SRAA.values)


SRAD_max1 = df.SRAA.max()
SRAD_min1 = df.SRAA.min()

TMAX_max1 = df.TMAXA.max()
TMAX_min1 = df.TMAXA.min()

TMIN_max1 = df.TMINA.max()
TMIN_min1 = df.TMINA.min()

ET_max1 = df.ETAA.max()
ET_min1 = df.ETAA.min()

print(SRAD_max1, SRAD_min1, TMAX_max1, TMAX_min1, TMIN_max1, TMIN_min1, ET_max1,
        ET_min1)

data1 = pd.read_csv('UFGA7801.WTH', skiprows=4, header=0,
        delim_whitespace=True, error_bad_lines=False)

df1 = data1[['SRAD','TMAX', 'TMIN']]
print(df1.shape)


SRAD_max2 = df1.SRAD.max()
SRAD_min2 = df1.SRAD.min()

TMAX_max2 = df1.TMAX.max()
TMAX_min2 = df1.TMAX.min()

TMIN_max2 = df1.TMIN.max()
TMIN_min2 = df1.TMIN.min()

#ET_max = df.ETAA.max()
#ET_min = df.ETAA.min()

print(SRAD_max2, SRAD_min2, TMAX_max2, TMAX_min2, TMIN_max2, TMIN_min2)


SRAD_max = np.array([SRAD_max1, SRAD_max2])
srad_max = np.average(SRAD_max)
SRAD_min = np.array([SRAD_min1, SRAD_min2])
srad_min = np.average(SRAD_min)
print(srad_max, srad_min)

TMAX_max = np.array([TMAX_max1, TMAX_max2])
tmax_max = np.average(TMAX_max)
TMAX_min = np.array([TMAX_min1, TMAX_min2])
tmax_min = np.average(TMAX_min)
print(tmax_max, tmax_min)

TMIN_max = np.array([TMIN_max1, TMIN_max2])
tmin_max = np.average(TMIN_max)
TMIN_min = np.array([TMIN_min1, TMIN_min2])
tmin_min = np.average(TMIN_min)
print(tmin_max, tmin_min)
