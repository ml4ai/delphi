import numpy as np
import pandas as pd

data = pd.read_csv('UFGA7801.WTH', skiprows=4, header=0,
        delim_whitespace=True, error_bad_lines=False)

df = data[['SRAD','TMAX', 'TMIN']]
#print(df.shape)
#df = df[~df.isna().any(axis=1)]
#print(df.shape)
#print(df.columns.values)


#df['SRAD'] = pd.to_numeric(df['SRAD'], errors='coerce')
#df['TMAX'] = pd.to_numeric(df['TMAX'], errors='coerce')
#df['TMIN'] = pd.to_numeric(df['TMIN'], errors='coerce')
#print(df.SRAD.values)


SRAD_max = df.SRAD.max()
SRAD_min = df.SRAD.min()

TMAX_max = df.TMAX.max()
TMAX_min = df.TMAX.min()

TMIN_max = df.TMIN.max()
TMIN_min = df.TMIN.min()

#ET_max = df.ETAA.max()
#ET_min = df.ETAA.min()

print(SRAD_max, SRAD_min, TMAX_max, TMAX_min, TMIN_max, TMIN_min
        )



