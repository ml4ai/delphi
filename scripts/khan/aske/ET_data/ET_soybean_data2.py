import numpy as np
import pandas as pd

data = pd.read_csv('PlantGro.OUT', skiprows=12, header=0,
        delim_whitespace=True, error_bad_lines=False)

df = data[['LAID']]
df = df[~df.isna().any(axis=1)]

#df['DAS'] = pd.to_numeric(df['DAS'], errors='coerce')
df['LAID'] = pd.to_numeric(df['LAID'], errors='coerce')


df1 = df[:127]
df2 = df[134:271]
df3 = df[258:389]
df4 = df[395:]
#print(df.columns.values)

LAI_max1 = df1['LAID'].max()
LAI_max2 = df2['LAID'].max()
LAI_max3 = df3['LAID'].max()
LAI_max4 = df4['LAID'].max()

LAI_min1 = df1['LAID'].min()
LAI_min2 = df2['LAID'].min()
LAI_min3 = df3['LAID'].min()
LAI_min4 = df4['LAID'].min()

LAI_max = np.array([LAI_max1, LAI_max2, LAI_max3, LAI_max4])
LAI_max_avg = np.average(LAI_max)


LAI_min = np.array([LAI_min1, LAI_min2, LAI_min3, LAI_min4])
LAI_min_avg = np.average(LAI_min)
#df1['DAS'] = pd.to_numeric(df1['DAS'], errors='coerce')
#df1['LAID'] = pd.to_numeric(df1['LAID'], errors='coerce')

print(LAI_max_avg, LAI_min_avg)
#print(df4)


