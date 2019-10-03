import numpy as np
import pandas as pd
import glob
from pathlib import Path



data = list()


filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/UFBG*.WTH')
# for filename in Path(path).glob('/*.WTH'):
for filename in filenames:
    # print(filename)
    
    data1 = pd.read_csv(filename, skiprows=4, header=0, delim_whitespace=True,
            error_bad_lines=False, warn_bad_lines=False)

    data1['DOY'] = data1.index + 1
    
    data2 = pd.read_csv(filename, skiprows=2, nrows = 1,
            delim_whitespace=True)
    cols = data2.columns
    data2 = data2.drop([cols[0], cols[-1]], axis=1)
    cols = cols[2:]
    data2.columns = cols

    data1['LAT'] = float(data2['LAT']) 
    data1['ELEV'] = float(data2['ELEV'])
    data1['WNDHT'] = float(data2['WNDHT'])
    # print(cols)    
    data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DOY', 'LAT', 'ELEV', 'WNDHT']])
    del data1, data2
# print(data2)    

filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/UFCI*.WTH')


for filename in filenames:
    
    data1 = pd.read_csv(filename, skiprows=4, header=0, delim_whitespace=True,
            error_bad_lines=False, warn_bad_lines=False)

    data1['DOY'] = data1.index + 1
    
    data2 = pd.read_csv(filename, skiprows=2, nrows = 1,
            delim_whitespace=True)
    cols = data2.columns
    data2 = data2.drop([cols[0], cols[-1]], axis=1)
    cols = cols[2:]
    data2.columns = cols

    data1['LAT'] = float(data2['LAT']) 
    data1['ELEV'] = float(data2['ELEV'])
    data1['WNDHT'] = float(data2['WNDHT'])
    data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DOY', 'LAT', 'ELEV', 'WNDHT']])
    del data1, data2


filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/UFGA7801.WTH')


for filename in filenames:
    
    data1 = pd.read_csv(filename, skiprows=4, header=0, delim_whitespace=True,
            error_bad_lines=False, warn_bad_lines=False)

    data1['DOY'] = data1.index + 1
    
    data2 = pd.read_csv(filename, skiprows=2, nrows = 1,
            delim_whitespace=True)
    cols = data2.columns
    data2 = data2.drop([cols[0], cols[-1]], axis=1)
    cols = cols[2:]
    data2.columns = cols

    data1['LAT'] = float(data2['LAT']) 
    data1['ELEV'] = float(data2['ELEV'])
    data1['WNDHT'] = float(data2['WNDHT'])
    data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DOY', 'LAT', 'ELEV', 'WNDHT']])
    del data1, data2



# filenames = glob.glob('PlantGro.OUT')

# for filename in filenames:
    # data1 = pd.read_csv(filename, skiprows=12, header=0,
         # delim_whitespace=True, error_bad_lines=False)

    # df0 = data1[['LAID']]
    # df0 = df0[~df0.isna().any(axis=1)]

    # df0['LAID'] = pd.to_numeric(df0['LAID'], errors='coerce')
    # df1 = df0[:127]
    # df2 = df0[134:271]
    # df3 = df0[258:389]
    # df4 = df0[395:]
    
# big_frame1 = pd.concat([df1,df2,df3,df4], ignore_index=True, sort=True)
# # print(big_frame1)


big_frame2 = pd.concat(data, ignore_index = True, sort=True)
print(big_frame2)

# big_frame = pd.concat([big_frame1, big_frame2])
# print(big_frame)


