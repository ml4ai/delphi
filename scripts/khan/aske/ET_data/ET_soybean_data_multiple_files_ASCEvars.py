import numpy as np
import pandas as pd
import glob
from pathlib import Path



data = list()

for weather_file in ['LUGO9001', 'LUGO9201', 'LUGO9301', 'LUGO9501', 'LUGO9601']:
    filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/' + weather_file + '.WTH')
    # for filename in Path(path).glob('/*.WTH'):
    for filename in filenames:
        # print(filename)
        
        data1 = pd.read_csv(filename, skiprows=4, header=0, delim_whitespace=True,
                error_bad_lines=False, warn_bad_lines=False)

        data1['DOY'] = data1.index + 1
        
        data2 = pd.read_csv(filename, skiprows=2, nrows = 1,
                delim_whitespace=True)
        cols = data2.columns
        # print(data2)
        # print(cols)
        data2 = data2.drop([cols[0], cols[-1]], axis=1)
        cols = cols[2:]
        data2.columns = cols
        # print(data2)
        # print(cols)

        data1['LAT'] = float(data2['LAT']) 
        data1['ELEV'] = float(data2['ELEV'])
        # data1['WNDHT'] = float(data2['WNDHT'])
        # wndht_lugo = float(data2['WNDHT'])
        # print(data1)    
        # data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DEWP', 'WIND', 'DOY', 'LAT', 'ELEV', 'WNDHT']])
        data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DEWP', 'WIND', 'DOY', 'LAT', 'ELEV']])
        del data1, data2
# print(data)    


for weather_file in ['LUGO9101', 'LUGO9801']:
    filenames = glob.glob('/Users/souratoshkhan/dssat-csm/Data/Weather/' + weather_file+ '.WTH')


    for filename in filenames:
        
        data1 = pd.read_csv(filename, skiprows=5, header=0, delim_whitespace=True,
                error_bad_lines=False, warn_bad_lines=False)
        # print(data1)
        data1['DOY'] = data1.index + 1
        
        data2 = pd.read_csv(filename, skiprows=3, nrows = 1,
                delim_whitespace=True)
        cols = data2.columns
        data2 = data2.drop([cols[0], cols[-1]], axis=1)
        cols = cols[2:]
        data2.columns = cols
        # print(data2)
        # print(cols)
        # print(data1.index)
            
        data1['LAT'] = float(data2['LAT']) 
        data1['ELEV'] = float(data2['ELEV'])
        # data1['WNDHT'] = float(data2['WNDHT'])
        # data1['WNDHT'] = wndht_lugo
        # data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DEWP', 'WIND', 'DOY', 'LAT', 'ELEV', 'WNDHT']])
        data.append(data1[['SRAD', 'TMAX', 'TMIN', 'DEWP', 'WIND', 'DOY', 'LAT', 'ELEV']])
        del data1, data2
# print(data)


big_frame2 = pd.concat(data, ignore_index=True)
print(big_frame2)



