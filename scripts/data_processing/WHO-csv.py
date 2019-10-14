import sys
import pandas as pd
from glob import glob
import numpy as np

dfs = list()

directoryPath = 'data/raw/data_for_november_2019_evaluation/south_sudan_data/WHO/'
filenames = glob(directoryPath + '*-SSudan.csv')
colnames = ['GHO (DISPLAY)', 'YEAR (DISPLAY)', 'REGION (DISPLAY)', 'Numeric']
for filename in filenames:
    df = pd.read_csv(filename, encoding='latin-1',usecols=lambda x: x in colnames)
    dfs.append(df)
    
big_frame = pd.concat(dfs, ignore_index=True, sort=True)
for col in colnames:
    if len(col.split()) > 1:
        big_frame.rename({col:col.split()[0]}, axis=1, inplace=True)

big_frame.rename({'GHO':'Variable', 'Numeric':'Value', 'REGION':'Country', 'YEAR':'Year'}, axis=1, inplace=True)




filename = 'data/raw/data_for_november_2019_evaluation/south_sudan_data/WHO/South Sudan WHO Statistics Summary.csv'
df = pd.read_csv(filename, index_col=False)
for col in df.columns:
    if col == 'Unnamed: 0':
        df.rename({col:'Country'}, axis=1, inplace=True)
    if len(col.split('.')) == 2:
        df.rename({col:col.split('.')[0]}, axis=1, inplace=True)




variables = df.iloc[0].values
values = variables[1:]
df_new1 = pd.DataFrame({'Year': values, 'Country':df.columns[1:]})


for i in range(1,df.shape[0]):
    variables = df.iloc[i].values
    indicator = variables[0]
    values = variables[1:]
    df_new2 = pd.DataFrame({'Value':values, 'Variable':indicator})
    df_new = pd.concat([df_new2, df_new1], axis=1, join='inner')
    big_frame = pd.concat([big_frame, df_new], sort=False, ignore_index=True)

big_frame = big_frame[big_frame['Country']!='Africa']
big_frame['Source'], big_frame['Month'], big_frame['County'], big_frame['State'] = 'WHO', None, None, None

big_frame.dropna(subset=['Value'], inplace=True)
big_frame = big_frame[big_frame['Value'] != 'No data']
big_frame['Value'] = big_frame['Value'].astype(str)
big_frame['Value'] = big_frame['Value'].str.split('[').str[0]
big_frame['Value'] = big_frame['Value'].str.split().str.get(0)


big_frame['Unit'] = np.where(big_frame['Variable'] == 'Neonates protected at birth against neonatal tetanus (PAB) (%)', '%',big_frame['Variable'].str.findall(r'(?<=\()[^(]*(?=\))').str[0])
big_frame['Variable'] = big_frame['Variable'].str.replace(r'\(.*?\)', '').str.strip()

big_frame.to_csv('data/WHO-data1.csv', index=False)




