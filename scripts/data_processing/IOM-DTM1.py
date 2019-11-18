import pandas as pd
import datetime

dir_path = 'data/raw/data_for_november_2019_evaluation/south_sudan_data/IOM DTM/'
filename = '20180202 BMR Locations IDP Estimates.xlsx'

df = pd.read_excel(dir_path + filename, header=None, sheet_name='IDP Estimates')
df.drop(df.index[1], inplace=True)
df.columns = df.iloc[0]
df.drop(df.index[0], inplace=True)

df.drop(columns=['Registered Population', 'Lat', 'Long', 'IDP Component'], axis=1, inplace=True)

colnames = df.columns

dfs = list()

for index, row in df.iterrows():
    values = row[df.columns[2]:df.columns[-1]].values
    df_new = pd.DataFrame({'Value': values, 'State':row['State'], 'County': row['County']})
    df_new['Date'] =  df.columns[2:]
    dfs.append(df_new)

big_frame = pd.concat(dfs, sort=False, ignore_index=True)
big_frame['Year'] = pd.DatetimeIndex(big_frame['Date']).year
big_frame['Month'] = pd.DatetimeIndex(big_frame['Date']).month
big_frame.drop('Date', axis=1, inplace=True)
big_frame.dropna(how='any', axis=0, inplace=True)

big_frame['Country'] = 'South Sudan'
big_frame ['Unit'] = None
big_frame['Variable'] =  'IDP (Internally Displaced People)' 
big_frame['Source'] = 'IOM-DTM'

big_frame.to_csv('data/IOM-DTM-data1.csv', index=False)

