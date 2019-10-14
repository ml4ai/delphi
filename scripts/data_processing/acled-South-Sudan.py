import pandas as pd
import matplotlib.pyplot as plt

filename = 'data/raw/data_for_november_2019_evaluation/south_sudan_data/Humanitarian Data Exchange/acled-data-2011-2018-SSudan.xlsx'

df = pd.read_excel(filename, header=1)
df.drop(df.index[0], inplace=True)
df['Month'] = df['event_date'].str.split('-').str.get(1)
df = df[['year', 'Month', 'event_type', 'country', 'admin1', 'source', 'fatalities']]
df.rename({'event_type':'Variable', 'admin1': 'State', 'fatalities':'Value'}, axis=1, inplace=True)
df.columns = df.columns.str.capitalize()

df = df[(df['Country'] == 'South Sudan') | (df['Country'] == 'Ethiopia')]
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Value'] = df['Value'].astype(int)
df1 = df.groupby(['Year', 'Month', 'State', 'Variable'])['Value'].sum()
df1 = df1.reset_index()



df1['Unit'], df1['County'] = None, None 

df1.to_csv('data/acled-data1.csv', index=False)


