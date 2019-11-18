import pandas as pd
import matplotlib.pyplot as plt

filename = 'data/raw/data_for_november_2019_evaluation/ethiopia_east_africa_data/Armed Conflict Location & Event Data Project - ACLED/acled-ethiopia-data.csv'

df = pd.read_csv(filename)
df.drop(df.index[0], inplace=True)
df['Month'] = df['event_date'].str.split('-').str.get(1)
df = df[['year', 'Month', 'event_type', 'country', 'admin1', 'source', 'fatalities']]
df.rename({'event_type':'Variable', 'admin1': 'State', 'fatalities':'Value'}, axis=1, inplace=True)
df.columns = df.columns.str.capitalize()


df = df[(df['Country'] == 'Ethiopia') | (df['Country'] == 'South Sudan')]
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Value'] = df['Value'].astype(int)
df1 = df.groupby(['Year', 'Month', 'State', 'Variable'])['Value'].sum()
df1 = df1.reset_index()
df1['Unit'], df1['County'] = None, None

df1.to_csv('data/acled-data2.csv', index=False)


