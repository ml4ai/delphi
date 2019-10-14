import numpy as np
import pandas as pd
from glob import glob

df = pd.read_excel('data/raw/data_for_november_2019_evaluation/south_sudan_data/WHO/Misc_WHO_Health_Data_to_2016_SSudan_Ethiopia.xlsx', sheet_name='WHO NLIS Data Export')
df.dropna(how='all',axis=1, inplace=True)

df['Start Month'].fillna(0, inplace=True)
df['Ending Month'].fillna(0, inplace=True)
df['Start Month'] = df['Start Month'].astype(int).astype(str)
df['Ending Month'] = df['Ending Month'].astype(int).astype(str)
df['Start Month'].replace('0', '', inplace=True)
df['Ending Month'].replace('0', '', inplace=True)

df['Month'] = np.where(df['Ending Month'] == '', df['Start Month'], df['Start Month'] + '-' + df['Ending Month'])


df['Start Year'].fillna(0, inplace=True)
df['End Year'].fillna(0, inplace=True)
df['Start Year'] = df['Start Year'].astype(int).astype(str)
df['End Year'] = df['End Year'].astype(int).astype(str)
df['Start Year'].replace('0', '', inplace=True)
df['End Year'].replace('0', '', inplace=True)

df['Year'] = np.where(df['End Year'] == '', df['Start Year'], df['Start Year'] + '-' + df['End Year'])

df.rename({'Country Name':'Country', 'Region/Sample':'County'}, axis=1, inplace=True)
df1 = df[['Year', 'Month', 'Country', 'County']]
df2 = df.loc[:, 'Global Hunger Index (GHI)':'Life expectancy at birth (years)']

val = df2['Global Hunger Index (GHI)'].values
indicator = 'Global Hunger Index (GHI)'

big_frame = pd.DataFrame({'Value':val, 'Variable':indicator})
big_frame = pd.concat([big_frame, df1], axis=1, join='inner')


for col in df2.columns[1:]:
    val = df2[col].values
    indicator = col
    df_new = pd.DataFrame({'Value':val, 'Variable':indicator})
    df_new = pd.concat([df_new, df1], axis=1, join='inner')
    big_frame = pd.concat([big_frame, df_new], sort=False, ignore_index=True)

big_frame['Source'], big_frame['State'] = 'WHO', None

big_frame.dropna(subset=['Value'], inplace=True) 

big_frame['Unit'] = big_frame['Variable'].str.findall(r'(?<=\()[^(]*(?=\))').str[0]
big_frame['Unit'] = np.where(big_frame['Variable'] == 'Minimum acceptable diet (MAD) in children 6-23 months (%)', '%', big_frame['Unit'])
big_frame['Unit'] = np.where(big_frame['Variable'] == 'General government expenditure on health as % of total government expenditure', '%', big_frame['Unit'])
big_frame['Unit'] = np.where(big_frame['Variable'] == 'Total expenditure on health as % of gross domestic product', '%', big_frame['Unit'])
big_frame['Unit'] = np.where(big_frame['Variable'] == 'Population below minimum level of dietary energy requirement (undernourishment) (%)', '%', big_frame['Unit'])

big_frame['Variable'] = big_frame['Variable'].str.replace(r'\(.*?\)', '').str.strip()

big_frame = big_frame[~big_frame['Year'].str.contains('-')]
big_frame = big_frame[~big_frame['Month'].str.contains('-')]

big_frame['Country'] = big_frame['Country'].str.title()

big_frame.to_csv('data/WHO-data2.csv', index=False)
