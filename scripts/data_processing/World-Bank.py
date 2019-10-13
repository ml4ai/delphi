import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/data_for_november_2019_evaluation/south_sudan_data/World Bank/all_indicators_ssd.csv')
df.drop(df.index[0], inplace=True)
df.rename({'Country Name':'Country', 'Indicator Name':'Variable'}, axis=1, inplace=True)
df = df[['Year', 'Country', 'Variable', 'Value']]
df['Unit'] = df['Variable'].str.findall(r'(?<=\()[^(]*(?=\))').str[0]

df['Variable'] = df['Variable'].str.replace(r'\(.*?\)', '').str.strip()

df['Source'], df['State'], df['County'], df['Month'] = 'World Bank', None, None, None

df.to_csv('data/World-Bank-data.csv', index=False)
