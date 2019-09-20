import numpy as np
import pandas as pd

df = pd.read_csv('Data for November 2019 Evaluation/South Sudan Data/World Bank/all_indicators_ssd.csv')
df.drop(df.index[0], inplace=True)
df.rename({'Country Name':'Country', 'Indicator Name':'Variable'}, axis=1, inplace=True)
df = df[['Year', 'Country', 'Variable', 'Value']]
df['Unit'] = np.where(df['Variable']== 'Population, total', 'total', '% of total')
print(df)
