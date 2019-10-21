import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

filenames = glob('data/raw/data_for_november_2019_evaluation/south_sudan_data/Humanitarian Data Exchange/201*.xlsx')

dfs = list()

for filename in filenames:
    df = pd.read_excel(filename)
    df.drop(df.index[0],inplace=True)
    dfs.append(df)

temp_df = pd.concat(dfs, ignore_index=True, sort=False)


dfs = list()

for col in df.columns[2:]:
    val = temp_df[col].values
    df1 = pd.DataFrame({'Variable':col, 'Value':val, 'Date': temp_df['Date'].values, 'Country':temp_df['Country'].values})
    dfs.append(df1)

big_frame = pd.concat(dfs, ignore_index=True, sort=False)
big_frame['Year'] = big_frame['Date'].str.split('-').str.get(0).astype(int)
big_frame['Month'] = big_frame['Date'].str.split('-').str.get(1).astype(int)

new_frame = big_frame.groupby(['Year', 'Month', 'Country', 'Variable'])['Value'].sum()
new_frame = new_frame.reset_index()


new_frame.dropna(subset=['Value'], inplace=True)

new_frame['Source'], new_frame['Unit'], new_frame['State'], big_frame['County'], big_frame['County'] = 'ACLED', None, None, None, None

new_frame.to_csv('data/acled-data3.csv', index=False)
