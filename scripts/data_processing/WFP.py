import pandas as pd
from glob import glob

dfs = list()

filenames = glob('data/raw/data_for_november_2019_evaluation/south_sudan_data/WFP/wfp*.xlsx')

for filename in filenames:
    df = pd.read_excel(filename)
    df.dropna(how='all', axis=1, inplace=True)
    df.drop(df.index[0], inplace=True)
    df.rename({'cmname':'Variable', 'price':'Value', 'unit':'Unit', 'country':'Country'}, axis=1, inplace=True)
    df['Year'] = df['date'].str.split('-').str.get(0)
    df['Month'] = df['date'].str.split('-').str.get(1)

    df = df[['Year', 'Month', 'Variable', 'Value', 'Unit', 'Country']]


    variables = df['Variable'].unique()

    for var in variables:
        df_temp = df[df['Variable']==var]
        dfs.append(df_temp)

big_frame = pd.concat(dfs, sort=False, ignore_index=True)

big_frame['Source'],  big_frame['State'], big_frame['County'] = 'WFP', None, None

big_frame.to_csv('data/WFP-data.csv', index=False)
