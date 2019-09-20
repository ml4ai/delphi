import pandas as pd
from glob import glob

dfs = list()

filenames = glob('Data for November 2019 Evaluation/South Sudan Data/WFP/wfp*.xlsx')

for filename in filenames:
    df = pd.read_excel(filename)
    # df = pd.read_excel('Data for November 2019 Evaluation/South Sudan Data/WFP/wfp_food_prices_south-sudan-2006-2018.xlsx')
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
print(big_frame)
