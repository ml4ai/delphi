import pandas as pd
from glob import glob
import numpy as np

excel_files = ['Life expectancy-Ethiopia-SSudan.xls', 'child_malnutrition-Ethiopia-SSudan.xlsx']

dfs = list()

for filename in excel_files:
    df = pd.read_excel('data/raw/data_for_november_2019_evaluation/south_sudan_data/WHO/'+ filename)


    col1 = df.iloc[0, 0]
    col2 = df.iloc[0,1]

    df.drop(df.index[0], inplace=True)

    df[col1] = df.iloc[0:,0]
    df[col2] = df.iloc[0:,1]

    df.drop(df.columns[[0,1]], axis=1, inplace=True)
    val = df.iloc[:,0].values
    variable = df.columns[0]

    big_frame = pd.DataFrame({'Value':val, 'Variable':variable, col1:df[col1].values, col2:df[col2].values})


    for col in df.columns[1:-2]:
        val = df[col].values
        variable = col
        df2 = pd.DataFrame({'Value':val,'Variable':variable, col1:df[col1].values, col2:df[col2].values})
        big_frame = pd.concat([big_frame, df2], sort=False, ignore_index=True)

    big_frame['Source'] = 'WHO'
    big_frame['County'], big_frame['Month'], big_frame['State'] = None, None, None
    big_frame['Unit'] = big_frame['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')] )
    dfs.append(big_frame)

big_frame = pd.concat(dfs, sort=False, ignore_index=True)

big_frame.dropna(subset=['Value'], inplace=True)

big_frame['Unit'] = big_frame['Variable'].str.findall(r'(?<=\()[^(]*(?=\))').str[0]
big_frame['Unit'] = np.where(big_frame['Variable'].str.contains('HALE'),'%',big_frame['Unit'])



big_frame['Variable'] = big_frame['Variable'].str.replace(r'\(.*?\)', '').str.strip()
big_frame['Variable'] = big_frame['Variable'].str.replace(r'\.[0-9]', '').str.strip()
big_frame['Variable'] = big_frame['Variable'].str.replace(r'\<br>', '')

big_frame['Year'] = big_frame['Year'].astype(str)
big_frame = big_frame[~big_frame['Year'].str.contains('-')]
big_frame['Country'] = big_frame['Country'].str.title()


big_frame.to_csv('data/WHO-data3.csv', index=False)
