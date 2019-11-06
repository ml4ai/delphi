import numpy as np
import pandas as pd
from glob import glob

dfs = list()

directoryPath = 'data/raw/data_for_november_2019_evaluation/south_sudan_data/IMF/'
filenames = glob(directoryPath + 'imf*.xlsx')
for filename in filenames:
    
    df = pd.read_excel(filename)
    df = df.transpose()

    index_val = df.index.values
    indicator = index_val[0]
    year_val = index_val[1:]
    

    colnames = df.iloc[0,:]

    df = df.iloc[1:, :]
    col_dict = dict(zip(list(range(df.shape[1])), colnames))
   
    df.rename(col_dict, axis=1, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    df.rename({'South Sudan, Republic of' : 'Value'}, axis=1,  inplace=True)
    df['Value'].replace('no data', np.nan, inplace=True)
    
    df['Variable'] = indicator
    df['Year'] = df.index
    df['Country'] = 'South Sudan'
    
    for col in colnames:    
        if 'Ethiopia' == col:
            ethiopia_ind_val = df[col].values
            df1 = pd.DataFrame({'Variable':indicator, 'Year':df.index, 'Value': ethiopia_ind_val, 'Country':'Ethiopia'})
            df = pd.concat([df, df1], sort=False, ignore_index=True)    
    
    

    df = df[['Year', 'Variable', 'Value', 'Country']]
    
    dfs.append(df)


big_frame = pd.concat(dfs, sort=False, ignore_index=False)
big_frame.index = list(range(big_frame.shape[0]))    
big_frame['Unit'] = big_frame['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')])     
big_frame['Source'], big_frame['Month'], big_frame['County'], big_frame['State'] = 'IMF', None, None, None

big_frame.dropna(subset=['Value'], inplace=True)
big_frame['Variable'] = big_frame['Variable'].str.replace(r'\(.*?\)', '').str.strip()

big_frame.to_csv('data/IMF-data.csv', index=False)

