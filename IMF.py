import numpy as np
import pandas as pd
from glob import glob

dfs = list()

filenames = glob('Data for November 2019 Evaluation/South Sudan Data/IMF/imf*.xlsx')
# filenames = ['Data for November 2019 Evaluation/South Sudan Data/IMF/imf-dm-export-inflation-2012-2017-SouthSudan.xlsx']
# filenames = ['Data for November 2019 Evaluation/South Sudan Data/IMF/imf-dm-export--Real_GDP-SouthSudan-Ethiopia.xlsx']
# filenames = ['Data for November 2019 Evaluation/South Sudan Data/IMF/imf-dm-export-Population-Ethiopia-SSudan.xlsx']
for filename in filenames:
    # print(filename)
    
    df = pd.read_excel(filename)
    # df.dropna(axis=0, how='all', inplace=True)
    df = df.transpose()

    index_val = df.index.values
    # print(index_val)
    indicator = index_val[0]
    year_val = index_val[1:]
    

    colnames = df.iloc[0,:]
    # print(colnames)

    df = df.iloc[1:, :]
    col_dict = dict(zip(list(range(df.shape[1])), colnames))
    # print(col_dict)
   
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
    
    # print(df.columns)
    # print(df.Country.values)
    dfs.append(df)


big_frame = pd.concat(dfs, sort=False, ignore_index=False)
big_frame.index = list(range(big_frame.shape[0]))    
big_frame['Unit'] = big_frame['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')])     
big_frame['Source'], big_frame['Month'], big_frame['County'], big_frame['State'] = 'IMF', None, None, None
print(big_frame)
