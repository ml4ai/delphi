import pandas as pd
from glob import glob
import numpy as np

dfs = list()

directoryPath = '../data/raw/data_for_november_2019_evaluation/south_sudan_data/'
filenames = glob(directoryPath + '*.csv')


colnames = ['HWAMA', 'TAVGA', 'PRCPA']
for filename in filenames:
    df = pd.read_csv(filename, encoding='latin-1',usecols=lambda x: x in colnames)
    df['Year'] = np.arange(2018-len(df), 2018)
    big_frame = pd.DataFrame({'Value':df.iloc[:,0].values, 'Variable': df.columns[0], 'Year':df['Year'].values})

    for col in df.columns[1:-1]:
        val = df[col].values
        indicator = col
        df_new = pd.DataFrame({'Value':val, 'Variable':indicator})
        df_new = pd.concat([df_new, df['Year']], axis=1, join='inner')
        big_frame = pd.concat([big_frame, df_new], sort=False, ignore_index=True)
        if filename == directoryPath + 'SSD_jonglei_maize_unimodal_historical_all.csv':
            big_frame['State'] = 'Jonglei'
        else:
            big_frame['State'] = None
    dfs.append(big_frame)
    
big_frame = pd.concat(dfs, ignore_index=True, sort=True)

big_frame['Country'], big_frame['Source'] = 'South Sudan', 'DSSAT'
big_frame['Month'], big_frame['Unit'], big_frame['County'] = None, None, None

dict_var = {'HWAMA': 'Average Harvested Weight at Maturity (Maize)', 'TAVGA' :
        'Average Temperature for Maize Production', 'PRCPA': 'Average Precipitation for Maize Production'}

big_frame['Variable'].replace(dict_var, inplace=True)

big_frame.to_csv('../data/dssat-maize-data1.csv', index=False)

