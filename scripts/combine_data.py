import pandas as pd

data_dir = '../data/'
fao_df = pd.read_csv(data_dir+'south_sudan_data_fao.csv', sep='|')
wdi_df = pd.read_csv(data_dir+'south_sudan_data_wdi.csv', sep='|')

fao_df['Indicator Name'] = fao_df['Element'] + ', ' + fao_df['Item']

del fao_df['Element']
del fao_df['Item']

wdi_df['Unit'] = (wdi_df['Indicator Name'].str.partition('(')[2]
                                          .str.partition(')')[0])
wdi_df['Indicator Name'] = wdi_df['Indicator Name'].str.partition('(')[0]
wdi_df = wdi_df.set_index('Indicator Name')
fao_df = fao_df.pivot_table(values='Value', index=['Indicator Name', 'Unit'],
                   columns = 'Year').reset_index().set_index('Indicator Name')
combined_df = pd.concat([fao_df, wdi_df], sort=True)
combined_df.to_csv(data_dir+'south_sudan_data.csv', sep='|')
