import pandas as pd
from glob import glob


dfs = list()

filenames = glob('Data for November 2019 Evaluation/South Sudan Data/WHO/*SSudan.csv')
colnames = ['GHO (DISPLAY)', 'YEAR (DISPLAY)', 'REGION (DISPLAY)', 'Numeric']
for filename in filenames:
    df = pd.read_csv(filename, encoding='latin-1',usecols=lambda x: x in colnames)
    dfs.append(df)
    
big_frame = pd.concat(dfs, ignore_index=True, sort=True)
for col in colnames:
    # print(col.split(), len(col.split()), col.split()[0])
    if len(col.split()) > 1:
        big_frame.rename({col:col.split()[0]}, axis=1, inplace=True)

big_frame.rename({'GHO':'Variable', 'Numeric':'Value', 'REGION':'Country', 'YEAR':'Year'}, axis=1, inplace=True)

big_frame['Unit'] = big_frame['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')]) 

# print(big_frame.columns)
# print(big_frame)


filename = 'Data for November 2019 Evaluation/South Sudan Data/WHO/South Sudan WHO Statistics Summary.csv'
df = pd.read_csv(filename, index_col=False)
for col in df.columns:
    if col == 'Unnamed: 0':
        df.rename({col:'Country'}, axis=1, inplace=True)
    if len(col.split('.')) == 2:
        df.rename({col:col.split('.')[0]}, axis=1, inplace=True)

# print(df)



variables = df.iloc[0].values
values = variables[1:]
df_new1 = pd.DataFrame({'Year': values, 'Country':df.columns[1:]})

# print(df_new1)

for i in range(1,df.shape[0]):
    variables = df.iloc[i].values
    indicator = variables[0]
    values = variables[1:]
    # print(indicator, values)
    df_new2 = pd.DataFrame({'Value':values, 'Variable':indicator})
    df_new = pd.concat([df_new2, df_new1], axis=1, join='inner')
    # print(df_new['Variable'].split('(').split(')'))    
    # print(df_new['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')]))
    df_new['Unit'] = df_new['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')]) 
    # print(df_new)
    big_frame = pd.concat([big_frame, df_new], sort=False, ignore_index=True)

big_frame['Source'], big_frame['Month'], big_frame['County'] = 'WHO', None, None

# print(df_new2)
print(big_frame.columns)



# df = pd.read_csv(filename, skiprows=1).set_index('Indicator')
# df = df.transpose()
# df.dropna(how='all',axis=1, inplace=True)
# df['YEAR'] = df.index
# df.reset_index(inplace=True)
# df.rename({"Unnamed: 0": "REGION"}, axis='columns')

# df_new = pd.merge([big_frame, df], how='inner', on='YEAR')
