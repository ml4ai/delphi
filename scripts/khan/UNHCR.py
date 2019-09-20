import pandas as pd

dfs = list()

filename = 'Data for November 2019 Evaluation/South Sudan Data/UNHCR/unhcr-demographics-residing-ssd.csv'
df = pd.read_csv(filename)
df.drop(df.index[0], inplace=True)
df.rename({'Country / territory of asylum/residence': 'Country', 'Location Name': 'State'}, axis=1, inplace=True)
df['County'] = df['State']
df_County = df[df['County'].str.contains('County')]
df_County.drop('State', axis=1, inplace=True)
df_State = df[df['State'].str.contains('State')]
df_State.drop('County', axis=1, inplace=True)
df_new = df_State.append(df_County, ignore_index=True, sort=False)
colnames = df_new.columns



for col in colnames[3:-1]:
    val = df_new[col].values
    df1 = pd.DataFrame({'Value':val, 'Variable':col, 'Year': df_new['Year'].values, 'Country':df_new['Country'].values, 'State':df_new['State'].values, 'County':df_new['County'].values})
    dfs.append(df1)

big_frame = pd.concat(dfs, sort=False, ignore_index=True)
big_frame['Source'], big_frame['Unit'] = 'UNHCR', None
print(big_frame)


# filename = 'Data for November 2019 Evaluation/South Sudan Data/UNHCR/unhcr-time-series-originating-ssd.csv'
# df = pd.read_csv(filename)
# df.drop(df.index[0], inplace=True)
# if 'Location Name' not in df.columns:
    # df.rename({'Country / territory of asylum/residence': 'Country'}, axis=1, inplace=True)
# else:
    # df.rename({'Country / territory of asylum/residence': 'Country', 'Location Name': 'State'}, axis=1, inplace=True)
# # df = df[(df['Country'] == 'South Sudan') | (df['Country'] == 'Ethiopia')] 

# # print(df['Country'].values)
# print(df.columns)
# print(df['Origin'].values)

# filename = 'Data for November 2019 Evaluation/South Sudan Data/UNHCR/unhcr_persons_of_concern_origin_ssd.csv'
# df = pd.read_csv(filename)
# print(df.columns)


# filename = 'Data for November 2019 Evaluation/South Sudan Data/UNHCR/unhcr_persons_of_concern_residence_ssd.csv'
# df = pd.read_csv(filename)
# print(df.columns)
