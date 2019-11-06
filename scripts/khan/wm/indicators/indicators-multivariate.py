import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

def get_column(col_name):
    
    conn = sqlite3.connect('delphi.db')
    cur = conn.cursor()
    cur.execute("Select " +col_name+ " From indicator")
    rows = cur.fetchall()
    lst = list()

    for row in rows:
        lst.append(row[0])
    
    conn.commit()
    return lst


def series_to_supervised(data, column_names, n_in=1, n_out=1,  dropnan=True):
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (column_names[j], i)) for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i== 0:
            names += [('%s(t)' % (column_names[j])) for j in range(n_vars)]
        else:
            names +- [('%s(t+%d)' % (column_names[j], i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg



column = ['Variable', 'Value', 'Unit', 'Source', 'Country', 'County', 'State', 'Year', 'Month']
data = dict()

for col in column:
    data.update({col:get_column(col)})

df = pd.DataFrame(data)
# print(df.Variable)

df1 = df[df.Variable == 'IPC Phase Classification']    ### 2914 rows
df2 = df[df.Variable == 'Historical Production (Maize)']   ### 159 rows
df3 = df[df.Variable == 'Historical Average Total Daily Rainfall (Maize)'] #### 800 rows
# df = df[df.Variable == 'Battle-related deaths'] ### 6 rows
# df = df[df.Variable == 'Production, Meat indigenous, total']  ### 2 rows
# df = df[df.Variable == 'Mortality caused by road traffc injury']    ### NO DATA
# df = df[df.Variable == 'Imports of goods and services']  ### 18 rows
# df = df[df.Variable == 'Infation Rate']   #### NO DATA 
# df = df[df.Variable == 'CPIA policies for social inclusion/equity cluster average'] ### 6 rows 
# df = df[df.Variable == 'Poverty headcount ratio at national poverty lines'] ### 4 rows
# df = df[df.Variable == ' Net offcial development assistance and offcial aid received']     ### NO DATA 

# print(df)

for col in df1:
    if len(df1[col].unique()) == 1:
        df1 = df1.drop(columns = col)

df1 = df1.rename(columns={'Value':'IPC'})
df1 = df1.drop(columns = 'County')
# df1 = df1.set_index('State')

for col in df2:
    if len(df2[col].unique()) == 1:
        df2 = df2.drop(columns = col)

df2 = df2.rename(columns={'Value':'Maize Production'})
# df2 = df2.set_index('State')

for col in df3:
    if len(df3[col].unique()) == 1:
        df3 = df3.drop(columns = col)

df3 = df3.rename(columns={'Value':'Rainfall'})


# print(df1)
# print(df2)
# print(df3)

from functools import reduce
dfs = [df1, df2, df3]
df = reduce(lambda left, right: pd.merge(left, right, on=['State', 'Year', 'Month']), dfs)

# df_temp = pd.merge(df3, df2, how = 'inner', on = ['State', 'Year', 'Month'])
# df_temp1 = pd.merge(df1, df2, how = 'inner', on = ['State', 'Year', 'Month'])
# df = pd.merge(df_temp, df_temp1, how= 'inner', on = ['State', 'Year', 'Month'])
df = df.drop_duplicates()
print(df)

# df['date'] = pd.to_datetime(df['Month'].astype(int).astype(str) + '-' + df['Year'].astype(str), format='%m-%Y')
# df = df.drop(columns = ['Year', 'Month'])
# df['IPC'] = df['IPC'].astype(int)
# df['Rainfall'] = df['Rainfall'].astype(int)
# df['Maize Production'] = df['Maize Production'].astype(int)
# df_new = df.groupby(['State', 'date']).agg({'IPC':'mean', 'Rainfall':'mean', 'Maize Production':'mean'})
# df_new = df_new.reset_index()
# print(df_new)

# # print(df1)
# # print(df1.index)
# arr = df.groupby(['County', 'State', 'date'])['Value'].count()
# # print(arr)
# df1 = df.groupby(['County', 'State', 'date'])['Value'].sum()/arr

col_names = ['Maize Production', 'Rainfall']
p = len(col_names)

for key, grp in df_new.groupby(['State']):
# # for key, grp in df1.groupby(['County', 'State']):
    
    # print(key, grp)
    ydata1 = grp['Maize Production'].values
    ydata2 = grp['Rainfall'].values
    xdata = grp['date'].values
    ydata = grp[['Maize Production', 'Rainfall']].values.tolist()
    # print(len(ydata))

    







