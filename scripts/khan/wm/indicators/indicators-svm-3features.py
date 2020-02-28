import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation


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

def train_model(X_train, y_train, X_test, y_test, epochs):
    model = Sequential(
            [ Dense(10, activation='relu', input_shape=(X_train.shape[1], 1)),
              Dense(10, activation='relu'),
              Dense(10, activation='relu'),
              Dense(1, activation='linear')
             ])
    model.compile(optimize='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, validation=(X_test, y_test))
    return model, history


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

df = pd.merge(df3, df2, how = 'inner', on = ['State', 'Year', 'Month'])
# print(df)

df['date'] = pd.to_datetime(df['Month'].astype(int).astype(str) + '-' + df['Year'].astype(str), format='%m-%Y')
df = df.drop(columns = ['Year', 'Month'])
# df['IPC'] = df['IPC'].astype(int)
df['Rainfall'] = df['Rainfall'].astype(int)
df['Maize Production'] = df['Maize Production'].astype(int)
# arr = df.groupby(['State', 'date'])['IPC'].count()
arr = df.groupby(['State', 'date'])['Rainfall'].count()
# df_new = df.groupby(['State', 'date']).agg({'IPC':'mean', 'Maize Production':'mean'})
df_new = df.groupby(['State', 'date']).agg({'Rainfall':'mean', 'Maize Production':'mean'})
# df_new = df.groupby(['State', 'date'])['IPC'].sum()/arr
df_new = df_new.reset_index()
df_new['month'] = df_new['date'].dt.month.astype(int)
# df_new['month_sin'] = np.sin(2*np.pi*df_new['date'].dt.month/12)
# df_new['month_cos'] = np.cos(2*np.pi*df_new['date'].dt.month/12)
# df_new.plot.scatter('month_sin', 'month_cos').set_aspect('equal')
# plt.show()
# print(df_new)

# # print(df1)
# # print(df1.index)
# arr = df.groupby(['County', 'State', 'date'])['Value'].count()
# # print(arr)
# df1 = df.groupby(['County', 'State', 'date'])['Value'].sum()/arr

col_names = ['Maize Production', 'Rainfall', 'month']
p = len(col_names)

for key, grp in df_new.groupby(['State']):
# # for key, grp in df1.groupby(['County', 'State']):
    
    # print(key, grp)
    ydata1 = grp['Maize Production'].values
    ydata2 = grp['Rainfall'].values
    xdata = grp['date'].values
    ydata = grp[['Maize Production', 'Rainfall', 'month']].values.tolist()
    # print(len(ydata))
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(ydata)

    n = data.shape[0]
    p = data.shape[1]

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n
    
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    X_train, y_train = data_train[:, 1:], data_train[:, 0]
    X_test, y_test = data_test[:, 1:], data_test[:, 0]

    print('X_train shape', X_train.shape)    
    print('X_test shape', X_test.shape)    
    print('y_train shape', y_train.shape)    
    print('y_test shape', y_test.shape)    

    svr_lin = SVR(kernel='linear', C=1e3)

    svr_lin.fit(X_train, y_train.ravel())

    predict_lin = svr_lin.predict(X_test)
    

    plt.figure()
    plt.title(key)
    plt.plot(xdata[:len(X_train)], y_train, color='b', label='Train')
    plt.plot(xdata[len(X_train):], y_test, color='g', label='Test')
    plt.plot(xdata[len(X_train):], predict_lin, color = 'orange', label = 'Linear Prediction')
    plt.xticks()
    plt.legend()
    plt.show()











