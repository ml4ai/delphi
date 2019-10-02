import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Flatten

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
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(ydata)

    data_reframed = series_to_supervised(data, col_names, 1, 1)
    # print('p value:', p)
    data_reframed.drop(data_reframed.columns[len(col_names)+1:], axis=1, inplace=True)


    new_data = data_reframed.values
    # print(series_to_supervised(data, col_names, 1, 1))
    print(data_reframed)

    n = data_reframed.shape[0]
    p = data_reframed.shape[1]

    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n

    data_train = new_data[np.arange(train_start, train_end), :]
    data_test = new_data[np.arange(test_start, test_end), :]

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    n = len(ydata)
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    regressor = Sequential()
    regressor.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dense(1))
    regressor.compile(loss='mean_squared_error', optimizer='adam')
    history = regressor.fit(X_train, y_train, epochs=50, batch_size=2,
            validation_data=(X_test, y_test), verbose=2, shuffle=False)
    
    # print(regressor.predict(X_test).shape)
    predicted_production = regressor.predict(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    inv_predict = np.concatenate((predicted_production, X_test[:,1:]), axis=1)
    inv_predict = scaler.inverse_transform(inv_predict)
    inv_predict = inv_predict[:,0]

    y_test = y_test.reshape((len(y_test), 1))
    inv_test = np.concatenate((y_test, X_test[:,1:]), axis=1)
    inv_test = scaler.inverse_transform(inv_test)
    inv_test = inv_test[:,0]

    y_train = y_train.reshape((len(y_train), 1))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[2]))
    inv_train = np.concatenate((y_train, X_train[:,1:]), axis=1)
    inv_train = scaler.inverse_transform(inv_train)
    inv_train = inv_train[:,0]
    # test_production = scaler.inverse_transform(y_test.reshape(-1,1))
    # train_new = scaler.inverse_transform(y_train.reshape(-1,1))
    
    # print(xdata.shape)
    # print(inv_train.shape)
    # print(inv_test.shape)
    # print(inv_predict.shape)
    
    plt.figure()
    plt.plot(xdata[:len(inv_train)], inv_train, label='train data')
    plt.plot(xdata[len(inv_train):len(inv_train)+len(inv_test)], inv_test, label = 'test data')
    plt.plot(xdata[len(inv_train):len(inv_train)+len(inv_test)], inv_predict, label = 'predicted data')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title('Train and Test Set Loss vs Epochs')
    plt.show()


    # data_train = ydata[np.arange(train_start, train_end)].reshape(-1,1)
    # data_test = ydata[np.arange(test_start, test_end)].reshape(-1,1)
    # # # print(data_train.shape)
    # # # print(data_test.shape)

    # scaler = MinMaxScaler()
    # data_train_scaled = scaler.fit_transform(data_train)
    # data_test_scaled = scaler.fit_transform(data_test)

    # look_back=1
    # X_train, y_train = create_dataset(data_train_scaled, look_back)
    # X_test, y_test = create_dataset(data_test_scaled, look_back)

    # # # print(X_train.shape, y_train.shape)
    # # # print(type(X_train), type(y_train))
    
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # # # print(X_train.shape, y_train.shape)
    # # # print(y_train)

    # svr_lin = SVR(kernel='linear', C=1e3)
    # # # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    # # # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # svr_lin.fit(X_train, y_train.ravel())
    # # # svr_poly.fit(X_train, y_train)
    # # # svr_rbf.fit(X_train, y_train)
    
    # predict_lin = svr_lin.predict(X_test)
    # predict_lin = scaler.inverse_transform(predict_lin.reshape(-1,1))

    # # predict_poly = svr_lin.predict(X_test)
    # # predict_poly = scaler.inverse_transform(predict_poly.reshape(-1,1))
    
    # # predict_rbf = svr_lin.predict(X_test)
    # predict_rbf = scaler.inverse_transform(predict_rbf.reshape(-1,1))
    
    # plt.figure()
    # plt.title(key)
    # plt.plot(xdata[:train_end+look_back+1], ydata[:train_end+look_back+1], color='b', label='Train')
    # plt.plot(xdata[train_end+look_back+1:], ydata[train_end+look_back+1:], color='g', label='Test')
    # plt.plot(xdata[train_end+look_back+1:], predict_lin, color = 'orange', label = 'Linear Prediction')
    # # # plt.plot(xdata[train_end+look_back+1:], predict_poly, color = 'k',label = 'Poly Prediction')
    # # # plt.plot(xdata[train_end+look_back+1:], predict_rbf, color = 'm', label = 'RBF Prediction')
    # plt.legend()
    # plt.show()




# import urllib.request, urllib.parse, urllib.error
# import config
# import json

# stock_price = list()

# symbol = ['AMZN', 'MSFT', 'IBM', 'AAPL']

# for SYMBOL in symbol:
    # url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+SYMBOL+'&outputsize=full&apikey='+config.public_api_key

    # data = urllib.request.urlopen(url).read().decode()
    # js = json.loads(data)

    # close_price = list()
    # dates = list()

    # for item in js["Time Series (Daily)"]:
        # close_price.append(float(js['Time Series (Daily)'][item]['4. close']))
        # dates.append(item)
    
    # stock_price.append(close_price)

# dict = {'close':stock_price}

# stock_df = pd.DataFrame(stock_price)
# stock_df = stock_df.transpose()
# stock_df.index = dates
# stock_df.index = pd.to_datetime(stock_df.index)
# stock_df = stock_df.sort_index(ascending=True)
# stock_df.columns = symbol


# n = stock_df.shape[0]
# p = stock_df.shape[1]


# stock_name = input('Enter any stock symbol from the set {AMZN, MSFT, IBM, AAPL} : ')
# column_names = list(stock_df.columns)
# a = column_names.index(stock_name)
# column_names[a] = column_names[0]
# column_names[0] = stock_name
# stock_df = stock_df[column_names]

# # plt.figure()
# stock_df[stock_name].plot(label = stock_name)
# plt.legend()
# plt.ylabel('Price in $')
# plt.title('Stock Prices of ' + stock_name)
# plt.show()

# data = stock_df.values
# print(data)







