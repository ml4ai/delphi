import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

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


def svm_model(X_train, y_train, X_test, y_test, C):
    
    param = C

    svr_lin = SVR(kernel='linear', C=param)

    svr_lin.fit(X_train, y_train.ravel())

    predict_lin = svr_lin.predict(X_test)

    # print('Accuracy Score:', svr_lin.score(X_test, y_test))
    print('Error:', mean_squared_error(y_test, predict_lin)) 
    
    return predict_lin, param



def train_model(X_train, y_train, X_test, y_test, epochs):
    model = Sequential(
            [Dense(10,activation='relu', input_shape=(X_train.shape[1],)),
            Dense(10, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1, activation='linear')
                ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs,
            validation_data=(X_test, y_test))
    return model, history



def plot_results(key, Model, X_train, y_train, y_test, predictions):

    df_train = pd.DataFrame(data=y_train,index=xdata[:len(X_train)])
    df_test = pd.DataFrame(data=y_test,index=xdata[len(X_train)+1:])
    df_predict = pd.DataFrame(data=predictions,index=xdata[len(X_train)+1:])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, ylabel='Normalized Maize Production', xlabel='Year')
    # df_train.plot(ax=ax1, color='b', label='Train')
    # df_test.plot(ax=ax1, color='g', label='Test')
    # df_predict.plot(ax=ax1, color='orange', label='Predictions')
    # plt.title(key+':Maize Production at timestep t as a function of Rainfall, Month Value, and Production at (t-1) timesteps')
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(xdata[:len(X_train)], y_train, color='b', label='Train')
    plt.plot(xdata[len(X_train)+1:], y_test, color='g', label='Test')
    plt.plot(xdata[len(X_train)+1:], predictions, color = 'orange', label = Model + 'Prediction')
    plt.title(key+':Maize Production at timestep t as a function of Rainfall, Month Value, and Production at (t-1) timesteps')
    plt.ylabel('Normalized Maize Production')
    plt.xlabel('Year')
    plt.legend()
    plt.show()




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

    data_reframed = series_to_supervised(data, col_names, 1, 1)
    data_reframed.drop(data_reframed.columns[len(col_names)+1:], axis=1, inplace=True)


    new_data = data_reframed.values
    # print(series_to_supervised(data, col_names, 1, 1))
    # print(data_reframed)
    # print(new_data)

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

    # print('X-train shape', X_train.shape)
    # print('X-test shape', X_test.shape)
    # print('y-train shape', y_train.shape)
    # print('y-test shape', y_test.shape)


    svm_predict, C = svm_model(X_train, y_train, X_test, y_test, C=0.1)
    
    plot_results(key, 'SVM Linear with C='+str(C), X_train, y_train, y_test, svm_predict)


    svm_predict, C = svm_model(X_train, y_train, X_test, y_test, C=1e1)
    
    plot_results(key, 'SVM Linear with C='+str(C), X_train, y_train, y_test, svm_predict)


    svm_predict, C = svm_model(X_train, y_train, X_test, y_test, C=1e3)
    
    plot_results(key, 'SVM Linear with C='+str(C), X_train, y_train, y_test, svm_predict)


    NN_results, loss = train_model(X_train, y_train, X_test, y_test,
            epochs=10)

    NN_predict = NN_results.predict(X_test)
    
    plot_results(key, 'NN', X_train, y_train, y_test, NN_predict)







