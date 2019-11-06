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


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back -1):
        a = dataset[i:i+look_back]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)


column = ['Variable', 'Value', 'Unit', 'Source', 'Country', 'County', 'State', 'Year', 'Month']
data = dict()

for col in column:
    data.update({col:get_column(col)})

df = pd.DataFrame(data)
df = df[df.Variable == 'IPC Phase Classification']

for col in df:
    if len(df[col].unique()) == 1:
        df = df.drop(columns = col)

df['date'] = pd.to_datetime(df['Month'].astype(int).astype(str) + '-' + df['Year'].astype(str), format='%m-%Y')
df = df.drop(columns = ['Year', 'Month'])
df['Value'] = df['Value'].astype(int)
arr = df.groupby(['County', 'State', 'date'])['Value'].count()
# print(arr)
df1 = df.groupby(['County', 'State', 'date'])['Value'].sum()/arr
df1 = df1.reset_index()
# print(df1)
# print(df1.index)


for key, grp in df1.groupby(['County', 'State']):
    
    # print(key, grp)
    ydata = grp['Value'].values
    xdata = grp['date'].values

    n = len(ydata)
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n

    data_train = ydata[np.arange(train_start, train_end)].reshape(-1,1)
    data_test = ydata[np.arange(test_start, test_end)].reshape(-1,1)
    # print(data_train.shape)
    # print(data_test.shape)

    scaler = MinMaxScaler()
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.fit_transform(data_test)

    look_back=3
    X_train, y_train = create_dataset(data_train_scaled, look_back)
    X_test, y_test = create_dataset(data_test_scaled, look_back)

    # print(X_train.shape, y_train.shape)
    # print(type(X_train), type(y_train))
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # print(X_train.shape, y_train.shape)
    # print(y_train)

    svr_lin = SVR(kernel='linear', C=1e3)
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(X_train, y_train.ravel())
    # svr_poly.fit(X_train, y_train)
    # svr_rbf.fit(X_train, y_train)
    
    predict_lin = svr_lin.predict(X_test)
    predict_lin = scaler.inverse_transform(predict_lin.reshape(-1,1))

    predict_poly = svr_lin.predict(X_test)
    predict_poly = scaler.inverse_transform(predict_poly.reshape(-1,1))
    
    predict_rbf = svr_lin.predict(X_test)
    predict_rbf = scaler.inverse_transform(predict_rbf.reshape(-1,1))
    
    plt.figure()
    plt.title(key)
    plt.plot(xdata[:train_end+look_back+1], ydata[:train_end+look_back+1], color='b', label='Train')
    plt.plot(xdata[train_end+look_back+1:], ydata[train_end+look_back+1:], color='g', label='Test')
    plt.plot(xdata[train_end+look_back+1:], predict_lin, color = 'orange', label = 'Linear Prediction')
    # # plt.plot(xdata[train_end+look_back+1:], predict_poly, color = 'k',label = 'Poly Prediction')
    # # plt.plot(xdata[train_end+look_back+1:], predict_rbf, color = 'm', label = 'RBF Prediction')
    plt.legend()
    plt.show()


    # plt.scatter(grp['date'], grp['Value'], label = key)
    # plt.legend()
    # plt.show()



# df1.unstack(1).plot.barh(figsize=(8,15))
# plt.show()
# print(df[:100])
# print(df.index)

# df2 = df.groupby('State')['Value'].sum()
# print(df2.index)
# plt.figure()
# df1.unstack().plot()
# plt.show()
