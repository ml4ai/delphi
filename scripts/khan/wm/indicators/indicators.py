import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
df1 = df.groupby(['County', 'State', 'date'])['Value'].sum()
# print(df1)
# print(df1.index)



for key, grp in df.groupby(['County', 'State']):
    ydata = grp['Value'].values
    xdata = grp['date'].values

    n = len(ydata)
    train_start = 0
    train_end = int(np.floor(0.6*n))
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

    # print(X_train)
    # print(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(LSTM(units=20))
    regressor.add(Dense(units=1))
    regressor.compile(loss='mean_squared_error', optimizer='adam')
    regressor.fit(X_train, y_train, epochs=10, batch_size=2)

    prediction = regressor.predict(X_test)
    prediction = scaler.inverse_transform(prediction)

    plt.figure()
    plt.title(key)
    plt.plot(xdata[:train_end+look_back+1], ydata[:train_end+look_back+1], color='b', label='Train')
    plt.plot(xdata[train_end+look_back+1:], ydata[train_end+look_back+1:], color='g', label='Test')
    plt.plot(xdata[train_end+look_back+1:], prediction, color = 'orange', label = 'Predicted Data')
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
