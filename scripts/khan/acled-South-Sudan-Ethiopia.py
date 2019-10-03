import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

filenames = glob('/Users/souratoshkhan/delphi_old2/scripts/Data for November 2019 Evaluation/South Sudan Data/Humanitarian Data Exchange/201*.xlsx')

dfs = list()

for filename in filenames:
    df = pd.read_excel(filename)
    df.drop(df.index[0],inplace=True)
    dfs.append(df)

temp_df = pd.concat(dfs, ignore_index=True, sort=False)
# print(temp_df)


dfs = list()

for col in df.columns[2:]:
    val = temp_df[col].values
    df1 = pd.DataFrame({'Variable':col, 'Value':val, 'Date': temp_df['Date'].values, 'Country':temp_df['Country'].values})
    # print(df1)
    dfs.append(df1)

big_frame = pd.concat(dfs, ignore_index=True, sort=False)
big_frame['Year'] = big_frame['Date'].str.split('-').str.get(0).astype(int)
big_frame['Month'] = big_frame['Date'].str.split('-').str.get(1).astype(int)

# big_frame['Date'] = pd.to_datetime(df['Date'])
arr = big_frame.groupby(['Date', 'Country', 'Variable'])['Value'].count()
df1 = big_frame.groupby(['Date', 'Country', 'Variable'])['Value'].sum()/arr
df1 = df1.reset_index()

for key, grp in df1.groupby(['Country', 'Variable']):

    ydata = grp['Value'].values
    xdata = grp['Date'].values
    

    plt.figure()
    plt.title(key)
    plt.plot(xdata, ydata, color = 'b', label='sind conflict values')
    plt.legend()
    plt.xticks(rotation=45)
    # plt.xticks(lst_xdata, xdata, rotation='vertical')
    plt.show()


# big_frame.drop({'Date'}, axis=1, inplace=True)
# print(big_frame)



# df[date] = pd.to_datetime(df[date], format='%Y-%m')
# print(df[date])
