import pandas as pd
import matplotlib.pyplot as plt

filename = '/Users/souratoshkhan/delphi_old2/scripts/Data for November 2019 Evaluation/South Sudan Data/Humanitarian Data Exchange/acled-data-2011-2018-SSudan.xlsx'

df = pd.read_excel(filename, header=1)
df.drop(df.index[0], inplace=True)
df['Month'] = df['event_date'].str.split('-').str.get(1)
df = df[['year', 'Month', 'event_type', 'country', 'admin1', 'source', 'fatalities']]
df.rename({'event_type':'Variable', 'admin1': 'State', 'fatalities':'Value'}, axis=1, inplace=True)
df.columns = df.columns.str.capitalize()


#Grouping Data for South Sudan (Jonglei) -- No data for Ethiopia
df = df[((df['Country'] == 'South Sudan') & (df['State'] == 'Jonglei')) | (df['Country'] == 'Ethiopia')]
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Value'] = df['Value'].astype(int)
df['date'] = pd.to_datetime(df['Month'].astype(str) + '-' + df['Year'].astype(str), format='%m-%Y')
arr = df.groupby(['date', 'State', 'Variable'])['Value'].count()
df1 = df.groupby(['date', 'State', 'Variable'])['Value'].sum()/arr
df1 = df1.reset_index()

for key, grp in df1.groupby(['State', 'Variable']):
   
    ydata = grp['Value'].values
    xdata = grp['date'].values

    plt.figure()
    plt.title(key)
    plt.plot(xdata, ydata, color='b', label= 'ACLED conflict data')
    plt.ylabel('Average number of fatalities')
    plt.xlabel('Year')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


