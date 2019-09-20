import pandas as pd
from glob import glob

excel_files = ['Life expectancy-Ethiopia-SSudan.xls', 'child_malnutrition-Ethiopia-SSudan.xlsx']

dfs = list()

for filename in excel_files:
    df = pd.read_excel('Data for November 2019 Evaluation/South Sudan Data/WHO/'+ filename)


    col1 = df.iloc[0, 0]
    col2 = df.iloc[0,1]

    df.drop(df.index[0], inplace=True)

    df[col1] = df.iloc[0:,0]
    df[col2] = df.iloc[0:,1]

    df.drop(df.columns[[0,1]], axis=1, inplace=True)
    val = df.iloc[:,0].values
    variable = df.columns[0]

    big_frame = pd.DataFrame({'Value':val, 'Variable':variable, col1:df[col1].values, col2:df[col2].values})


    for col in df.columns[1:-2]:
        val = df[col].values
        variable = col
        df2 = pd.DataFrame({'Value':val,'Variable':variable, col1:df[col1].values, col2:df[col2].values})
        big_frame = pd.concat([big_frame, df2], sort=False, ignore_index=True)

    big_frame['Source'] = 'WHO'
    big_frame['County'], big_frame['Month'] = None, None
    big_frame['Unit'] = big_frame['Variable'].apply(lambda st: st[st.find('(') + 1:st.find(')')] )
    # print(big_frame.columns)
    # print(big_frame)
    dfs.append(big_frame)

big_frame = pd.concat(dfs, sort=False, ignore_index=True)
print(big_frame)

# print(big_frame)

