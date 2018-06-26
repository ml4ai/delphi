import pandas as pd

fao_data = '../data/south_sudan_data_fao.csv'
df = pd.read_csv(fao_data, sep='|')
print(df)
