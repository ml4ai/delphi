import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../scripts/plots/No_Data_run_100,000/_7_Predictions_a_ind.csv')
#df = df[df['Time Step'] == 35]
filter = 2000000
filter = 200000
filter = 30000
filter = 3000
#filter = 300

df = df[(-filter < df['Prediction']) & (df['Prediction'] < filter)]
print(min(df['Prediction']))
print(max(df['Prediction']))

sns.boxplot(x="Time Step", y="Prediction", data=df)
plt.show()
exit()

df = df[df['Time Step'] == 35]

print(min(df['Prediction']))
print(max(df['Prediction']))
df = df[(-50000 < df['Prediction']) & (df['Prediction'] < 30000)]
#df = df[(-50000 < df['Prediction']) & (df['Prediction'] < -5000)]
#df = df[(2000 < df['Prediction']) & (df['Prediction'] < 30000)]
#print(df)

sns.histplot(df, x='Prediction', element='step',
             color=(0.9375, 0.5, 0.5))

plt.show()


exit()


df = pd.read_csv('../scripts/plots/No_Data_run_100,000/_4_Derivatives.csv')
df = df[df['Concept'] == 'a']
#print(df)
#print(df.columns)

samples = []
for idx, row in df.iterrows():
    #print(row[1])
    samples += [row["Derivative"]] * row["# of Samples"]

print(len(samples))
sns.histplot(samples, element='step', stat='density', bins=20)
sns.kdeplot(samples, color='red')
#sns.lineplot(x="Derivative", y="# of Samples", data=df)
plt.show()
exit()

