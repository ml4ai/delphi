# libraries
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
# Get the data (csv file is hosted on the web)
# url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
# data = pd.read_csv(url)

data = pd.read_csv('../../delphi_timing/base/timing_2-12.csv')
print(data)

# Transform it to a long format
# df=data.unstack().reset_index()
df = data[['Nodes', 'Edges', 'Train']]
print(df)
# exit()
df.columns=["X","Y","Z"]

# And transform the old column name in something numeric
# df['X']=pd.Categorical(df['X'])
# df['X']=df['X'].cat.codes

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(df['Y'], df['X'], df['Z'], s=60)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)
plt.show()
exit()


# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()

# to Add a color bar which maps values to colors.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()

# Rotate it
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
ax.view_init(30, 45)
plt.show()

# Other palette
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
plt.show()