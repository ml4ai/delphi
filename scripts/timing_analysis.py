import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 18, 'figure.dpi': 150})

out_dir = 'plots/timing_analysis/'
if out_dir:
    out_path = pathlib.Path(out_dir)
    if not out_path.is_dir():
        print(f'\nMaking output directory: {out_dir}')
        out_path.mkdir(parents=True, exist_ok=True)

df_timing = pd.read_csv('timing.csv')
df_timing['Training (minutes)'] = df_timing['Train'].apply(lambda ms: ms / 60000.0)
df_timing['Predicting (seconds)'] = df_timing['Predict'].apply(lambda ms: ms / 1000.0)

times = ['Training (minutes)', 'Predicting (seconds)']
plot_no = 1


for y in times:
    sns.boxplot(data=df_timing, x='Nodes', y=y)
    plt.title(f'Training Times')
    plt.savefig(f'{out_dir}{plot_no}_{y} - box.png')
    plt.close()
    plot_no += 1

    sns.violinplot(data=df_timing, x='Nodes', y=y, scale='count', bw=.15, inner='box')
    plt.title(f'Training Times')
    plt.savefig(f'{out_dir}{plot_no}_{y} - violin.png')
    plt.close()
    plot_no += 1

    sns.lineplot(data=df_timing, x='Nodes', y=y, marker='o')
    plt.title(f'Prediction Times')
    plt.savefig(f'{out_dir}{plot_no}_{y} - line.png')
    plt.close()
    plot_no += 1

    sns.scatterplot(data=df_timing, x='Nodes', y=y)
    plt.title(f'Training Times')
    plt.savefig(f'{out_dir}{plot_no}_{y} - scatter.png')
    plt.close()
    plot_no += 1


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Axes3D.plot_surface(df_timing['Nodes'], df_timing['Edges'], df_timing['Training (minutes)'])
plt.show()
'''
'''
for nodes in range(2, 14):
    for y in times:
        sns.lineplot(data=df_timing[df_timing['Nodes'] == nodes], x='Edges', y=y, marker='o')
        plt.title(f'Training Times - {nodes} Nodes CAG')
        plt.savefig(f'{out_dir}nodes_{nodes}_{y} - line.png')
        plt.close()
        #plot_no += 1
'''