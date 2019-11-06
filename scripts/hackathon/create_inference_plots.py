# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
sns.set_style('darkgrid')

def bandPassFilter(xs, xmin, xmax):
    return [x for x in xs if xmin < x < xmax]

def make_inference_plots(input, output):
    with open(sys.argv[1], "rb") as f:
        G = pickle.load(f)
    G.create_bmi_config_file()
    s0 = pd.read_csv("bmi_config.txt", index_col=0)
    s0.loc["∂(UN/events/weather/precipitation)/∂t"] = 0.1
    s0.to_csv("bmi_config.txt")
    G.initialize()
    G.update()
    G.update()
    for n in tqdm(G.nodes(data=True)):
        fig, ax = plt.subplots(figsize=(4,4))
        indicator = list(n[1]["indicators"]).values()[0]
        sns.distplot(bandPassFilter(indicator.dataset, 0, 2), ax=ax)
        label = indicator.name
        ax.set_xlabel(indicator.name)
        fig.savefig(list(n[1]["indicators"].values())[0].name+".pdf")
    with open("figs.tex", "w") as f:
        f.write("\n".join([f"\\includegraphics[width=0.4\\textwidth]{{{list(n[1]['indicators'].values())[0].name}.pdf}}"
            for n in G.nodes(data=True)]))





if __name__ == "__main__":
    make_inference_plots(sys.argv[1], sys.argv[2])
