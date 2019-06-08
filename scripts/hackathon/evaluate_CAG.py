import sys
import pickle
from delphi import AnalysisGraph as AG
import pandas as pd
from delphi.db import engine
import numpy as np
import random
np.random.seed(87)
random.seed(87)

def run_trial(G, intervened_node, delta, n_timesteps: int, dampen=False):
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({intervened_node})/∂t"] = delta
    s0.to_csv("bmi_config.txt", index_label="variable")
    G.initialize()

    for t in range(1, n_timesteps + 1):
        G.update(dampen=dampen)


def evaluate_CAG(input,output,target,intervened_node,n_timesteps: int =
        1,trials = 10,start_year = 2012,start_month = None,dampen=False):
    with open(input,"rb") as f:
        G = pickle.load(f)

    preds = np.zeros(trials)
    target_indicator = list(G.nodes(data=True)[target]['indicators'].keys())[0]
    intervened_indicator = list(G.nodes(data=True)[intervened_node]['indicators'].keys())[0]
    df = pd.read_sql_table("indicator", con=engine)


    for i in range(1,trials):
        G.parameterize(year = start_year,month = start_month)
        G.get_timeseries_values_for_indicators()
        G.res = 200
        G.assemble_transition_model_from_gradable_adjectives()
        G.sample_from_prior()
        G.get_timeseries_values_for_indicators()
