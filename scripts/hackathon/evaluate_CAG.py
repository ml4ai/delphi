import sys
import pickle
from delphi import AnalysisGraph as AG
import pandas as pd
from delphi.db import engine
import numpy as np
import random
from scipy import stats
np.random.seed(87)
random.seed(87)


def get_predictions(G, target_node, intervened_node, delta, n_timesteps, dampen=False):
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({intervened_node})/∂t"] = delta
    s0.to_csv("bmi_config.txt", index_label="variable")
    G.initialize()

    pred = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        G.update(dampen=dampen)
        pred[t]=np.median(list(G.nodes(data=True)[target_node]['indicators'].values())[0].samples)

    return pred


def get_true_values(G,target_node,n_timesteps,start_year,start_month):
    df = pd.read_sql_table("indicator", con=engine)
    target_indicator = list(G.nodes(data=True)[target_node]['indicators'].keys())[0]
    target_df = df[df['Variable'] == target_indicator]

    true_vals = np.zeros(n_timesteps)
    year = start_year
    month = start_month
    for j in range(n_timesteps):
        if target_df[target_df['Year'] == year].empty:
           true_vals[j] = target_df['Value'].values.astype(float).mean()
        elif target_df[target_df['Year'] == year][target_df['Month'] ==
                month].empty:
            true_vals[j] = target_df[target_df['Year'] ==
                    year]['Value'].values.astype(float).mean()
        else:
            true_vals[j] = target_df[target_df['Year'] == year][target_df['Month']
                == month]['Value'].values.astype(float).mean()
        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return true_vals


def calculate_timestep(start_year,start_month,end_year,end_month):
    diff_year = end_year - start_year
    year_to_month = diff_year*12
    return year_to_month - (start_month - 1) + (end_month - 1)


# This is specifically for when the dampen argument for G.update is set to False.
def estimate_delta(G,intervened_node,n_timesteps,start_year,start_month,end_year,end_month):
    df = pd.read_sql_table("indicator", con=engine)
    intervener_indicator = list(G.nodes(data=True)[intervened_node]['indicators'].keys())[0]
    intervened_df = df[df['Variable'] == intervener_indicator]

    if intervened_df[intervened_df['Year'] == start_year].empty:
        start_val = intervened_df['Value'].values.astype(float).mean()
    elif intervened_df[intervened_df['Year'] ==
            start_year][intervened_df['Month'] == start_month].empty:
        start_val = intervened_df[intervened_df['Year'] ==
                start_year]['Value'].values.astype(float).mean()
    else:
        start_val = intervened_df[intervened_df['Year'] ==
            start_year][intervened_df['Month'] ==
                    start_month]['Value'].values.astype(float).mean()

    if intervened_df[intervened_df['Year'] == end_year].empty:
        end_val = intervened_df['Value'].values.astype(float).mean()
    elif intervened_df[intervened_df['Year'] ==
            end_year][intervened_df['Month'] == end_month].empty:
        end_val = intervened_df[intervened_df['Year'] ==
                end_year]['Value'].values.astype(float).mean()
    else:
        end_val = intervened_df[intervened_df['Year'] ==
            end_year][intervened_df['Month'] ==
                    end_month]['Value'].values.astype(float).mean()

    diff_val = end_val - start_val
    return diff_val/n_timesteps

def evaluate_CAG(
    input,
    target_node: str,
    intervened_node: str,
    start_year=2012,
    start_month=None,
    end_year=2017,
    end_month=None,
    dampen=False,
    res = 200
):
    with open(input,"rb") as f:
        G = pickle.load(f)

    G.parameterize(year = start_year,month = start_month)
    G.get_timeseries_values_for_indicators()
    G.res = res
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.get_timeseries_values_for_indicators()

    if start_month is None:
        start_month = 1
    if end_month is None:
        end_month = 1

    n_timesteps = calculate_timestep(start_year,start_month,end_year,end_month)

    true_vals = get_true_values(G,target_node,n_timesteps,start_year,start_month)

    delta = estimate_delta(G,intervened_node,n_timesteps,start_year,start_month,end_year,end_month)

    preds = get_predictions(G, target_node, intervened_node, delta, n_timesteps, dampen=False)

    sq_error = (preds-true_vals)**2

    mean_sq_error = np.mean(sq_error)

    print(sq_error, "\n")
    print(mean_sq_error)

if __name__ == "__main__":
    evaluate_CAG(input = sys.argv[1],target_node = sys.argv[2], intervened_node
            = sys.argv[3], start_year = int(sys.argv[4]), end_year = int(
                sys.argv[5]),
            res = int(sys.argv[6]))
