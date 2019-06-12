import sys
import pickle
import .AnalysisGraph as AG
import pandas as pd
from .db import engine
import numpy as np
import random
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def get_predictions(G, target_node, intervened_node, deltas, n_timesteps):
    """ Get predicted values for each timestep for a target node's indicator
    variable given a intervened node and a set of deltas.

    Args:
        G: A completely parameterized and quantified CAG with indicators,
        estimated transition matrx, and indicator values.

        target_node: A string of the full name of the node in which we
        wish to predict values for its attached indicator variable.

        intervened_node: A string of the full name of the node in which we
        are intervening on.

        deltas: 1D array-like, contains rate of change (deltas) for each
        time step. Its length must match equal n_timesteps.

        n_timesteps: Number of times steps.

    Returns:
        Pandas Dataframe containing predictions.
    """
    assert (len(deltas) == n_timesteps), "The length of deltas must be equal to n_timesteps"
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({intervened_node})/∂t"] = deltas[0]
    s0.to_csv("bmi_config.txt", index_label="variable")
    G.initialize()

    pred = np.zeros(n_timesteps)
    target_indicator = list(G.nodes(data=True)[target_node]['indicators'].keys())[0]
    for t in range(n_timesteps):
        if t == 0:
            G.update(dampen=False)
        else:
            G.update(dampen=False,set_delta = deltas[t])
        pred[t]=np.median(list(G.nodes(data=True)[target_node]['indicators'].values())[0].samples)

    return pd.DataFrame(pred,columns=[target_indicator+'(Predictions)'])


def get_true_values(G,target_node,n_timesteps,start_year,start_month):
    """ Get the true values of the indicator variable attached to the given
    target node.

    Args:
        G: A completely parameterized and quantified CAG with indicators,
        estimated transition matrx, and indicator values.

        target_node: A string of the full name of the node in which we
        wish to predict values for its attached indicator variable.

        n_timesteps: Number of times steps.

        start_year: An integer, designates the starting year in which to obtain
        values.

        start_month: An integer, starting month (1-12)

    Returns:
        Pandas Dataframe containing true values for target node's indicator
        variable. The values are indexed by date.
    """
    df = pd.read_sql_table("indicator", con=engine)
    target_indicator = list(G.nodes(data=True)[target_node]['indicators'].keys())[0]
    target_df = df[df['Variable'] == target_indicator]

    true_vals = np.zeros(n_timesteps)
    year = start_year
    month = start_month+1
    date = []
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

        date.append(str(year)+'-'+str(month))

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return pd.DataFrame(true_vals,date,columns=[target_indicator+'(True)'])


def calculate_timestep(start_year,start_month,end_year,end_month):
    """ Utility function that converts a time range given a start date and end
    date into a integer value.

    Args:
        start_year: An integer, designates the starting year (ex: 2012)

        start_month: An integer, starting month (1-12)

        end_year: An integer, ending year

        end_month: An integer, ending month

    Returns:
        Integer value
    """

    assert (start_year <= end_year), "Starting date cannot exceed ending date."
    if start_year == end_year:
        assert (start_month <= end_month), "Starting date cannot exceed ending date."

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

    diff_val_per = (end_val - start_val)/start_val
    return diff_val_per/n_timesteps

def estimate_deltas(G,intervened_node,n_timesteps,start_year,start_month):
    df = pd.read_sql_table("indicator",con=engine)
    intervener_indicator = list(G.nodes(data=True)[intervened_node]['indicators'].keys())[0]
    intervened_df = df[df['Variable'] == intervener_indicator]

    int_vals = np.zeros(n_timesteps+1)
    year = start_year
    month = start_month
    for j in range(n_timesteps+1):
        if intervened_df[intervened_df['Year'] == year].empty:
           int_vals[j] = intervened_df['Value'].values.astype(float).mean()
        elif intervened_df[intervened_df['Year'] == year][intervened_df['Month'] ==
                month].empty:
            int_vals[j] = intervened_df[intervened_df['Year'] ==
                    year]['Value'].values.astype(float).mean()
        else:
            int_vals[j] = intervened_df[intervened_df['Year'] ==
                    year][intervened_df['Month']
                == month]['Value'].values.astype(float).mean()

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    per_ch = (np.roll(int_vals,-1) - int_vals)

    per_ch = per_ch/int_vals

    per_mean = np.abs(np.mean(per_ch[np.isfinite(per_ch)]))

    per_ch[np.isnan(per_ch)] = 0
    per_ch[np.isposinf(per_ch)] = per_mean
    per_ch[np.isneginf(per_ch)] = -per_mean

    return np.delete(per_ch,-1)


def evaluate_CAG(
    G,
    target_node: str,
    intervened_node: str,
    input = None,
    start_year=2012,
    start_month=None,
    end_year=2017,
    end_month=None,
    dampen=False,
):
    if input is not None:
        with open(input,"rb") as f:
            G = pickle.load(f)

    G.parameterize(year = start_year,month = start_month)
    G.get_timeseries_values_for_indicators()

    if start_month is None:
        start_month = 1
    if end_month is None:
        end_month = 1

    n_timesteps = calculate_timestep(start_year,start_month,end_year,end_month)

    true_vals_df = get_true_values(G,target_node,n_timesteps,start_year,start_month)

    true_vals = true_vals_df.values

    deltas = estimate_deltas(G,intervened_node,n_timesteps,start_year,start_month)

    preds_df = get_predictions(G, target_node, intervened_node, deltas, n_timesteps, dampen=False)

    preds_df = preds_df.set_index(true_vals_df.index)

    preds = preds_df.values

    sq_error = (preds-true_vals)**2

    mean_sq_error = np.mean(sq_error)

    compare_df = pd.concat([true_vals_df,preds_df],axis=1,join_axes=[true_vals_df.index])

    sns.set(rc={'figure.figsize':(15,8)},style='whitegrid')
    ax = sns.lineplot(data=compare_df)
    ax.set_xticklabels(compare_df.index,rotation=45,ha='right',fontsize=8)

if __name__ == "__main__":
    evaluate_CAG(input = sys.argv[1],target_node = sys.argv[2], intervened_node
            = sys.argv[3], start_year = int(sys.argv[4]), end_year = int(
                sys.argv[5]),
            res = int(sys.argv[6]))
