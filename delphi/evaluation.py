from typing import List
import pickle
import pandas as pd
from .db import engine
import numpy as np
import seaborn as sns
import warnings


def get_predictions(
    G,
    target_node: str,
    intervened_node: str,
    deltas: List[float],
    n_timesteps: int,
) -> pd.DataFrame:
    """ Get predicted values for each timestep for a target node's indicator
    variable given a intervened node and a set of deltas.

    Args:
        G: A completely parameterized and quantified CAG with indicators,
        estimated transition matrx, and indicator values.

        target_node: The full name of the node in which we
        wish to predict values for its attached indicator variable.

        intervened_node: The full name of the node in which we
        are intervening on.

        deltas: 1D array-like, contains rate of change (deltas) for each
        time step. Its length must match equal n_timesteps.

        n_timesteps: Number of time steps.

    Returns:
        Pandas Dataframe containing predictions.
    """
    assert (
        len(deltas) == n_timesteps
    ), "The length of deltas must be equal to n_timesteps."
    G.create_bmi_config_file()
    s0 = pd.read_csv(
        "bmi_config.txt", index_col=0, header=None, error_bad_lines=False
    )[1]
    s0.loc[f"∂({intervened_node})/∂t"] = deltas[0]
    s0.to_csv("bmi_config.txt", index_label="variable")
    G.initialize()

    pred = np.zeros(n_timesteps)
    target_indicator = list(
        G.nodes(data=True)[target_node]["indicators"].keys()
    )[0]
    for t in range(n_timesteps):
        if t == 0:
            G.update(dampen=False)
        else:
            G.update(dampen=False, set_delta=deltas[t])
        pred[t] = np.median(
            list(G.nodes(data=True)[target_node]["indicators"].values())[
                0
            ].samples
        )

    return pd.DataFrame(pred, columns=[target_indicator + "(Predictions)"])


def get_true_values(
    G, target_node: str, n_timesteps: str, start_year: int, start_month: int
) -> pd.DataFrame:
    """ Get the true values of the indicator variable attached to the given
    target node.

    If multiple data entries exist for one time point then the mean is used
    for those time points. If there are no data entries for a given year,
    then the mean over all data values is used. If there are no data entries
    for a given month (but there are data entries for that year), then the
    overall value for the year is used (the mean for that year is used if there
    are multiple values for that year).

    Args:
        G: A completely parameterized and quantified CAG with indicators,
        estimated transition matrx, and indicator values.

        target_node: The full name of the node in which we
        wish to predict values for its attached indicator variable.

        n_timesteps: Number of time steps.

        start_year: The starting year in which to obtain
        values.

        start_month: The starting month (1-12).

    Returns:
        Pandas Dataframe containing true values for target node's indicator
        variable. The values are indexed by date.
    """
    df = pd.read_sql_table("indicator", con=engine)
    target_indicator = list(
        G.nodes(data=True)[target_node]["indicators"].keys()
    )[0]
    target_df = df[df["Variable"] == target_indicator]

    true_vals = np.zeros(n_timesteps)
    year = start_year
    month = start_month + 1
    date = []
    for j in range(n_timesteps):
        if target_df[target_df["Year"] == year].empty:
            true_vals[j] = target_df["Value"].values.astype(float).mean()
        elif target_df[target_df["Year"] == year][
            target_df["Month"] == month
        ].empty:
            true_vals[j] = (
                target_df[target_df["Year"] == year]["Value"]
                .values.astype(float)
                .mean()
            )
        else:
            true_vals[j] = (
                target_df[target_df["Year"] == year][
                    target_df["Month"] == month
                ]["Value"]
                .values.astype(float)
                .mean()
            )

        date.append(str(year) + "-" + str(month))

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return pd.DataFrame(true_vals, date, columns=[target_indicator + "(True)"])


def calculate_timestep(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> int:
    """ Utility function that converts a time range given a start date and end
    date into a integer value.

    Args:
        start_year: The starting year (ex: 2012)

        start_month: Starting month (1-12)

        end_year: Ending year

        end_month: Ending month

    Returns:
        The computed time step.
    """

    assert start_year <= end_year, "Starting date cannot exceed ending date."
    if start_year == end_year:
        assert (
            start_month <= end_month
        ), "Starting date cannot exceed ending date."

    diff_year = end_year - start_year
    year_to_month = diff_year * 12
    return year_to_month - (start_month - 1) + (end_month - 1)


def estimate_deltas(
    G,
    intervened_node: str,
    n_timesteps: int,
    start_year: int,
    start_month: int,
):
    """ Utility function that estimates Rate of Change (deltas) for the
    intervened node per timestep.

    Deltas are estimated by percent change between each time step. (i.e,
    (current - next)/current). Heuristics are in place to handle NAN and INF
    values. If changed from 0 to 0 (NAN case), then delta = 0. If increasing
    from 0 (+INF case), then delta = positive absolute mean of all finite
    deltas. If decreasing from 0 (-INF case), then delta = negative absolute
    mean of all finite deltas.

    See function get_true_values to see how the data is aggregated to fill in
    values for missing time points which calculating the deltas.

    Args:
        G: A completely parameterized and quantified CAG with indicators,
        estimated transition matrx, and indicator values.

        intervened_node: A string of the full name of the node in which we
        are intervening on.

        n_timesteps: Number of time steps.

        start_year: The starting year (e.g, 2012).

        start_month: The starting month (1-12).

    Returns:
        1D numpy array of deltas.
    """

    df = pd.read_sql_table("indicator", con=engine)
    intervener_indicator = list(
        G.nodes(data=True)[intervened_node]["indicators"].keys()
    )[0]
    intervened_df = df[df["Variable"] == intervener_indicator]

    int_vals = np.zeros(n_timesteps + 1)
    year = start_year
    month = start_month
    for j in range(n_timesteps + 1):
        if intervened_df[intervened_df["Year"] == year].empty:
            int_vals[j] = intervened_df["Value"].values.astype(float).mean()
        elif intervened_df[intervened_df["Year"] == year][
            intervened_df["Month"] == month
        ].empty:
            int_vals[j] = (
                intervened_df[intervened_df["Year"] == year]["Value"]
                .values.astype(float)
                .mean()
            )
        else:
            int_vals[j] = (
                intervened_df[intervened_df["Year"] == year][
                    intervened_df["Month"] == month
                ]["Value"]
                .values.astype(float)
                .mean()
            )

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    per_ch = np.roll(int_vals, -1) - int_vals

    per_ch = per_ch / int_vals

    per_mean = np.abs(np.mean(per_ch[np.isfinite(per_ch)]))

    per_ch[np.isnan(per_ch)] = 0
    per_ch[np.isposinf(per_ch)] = per_mean
    per_ch[np.isneginf(per_ch)] = -per_mean

    return np.delete(per_ch, -1)


def setup_evaluate(G=None, input=None, res=200):
    """ Optional Utility function that takes a CAG and assembles the transition
    model.

    Args:
        G: A CAG. It Must have indicators variables mapped to nodes.

        input: This allows you to upload a CAG from a pickle file, instead of
        passing it directly as an argument. The CAG must have mapped
        indicators.

        res: Sampling resolution. Default is 200 samples.

    Returns:
        Returns CAG.
    """

    if input is not None:
        if G is not None:
            warnings.warn(
                "The CAG passed to G will be suppressed by the CAG loaded from "
                "the pickle file."
            )
        with open(input, "rb") as f:
            G = pickle.load(f)

    assert G is not None, (
        "A CAG must be passed to G or a pickle file containing a CAG must be "
        "passed to input."
    )
    G.res = res
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    return G


def evaluate(
    target_node: str,
    intervened_node: str,
    G=None,
    input=None,
    start_year=2012,
    start_month=None,
    end_year=2017,
    end_month=None,
    plot=False,
    plot_type="Compare",
):
    """ This is the main function of this module. This parameterizes a given
    CAG (see requirements in Args) and calls other functions within this module
    to predict values for a specified target node's indicator variable given a
    start date and end date. Returns pandas dataframe containing predicted
    values, true values, and error.

    Args:
        target_node: A string of the full name of the node in which we
        wish to predict values for its attached indicator variable.

        intervened_node: A string of the full name of the node upon which we
        are intervening.

        G: A CAG. It must have mapped indicator values and estimated transition
        matrix.

        input: This allows you to upload a CAG from a pickle file, instead of
        passing it directly as an argument. The CAG must have mapped
        indicators and an estimated transition matrix.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        plot: Set to true to display a plot according to the plot type.

        plot_type: By default setting plot to true displays the "Compare" type
        plot which plots the predictions and true values of the target node's
        indicator variable on one plot labeled by time steps (Year-Month).
        There is also "Error" type, which plots the errors or residuals, with a
        reference line at 0.

    Returns:
        Returns a pandas dataframe.
    """

    if input is not None:
        if G is not None:
            warnings.warn(
                "The CAG passed to G will be suppressed by the CAG loaded from "
                "the pickle file."
            )
        with open(input, "rb") as f:
            G = pickle.load(f)

    assert G is not None, (
        "A CAG must be passed to G or a pickle file containing a CAG must be "
        "passed to input."
    )

    G.parameterize(year=start_year, month=start_month)
    G.get_timeseries_values_for_indicators()

    if start_month is None:
        start_month = 1
    if end_month is None:
        end_month = 1

    n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )

    true_vals_df = get_true_values(
        G, target_node, n_timesteps, start_year, start_month
    )

    true_vals = true_vals_df.values

    deltas = estimate_deltas(
        G, intervened_node, n_timesteps, start_year, start_month
    )

    preds_df = get_predictions(
        G, target_node, intervened_node, deltas, n_timesteps
    )

    preds_df = preds_df.set_index(true_vals_df.index)

    preds = preds_df.values

    error = preds - true_vals

    error_df = pd.DataFrame(error, columns=["Errors"])

    error_df = error_df.set_index(true_vals_df.index)

    compare_df = pd.concat(
        [true_vals_df, preds_df], axis=1, join_axes=[true_vals_df.index]
    )

    if plot:
        if plot_type == "Error":
            sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
            ax = sns.lineplot(data=error_df)
            ax.set_xticklabels(
                error_df.index, rotation=45, ha="right", fontsize=8
            )
            ax.axhline(c="red")
        else:
            sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
            ax = sns.lineplot(data=compare_df)
            ax.set_xticklabels(
                compare_df.index, rotation=45, ha="right", fontsize=8
            )

    return pd.concat(
        [compare_df, error_df], axis=1, join_axes=[true_vals_df.index]
    )
