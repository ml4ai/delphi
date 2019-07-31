from typing import Dict, Optional, Union, Callable, Tuple, List, Iterable
from tqdm import trange
import pickle
import pandas as pd
from scipy import stats
from .db import engine
import numpy as np
import seaborn as sns
import warnings


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


# ==========================================================================
# Utility functions
# ==========================================================================


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


def get_data_value(
    indicator: str,
    country: Optional[str] = "South Sudan",
    state: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    unit: Optional[str] = None,
) -> float:
    """ Get a indicator value from the delphi database.

    If multiple data entries exist for one time point then the mean is used
    for those time points. If there are no data entries for a given year,
    then the mean over all data values is used. If there are no data entries
    for a given month (but there are data entries for that year), then the
    overall value for the year is used (the mean for that year is used if there
    are multiple values for that year). Default settings are used if state,
    country, or unit are not found. WARNING: All specifications should be
    same as what was passed to G.parameterize() or else you could get mismatched data.

    Args:
        indicator: Name of the target indicator variable.

        country: Specified Country to get a value for.

        state: Specified State to get value for.

        year: Specified Year to get a value for.

        month: Specified Month to get a value for.

        unit: Specified Units to get a value for.

    Returns:
        Specified float value given the specified parameters.
    """

    query_base = " ".join(
        [f"select * from indicator", f"where `Variable` like '{indicator}'"]
    )

    query_parts = {"base": query_base}

    if country is not None:
        check_q = query_parts["base"] + f"and `Country` is '{country}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected Country not found for {indicator}! Using default settings (South Sudan)!"
            )
            query_parts["country"] = f"and `Country` is 'South Sudan'"
        else:
            query_parts["country"] = f"and `Country` is '{country}'"
    if state is not None:
        check_q = query_parts["base"] + f"and `State` is '{state}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected State not found for {indicator}! Using default settings (Aggregration over all States)"
            )
            query_parts["state"] = ""
        else:
            query_parts["state"] = f"and `State` is '{state}'"

    if unit is not None:
        check_q = query_parts["base"] + f" and `Unit` is '{unit}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected units not found for {indicator}! Falling back to default units!"
            )
            query_parts["unit"] = ""
        else:
            query_parts["unit"] = f"and `Unit` is '{unit}'"

    query_parts["year"] = f"and `Year` is '{year}'"
    query_parts["month"] = f"and `Month` is '{month}'"

    query = " ".join(query_parts.values())
    results = list(engine.execute(query))

    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        val = np.mean(
            [float(r["Value"]) for r in results if r["Unit"] == unit]
        )
        return val

    query_parts["month"] = ""
    query = " ".join(query_parts.values())
    results = list(engine.execute(query))

    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        val = np.mean(
            [float(r["Value"]) for r in results if r["Unit"] == unit]
        )
        return val

    query_parts["year"] = ""
    query = " ".join(query_parts.values())
    results = list(engine.execute(query))

    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        val = np.mean(
            [float(r["Value"]) for r in results if r["Unit"] == unit]
        )
        return val


# ==========================================================================
# Inference Output functions
# ==========================================================================


def data_to_df(
    indicator: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    **kwargs,
) -> pd.DataFrame:
    """ Get the true values of the indicator variable given a start date and
    end data. Allows for other specifications as well.

    If multiple data entries exist for one time point then the mean is used
    for those time points. If there are no data entries for a given year,
    then the mean over all data values is used. If there are no data entries
    for a given month (but there are data entries for that year), then the
    overall value for the year is used (the mean for that year is used if there
    are multiple values for that year). WARNING: The **kwargs parameters should
    be same as what was passed to G.parameterize() or else you could get mismatched data.

    Args:
        variable: Name of target indicator variable.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        **kwargs: These are options for which you can specify
        country, state, units.

    Returns:
        Pandas Dataframe containing true values for target node's indicator
        variable. The values are indexed by date.
    """
    country = kwargs.get("country", "South Sudan")
    state = kwargs.get("state")
    unit = kwargs.get("unit")

    n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )
    vals = np.zeros(n_timesteps + 1)
    year = start_year
    month = start_month
    date = []
    for j in range(n_timesteps + 1):
        vals[j] = get_data_value(indicator, country, state, year, month, unit)
        date.append(f"{year}-{month}")

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return pd.DataFrame(vals, date, columns=[f"{indicator}(True)"])


def pred_to_array(
    preds: Tuple[List[str], List[List[Dict[str, Dict[str, float]]]]],
    indicator: str,
) -> np.ndarray:
    """ Outputs raw predictions for a given indicator that were generated by
    generate_predictions. Each column is a time series.

    Args:
        G: A CAG. It must have indicators variables mapped to nodes and the
        inference model must have been trained and predictions generated.

        indicator: A string representing the indicator variable for which we
        want predictions printed.

    Returns:
        np.ndarray
    """
    pred_range, predictions = preds
    time_range = len(pred_range)
    m_samples = len(predictions)
    pred_raw = np.zeros((m_samples, time_range))
    for i in range(m_samples):
        for j in range(time_range):
            for _, inds in predictions[i][j].items():
                if indicator in inds.keys():
                    pred_raw[i][j] = float(inds[indicator])
    return pred_raw


def mean_pred_to_df(
    preds: Tuple[List[str], List[List[Dict[str, Dict[str, float]]]]],
    indicator: str,
    ci: float = 0.95,
    true_vals: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """ Outputs mean predictions for a given indicator that were generated by
    generate_predictions. The rows are indexed by date. Other output includes
    the confidence intervals for the mean predictions and with true_vals =
    True, the true data values, residual error, and error bounds. Setting
    true_vals = True, assumes that real data exists for the given prediction
    range. A heuristic estimate is calculated for each missing data value in
    the true dateset.

    Args:
        G: A CAG. It must have indicators variables mapped to nodes and the
        inference model must have been trained and predictions generated.

        indicator: A string representing the indicator variable for which we
        want mean predictions,etc printed.

        ci: Confidence Level (as decimal). Default is 0.95 or 95%.

        true_vals: A boolean, if set to True then the true data values,
        residual errors, and error bounds are return in the dataframe. If set
        to False (default), then only the mean predictions and confidence
        intervals (for the mean predictions) are returned.

        **kwargs: Here country, state, and units can be specified. The same
        kwargs (excluding k), should have been passed to train_model.

    Returns:
        np.ndarray
    """

    pred_raw = pred_to_array(preds, indicator)
    pred_range, _ = preds
    pred_stats = np.apply_along_axis(stats.bayes_mvs, 1, pred_raw.T, ci)[:, 0]
    pred_mean = np.zeros((len(pred_range), 3))
    for i, (mean, interval) in enumerate(pred_stats):
        pred_mean[i, 0] = mean
        pred_mean[i, 1] = interval[0]
        pred_mean[i, 2] = interval[1]

    mean_df = pd.DataFrame(
        pred_mean,
        columns=[
            f"{indicator}(Mean Prediction)",
            f"{indicator}(Lower Confidence Bound)",
            f"{indicator}(Upper Confidence Bound)",
        ],
    )
    if true_vals == True:
        warnings.warn(
            "The selected output settings assume that real data exists "
            "for the given prediction time range. Any missing data values are "
            "filled with a heuristic estimate based on existing data."
        )
        start_date = pred_range[0]
        start_year = int(start_date[0:4])
        start_month = int(start_date[5:7])
        end_date = pred_range[-1]
        end_year = int(end_date[0:4])
        end_month = int(end_date[5:7])
        true_data_df = data_to_df(
            indicator, start_year, start_month, end_year, end_month, **kwargs
        )
        mean_df = mean_df.set_index(true_data_df.index)

        error = mean_df.values - true_data_df.values.reshape(-1, 1)
        error_df = pd.DataFrame(
            error, columns=["Error", "Lower Error Bound", "Upper Error Bound"]
        )
        error_df = error_df.set_index(true_data_df.index)

        return pd.concat(
            [mean_df, true_data_df, error_df],
            axis=1,
            join_axes=[true_data_df.index],
        )
    else:
        mean_df = mean_df.set_index(pd.Index(pred_range))
        return mean_df


def pred_plot(
    preds: Tuple[List[str], List[List[Dict[str, Dict[str, float]]]]],
    indicator: str,
    ci: float = 0.95,
    plot_type: str = "Prediction",
    save_as: Optional[str] = None,
    **kwargs,
) -> None:
    """ Creates a line plot of the mean predictions for a given indicator that were generated by
    generate_predictions. The y-axis are the indicator values(or errors) and the x-axis
    are the prediction dates. Certain settings assume that true data exists for
    the given prediction range.

    There are 3 plots types:

        -Prediction(Default): Plots just the mean prediction with confidence bounds

        -Comparison: Plots the same as Prediction, but includes a line
        representing the true data values for the given prediction range.

        -Error: Plots the residual errors between the mean prediction and true
        values along with error bounds. A reference line is included at 0.

    Args:
        G: A CAG. It must have indicators variables mapped to nodes and the
        inference model must have been trained and predictions generated.

        indicator: A string representing the indicator variable for which we
        want mean predictions,etc printed.

        ci: Confidence Level (as decimal). Default is 0.95 or 95%.

        plot_type: A string that specifies plot type. Set as 'Prediction'(default),
        'Comparison', or 'Error'.

        save_as: A string representing the path and file name in which to save
        the plot, must include extensions. If None, no figure is saved to file.

        **kwargs: Here country, state, and units can be specified. The same
        kwargs (excluding k), should have been passed to train_model.

    Returns:
        None
    """

    if plot_type == "Comparison":
        df = mean_pred_to_df(preds, indicator, ci, True, **kwargs)
        df_compare = df.drop(df.columns[[1, 2, 4, 5, 6]], axis=1)
        sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
        ax = sns.lineplot(data=df_compare, sort=False, **kwargs)
        ax.fill_between(
            x=df_compare.index,
            y1=df[f"{indicator}(Upper Confidence Bound)"].values,
            y2=df[f"{indicator}(Lower Confidence Bound)"].values,
            alpha=0.5,
            **kwargs,
        )
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Predictions vs. True values for {indicator}")
    elif plot_type == "Error":
        df = mean_pred_to_df(preds, indicator, ci, True, **kwargs)
        df_error = df.drop(df.columns[[0, 1, 2, 3, 5, 6]], axis=1)
        sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
        ax = sns.lineplot(data=df_error, sort=False, **kwargs)
        ax.fill_between(
            x=df_error.index,
            y1=df["Upper Error Bound"].values,
            y2=df["Lower Error Bound"].values,
            alpha=0.5,
            **kwargs,
        )
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
        ax.axhline(color="r")
        ax.set_title(f"Prediction Error for {indicator}")
    else:
        df = mean_pred_to_df(preds, indicator, ci, False, **kwargs)
        df_pred = df.drop(df.columns[[1, 2]], axis=1)
        sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
        ax = sns.lineplot(data=df_pred, sort=False, **kwargs)
        ax.fill_between(
            x=df_pred.index,
            y1=df[f"{indicator}(Upper Confidence Bound)"].values,
            y2=df[f"{indicator}(Lower Confidence Bound)"].values,
            alpha=0.5,
            **kwargs,
        )
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Predictions for {indicator}")

    if save_as is not None:
        fig = ax.get_figure()
        fig.savefig(save_as)


# ==========================================================================
# Evaluation and Validation functions
# ==========================================================================


def walk_forward_val(
    initial_training_window: Tuple[Tuple[int, int], Tuple[int, int]],
    end_prediction_date: Tuple[int, int],
    burn: int = 10000,
    res: int = 200,
    **kwargs,
) -> pd.DataFrame:
    training_year_start, training_month_start, training_year_end, training_month_end = (
        initial_training_window
    )

    if (training_month_start) > 12 or (training_month_start < 1):
        temp_x = training_month_start
        training_month_start = training_year_start
        training_year_start = temp_x
    if (training_month_end) > 12 or (training_month_end < 1):
        temp_x = training_month_end
        training_month_end = training_year_end
        training_year_end = temp_x

    print("test")


# ==========================================================================
# Interventions: This section is under construction
# ==========================================================================


def estimate_deltas(
    G,
    intervened_node: str,
    n_timesteps: int,
    start_year: int,
    start_month: int,
    country: Optional[str] = "South Sudan",
    state: Optional[str] = None,
):
    """ Utility function that estimates Rate of Change (deltas) for the
    intervened node per timestep. This will use the units that the CAG
    was parameterized with. WARNING: The state and country should be same as what was
    passed to G.parameterize() or else you could get mismatched data.

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

    intervener_indicator = list(
        G.nodes(data=True)[intervened_node]["indicators"].keys()
    )[0]

    query_base = " ".join(
        [
            f"select * from indicator",
            f"where `Variable` like '{intervener_indicator}'",
        ]
    )

    query_parts = {"base": query_base}

    if country is not None:
        check_q = query_parts["base"] + f"and `Country` is '{country}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected Country not found for {intervener_indicator}! Using default settings (South Sudan)"
            )
            query_parts["country"] = f"and `Country` is 'South Sudan'"
        else:
            query_parts["country"] = f"and `Country` is '{country}'"
    if state is not None:
        check_q = query_parts["base"] + f"and `State` is '{state}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected State not found for {intervener_indicator}! Using default settings (Aggregration over all States)"
            )
            query_parts["state"] = ""
        else:
            query_parts["state"] = f"and `State` is '{state}'"

    unit = list(G.nodes(data=True)[intervened_node]["indicators"].values())[
        0
    ].unit

    int_vals = np.zeros(n_timesteps + 1)
    int_vals[0] = list(
        G.nodes(data=True)[intervened_node]["indicators"].values()
    )[0].mean
    year = start_year
    month = start_month
    for j in range(1, n_timesteps + 1):
        query_parts["year"] = f"and `Year` is '{year}'"
        query_parts["month"] = f"and `Month` is '{month}'"

        query = " ".join(query_parts.values())
        results = list(engine.execute(query))

        if results != []:
            int_vals[j] = np.mean(
                [float(r["Value"]) for r in results if r["Unit"] == unit]
            )

            if month == 12:
                year = year + 1
                month = 1
            else:
                month = month + 1
            continue

        query_parts["month"] = ""
        query = " ".join(query_parts.values())
        results = list(engine.execute(query))

        if results != []:
            int_vals[j] = np.mean(
                [float(r["Value"]) for r in results if r["Unit"] == unit]
            )

            if month == 12:
                year = year + 1
                month = 1
            else:
                month = month + 1
            continue

        query_parts["year"] = ""
        query = " ".join(query_parts.values())
        results = list(engine.execute(query))

        if results != []:
            int_vals[j] = np.mean(
                [float(r["Value"]) for r in results if r["Unit"] == unit]
            )

            if month == 12:
                year = year + 1
                month = 1
            else:
                month = month + 1
            continue

    per_ch = np.roll(int_vals, -1) - int_vals

    per_ch = per_ch / int_vals

    per_mean = np.abs(np.mean(per_ch[np.isfinite(per_ch)]))

    per_ch[np.isnan(per_ch)] = 0
    per_ch[np.isposinf(per_ch)] = per_mean
    per_ch[np.isneginf(per_ch)] = -per_mean

    return np.delete(per_ch, -1)


def intervention(
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
    **kwargs,
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

        **kwargs: These are options for parameterize() which specify
        country, state, units, fallback aggregation axes, and aggregation
        function. Country and State also get passed into estimate_deltas()
        and get_true_values(). The appropriate arguments are country, state,
        units, fallback_aggaxes, and aggfunc.

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

    if "country" in kwargs:
        country = kwargs["country"]
    else:
        country = "South Sudan"
    if "state" in kwargs:
        state = kwargs["state"]
    else:
        state = None
    if "units" in kwargs:
        units = kwargs["units"]
    else:
        units = None
    if "fallback_aggaxes" in kwargs:
        fallback_aggaxes = kwargs["fallback_aggaxes"]
    else:
        fallback_aggaxes = ["year", "month"]
    if "aggfunc" in kwargs:
        aggfunc = kwargs["aggfunc"]
    else:
        aggfunc = np.mean

    G.parameterize(
        country=country,
        state=state,
        year=start_year,
        month=start_month,
        units=units,
        fallback_aggaxes=fallback_aggaxes,
        aggfunc=aggfunc,
    )
    G.get_timeseries_values_for_indicators()

    if start_month is None:
        start_month = 1
    if end_month is None:
        end_month = 1

    n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )

    true_vals_df = get_true_values(
        G, target_node, n_timesteps, start_year, start_month, country, state
    )

    true_vals = true_vals_df.values

    deltas = estimate_deltas(
        G,
        intervened_node,
        n_timesteps,
        start_year,
        start_month,
        country,
        state,
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
