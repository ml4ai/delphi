from typing import Dict, Optional, Union, Callable, Tuple, List, Iterable
import pandas as pd
from scipy import stats
from .db import engine
import numpy as np
import seaborn as sns
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    county: Optional[str] = None,
    year: int = 2015,
    month: int = 1,
    unit: Optional[str] = None,
    use_heuristic: bool = False,
) -> List[float]:
    """ Get a indicator value from the delphi database.

    Args:
        indicator: Name of the target indicator variable.

        country: Specified Country to get a value for.

        state: Specified State to get value for.

        year: Specified Year to get a value for.

        month: Specified Month to get a value for.

        unit: Specified Units to get a value for.

        use_heuristic: a boolean that indicates whether or not use a built-in
        heurstic for partially missing data. In cases where data for a given
        year exists but no monthly data, setting this to true divides the
        yearly value by 12 for any month.


    Returns:
        Specified float value given the specified parameters.
    """

    query_base = " ".join(
        [f"select * from indicator", f"where `Variable` like '{indicator}'"]
        #[f"select Country, Year, Month, avg(Value) as Value, Unit from indicator", f"where `Variable` like '{indicator}'"]
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
    else:
        query_parts["country"] = f"and `Country` is 'None'"

    if state is not None:
        check_q = query_parts["base"] + f"and `State` is '{state}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                (
                    f"Selected State not found for {indicator}! Using default ",
                    "settings (Getting data at Country level only instead of State ",
                    "level)!",
                )
            )
            query_parts["state"] = f"and `State` is 'None'"
        else:
            query_parts["state"] = f"and `State` is '{state}'"
    #else:
    #    query_parts["state"] = f"and `State` is 'None'"

    if county is not None:
        check_q = query_parts["base"] + f"and `County` is '{county}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                (
                    f"Selected County not found for {indicator}! Using default ",
                    "settings (Attempting to get data at state level instead)!",
                )
            )
            query_parts["county"] = f"and `County` is 'None'"
        else:
            query_parts["county"] = f"and `County` is '{county}'"
    #else:
    #    query_parts["county"] = f"and `County` is 'None'"

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

    #query_parts["groupby"] = f"group by `Year`, `Month`"

    query = " ".join(query_parts.values())
    #print(query)
    results = list(engine.execute(query))
    #print(results)

    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        vals = [float(r["Value"]) for r in results if r["Unit"] == unit]
        return vals

    if not use_heuristic:
        return []

    query_parts["month"] = f"and `Month` is '0'"
    query = " ".join(query_parts.values())
    results = list(engine.execute(query))

    if results != []:
        unit = sorted(list({r["Unit"] for r in results}))[0]
        vals = [float(r["Value"]) for r in results if r["Unit"] == unit]
        return list(map(lambda x: x / 12, vals))

    return []


# ==========================================================================
# Inference Output functions
# ==========================================================================


def data_to_list(
    indicator: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    use_heuristic: bool = False,
    **kwargs,
) -> Tuple[List[str], List[List[int]]]:
    """ Get the true values of the indicator variable given a start date and
    end date. Allows for other specifications as well.

    Args:
        variable: Name of target indicator variable.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        use_heuristic: a boolean that indicates whether or not use a built-in
        heurstic for partially missing data. In cases where data for a given
        year exists but no monthly data, setting this to true divides the
        yearly value by 12 for any month.

        **kwargs: These are options for which you can specify
        country, state, units.

    Returns:
        Returns a tuple where the first element is a list of the specified
        dates in year-month format and the second element is a
        list of lists of the true data for a given indicator.
        Each element of the outer list represents a time step and the inner
        lists contain the data for that time point.
    """
    country = kwargs.get("country", "South Sudan")
    state = kwargs.get("state")
    county = kwargs.get("county")
    unit = kwargs.get("unit")

    n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )
    vals = []
    year = start_year
    month = start_month
    date = []
    for j in range(n_timesteps + 1):
        vals.append(
            get_data_value(
                indicator,
                country,
                state,
                county,
                year,
                month,
                unit,
                use_heuristic,
            )
        )
        date.append(f"{year}-{month}")

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return date, vals


def mean_data_to_df(
    indicator: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    use_heuristic: bool = False,
    ci: Optional[float] = None,
    **kwargs,
) -> pd.DataFrame:
    """ Get the true values of the indicator variable given a start date and
        end data. Allows for other specifications as well.
        variable: Name of target indicator variable.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        use_heuristic: a boolean that indicates whether or not use a built-in
        heurstic for partially missing data. In cases where data for a given
        year exists but no monthly data, setting this to true divides the
        yearly value by 12 for any month.

        ci: confidence level. Only the mean is reported if left as None.

        **kwargs: These are options for which you can specify
        country, state, units.

    Returns:
        Pandas Dataframe containing true values for target node's indicator
        variable. The values are indexed by date.
    """
    date, vals = data_to_list(
        indicator,
        start_year,
        start_month,
        end_year,
        end_month,
        use_heuristic,
        **kwargs,
    )
    if ci is not None:
        data_mean = np.zeros((len(date), 3))
        for i, obs in enumerate(vals):
            if len(obs) > 1:
                mean, _, _ = stats.bayes_mvs(obs, ci)
                data_mean[i, 0] = mean[0]
                data_mean[i, 1] = mean[1][0]
                data_mean[i, 2] = mean[1][1]
            elif len(obs) == 1:
                data_mean[i, 0] = obs[0]
                data_mean[i, 1] = np.nan
                data_mean[i, 2] = np.nan
            else:
                data_mean[i, 0] = np.nan
                data_mean[i, 1] = np.nan
                data_mean[i, 2] = np.nan

        return pd.DataFrame(
            data_mean,
            date,
            columns=[
                f"{indicator}(True)",
                f"{indicator}(True)(Lower Confidence Bound)",
                f"{indicator}(True)(Upper Confidence Bound)",
            ],
        )
    else:
        data_mean = np.zeros((len(date), 1))
        for i, obs in enumerate(vals):
            data_mean[i] = np.mean(obs)

        return pd.DataFrame(data_mean, date, columns=[f"{indicator}(True)"])


def pred_to_array(
    preds: Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        List[str],
        List[List[Dict[str, Dict[str, float]]]],
    ],
    indicator: str,
) -> np.ndarray:
    """ Outputs raw predictions for a given indicator that were generated by
    generate_prediction(). Each column is a time step and the rows are the
    samples for that time step.

    Args:
        preds: This is the entire prediction set returned by the
        generate_prediction() method in AnalysisGraph.cpp.

        indicator: A string representing the indicator variable for which we
        want predictions printed.

    Returns:
        np.ndarray
    """
    _, pred_range, predictions = preds
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
    preds: Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        List[str],
        List[List[Dict[str, Dict[str, float]]]],
    ],
    indicator: str,
    ci: Optional[float] = 0.95,
    true_vals: bool = False,
    use_heuristic_for_true: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """ Outputs mean predictions for a given indicator that were generated by
    generate_prediction(). The rows are indexed by date. Other output includes
    the confidence intervals for the mean predictions and with true_vals =
    True, the true data values, residual error, and error bounds. Setting
    true_vals = True, assumes that real data exists for the given prediction
    range. A heuristic estimate is calculated for each missing data value in
    the true dateset.

    Args:
        preds: This is the entire prediction set returned by the
        generate_prediction() method in AnalysisGraph.cpp.

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

    if ci is not None:
        pred_raw = pred_to_array(preds, indicator)
        _, pred_range, _ = preds

        # WARNING: This line generates a warning!
        # VisibleDeprecationWarning: Creating an ndarray from ragged nested
        # sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays
        # with different lengths or shapes) is deprecated. If you meant to do
        # this, you must specify 'dtype=object' when creating the ndarray
        #   return array(a, dtype, copy=False, order=order, subok=True)
        # When I do a similar call at the Python command line, the warning does
        # not appear.
        # Could not figure out how to make this warning go away.
        pred_stats = np.apply_along_axis(stats.bayes_mvs, 1, pred_raw.T, ci)[
            :, 0
        ]
        pred_mean = np.zeros((len(pred_range), 3))
        for i, (mean, interval) in enumerate(pred_stats):
            pred_mean[i, 0] = mean
            pred_mean[i, 1] = interval[0]
            pred_mean[i, 2] = interval[1]

        mean_df = pd.DataFrame(
            pred_mean,
            columns=[
                f"{indicator} (Mean Prediction)",
                f"{indicator} (Lower Confidence Bound)",
                f"{indicator} (Upper Confidence Bound)",
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
            true_data_df = mean_data_to_df(
                indicator,
                start_year,
                start_month,
                end_year,
                end_month,
                use_heuristic_for_true,
                ci,
                **kwargs,
            )
            mean_df = mean_df.set_index(true_data_df.index)

            error = mean_df.values - true_data_df.values[:, 0].reshape(-1, 1)
            error_df = pd.DataFrame(
                error,
                columns=["Error", "Lower Error Bound", "Upper Error Bound"],
            )
            error_df = error_df.set_index(true_data_df.index)

            return pd.concat(
                [mean_df, true_data_df, error_df],
                axis=1,
            )
        else:
            mean_df = mean_df.set_index(pd.Index(pred_range))
            return mean_df
    else:
        pred_raw = pred_to_array(preds, indicator)
        _, pred_range, _ = preds
        pred_mean = np.mean(pred_raw, axis=0).reshape(-1, 1)

        mean_df = pd.DataFrame(
            pred_mean, columns=[f"{indicator}(Mean Prediction)"]
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
            true_data_df = mean_data_to_df(
                indicator,
                start_year,
                start_month,
                end_year,
                end_month,
                use_heuristic_for_true,
                ci,
                **kwargs,
            )
            mean_df = mean_df.set_index(pd.Index(pred_range))

            error = mean_df.values - true_data_df.values.reshape(-1, 1)
            error_df = pd.DataFrame(error, columns=["Error"])
            error_df = error_df.set_index(true_data_df.index)

            return pd.concat(
                [mean_df, true_data_df, error_df],
                axis=1,
            )
        else:
            mean_df = mean_df.set_index(pd.Index(pred_range))
            return mean_df


def calculate_prediction_rmse(
    preds: Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        List[str],
        List[List[Dict[str, Dict[str, float]]]],
    ],
    indicator: str,
    **kwargs,
) -> float:
    df = mean_pred_to_df(preds, indicator, 0.95, True, **kwargs)
    sqr_residuals = df["Error"].values ** 2
    return np.sqrt(np.nanmean(sqr_residuals))


def pred_plot(
    preds: Tuple[
        Tuple[Tuple[int, int], Tuple[int, int]],
        List[str],
        List[List[Dict[str, Dict[str, float]]]],
    ],
    indicator: str,
    ci: Optional[float] = 0.95,
    plot_type: str = "Prediction",
    show_rmse: bool = False,
    show_training_data: bool = False,
    save_as: Optional[str] = None,
    use_heuristic_for_true: bool = False,
    **kwargs,
) -> None:
    """ Creates a line plot of the mean predictions for a given indicator that were generated by
    generate_prediction(). The y-axis are the indicator values(or errors) and the x-axis
    are the prediction dates. Certain settings assume that true data exists for
    the given prediction range.

    There are 3 plots types:

        -Prediction(Default): Plots just the mean prediction with confidence bounds

        -Comparison: Plots the same as Prediction, but includes a line
        representing the true data values for the given prediction range.

        -Error: Plots the residual errors between the mean prediction and true
        values along with error bounds. A reference line is included at 0.

    Args:
        preds: This is the entire prediction set returned by the
        generate_prediction() method in AnalysisGraph.cpp.

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

    # TODO: Test option may not work properly

    df = mean_pred_to_df(
        preds,
        indicator,
        ci,
        plot_type == 'Error',
        use_heuristic_for_true,
        **kwargs,
    )
    df.index = pd.to_datetime(df.index)

    title = f'Predictions for {indicator}'
    y_label = f'{indicator}'

    if plot_type == 'Error':
        title = f"Prediction Error for {indicator}"
        y_label = 'Error'

        if ci:
            ci_upper = 'Upper Error Bound'
            ci_lower = 'Lower Error Bound'

            df_plot = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 7, 8]], axis=1)
        else:
            df_plot = df.drop(df.columns[[0, 1]], axis=1)
    else:
        if ci or plot_type == 'Test':
            ci_upper = f'{indicator} (Upper Confidence Bound)'
            ci_lower = f'{indicator} (Lower Confidence Bound)'

            df_plot = df.drop(df.columns[[1, 2]], axis=1)

            if plot_type == 'Test':
                # TODO: Test option may not work properly
                title = f"Predictions vs. True values (Synthetic) for {indicator}"

                synthetic_data = kwargs.get("test_data")
                assert test_data is not None
                df_plot[f"{indicator} (Synthetic)"] = synthetic_data
        else:
            df_plot = df

    sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")

    ax = sns.lineplot(data=df_plot, sort=False, markers=['D'])

    training_range, pred_range, _ = preds

    if plot_type == 'Comparison' and show_training_data:
        start_year = training_range[0][0]
        start_month = training_range[0][1]
    else:
        start_year = int(pred_range[0][0:4])
        start_month = int(pred_range[0][5:7])

    end_year = int(pred_range[-1][0:4])
    end_month = int(pred_range[-1][5:7])

    true_date, true_data = data_to_list(
        indicator,
        start_year,
        start_month,
        end_year,
        end_month,
        use_heuristic_for_true,
        **kwargs,
    )

    if plot_type == 'Comparison':
        title = f"Predictions vs. True values for {indicator}"

        true_data_x = []
        true_data_y = []
        for i, obs in enumerate(true_data):
            if len(obs) > 0:
                for o in obs:
                    true_data_y.append(o)
                    true_data_x.append(true_date[i])
            else:
                true_data_y.append(np.nan)
                true_data_x.append(true_date[i])

        df_true = pd.DataFrame({f'{indicator} (True)': true_data_y, 'Year-Month':
                true_data_x})

        # Aggregate multiple coinciding data points
        df_true['frequency'] = df_true['Year-Month'].apply(lambda x: 1)
        df_true_grp = df_true.groupby(by=['Year-Month', f'{indicator} (True)'], as_index=False).count()
        df_true_grp['Year-Month'] = pd.to_datetime(df_true_grp['Year-Month'])

        sns.scatterplot(
            data=df_true_grp,
            y=f'{indicator} (True)',
            x='Year-Month',
            marker='o',
            size=df_true_grp['frequency'].tolist(),
            sizes=(0, 250),
            hue=df_true_grp['frequency'].tolist(),
            palette='ch:r=-.8, l=.75',
            ax=ax,
            label=f'{indicator} (True)',
        )

    if ci is not None:
        df.dropna(inplace=True)

        ax.fill_between(
            x=df.index,
            y1=df[ci_upper],
            y2=df[ci_lower],
            alpha=0.2,
            color='blue'
        )

    if plot_type == 'Error':
        ax.axhline(color="r")

    ax.set_title(title)
    ax.set_xlabel('Year-Month')
    ax.set_ylabel(y_label)

    # Set x-axis tick marks
    ax.set_xticks(pd.to_datetime(true_date))
    ax.set_xticklabels(
        pd.to_datetime(true_date), rotation=45, ha="right", fontsize=8
    )
    xfmt = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_formatter(xfmt)

    handles, labels = ax.get_legend_handles_labels()

    ax.get_lines()[0].set_color("blue")
    ax.get_lines()[0].set_linestyle("-")
    handles[0].set_color('blue')

    if show_rmse:
        rmse = round(
            calculate_prediction_rmse(preds, indicator, **kwargs),
            4,
        )
        rmse_str = f"Root Mean Squared Error: {rmse}"

        handles.append(
            mpatches.Patch(color="none", label=rmse_str)
        )

        ax.legend(handles=handles)

    if save_as is not None:
        fig = ax.get_figure()
        fig.savefig(save_as)
        plt.close()


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
        [true_vals_df, preds_df], axis=1
    )

    if plot:
        if plot_type == "Error":
            sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
            ax = sns.lineplot(data=error_df)
            ax.set_xticks(error_df.index)
            ax.set_xticklabels(
                error_df.index, rotation=45, ha="right", fontsize=8
            )
            ax.axhline(c="red")
        else:
            sns.set(rc={"figure.figsize": (15, 8)}, style="whitegrid")
            ax = sns.lineplot(data=compare_df)
            ax.set_xticks(compare_df.index)
            ax.set_xticklabels(
                compare_df.index, rotation=45, ha="right", fontsize=8
            )

    return pd.concat(
        [compare_df, error_df], axis=1
    )
