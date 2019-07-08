from typing import Dict, Optional, Union, Callable, Tuple, List, Iterable
from tqdm import trange
import pickle
import pandas as pd
from .db import engine
import numpy as np
import seaborn as sns
import warnings

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
    variable: str,
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
        variable: Name of the target indicator variable.

        country: Specified Country to get a value for.

        state: Specified State to get value for.

        year: Specified Year to get a value for.

        month: Specified Month to get a value for.

        unit: Specified Units to get a value for.

    Returns:
        Specified float value given the specified parameters.
    """

    query_base = " ".join(
        [f"select * from indicator", f"where `Variable` like '{variable}'"]
    )

    query_parts = {"base": query_base}

    if country is not None:
        check_q = query_parts["base"] + f"and `Country` is '{country}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected Country not found for {variable}! Using default settings (South Sudan)!"
            )
            query_parts["country"] = f"and `Country` is 'South Sudan'"
        else:
            query_parts["country"] = f"and `Country` is '{country}'"
    if state is not None:
        check_q = query_parts["base"] + f"and `State` is '{state}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected State not found for {variable}! Using default settings (Aggregration over all States)"
            )
            query_parts["state"] = ""
        else:
            query_parts["state"] = f"and `State` is '{state}'"

    if unit is not None:
        check_q = query_parts["base"] + f" and `Unit` is '{unit}'"
        check_r = list(engine.execute(check_q))
        if check_r == []:
            warnings.warn(
                f"Selected units not found for {variable}! Falling back to default units!"
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


def set_observed_state_from_data(G, year: int, month: int, **kwargs) -> Dict:
    """ Set the observed state for a given time point from data. See
    get_data_value() for missing data rules. Note: units are automatically set
    according to the parameterization of the given CAG.

    Args:
        G: A CAG, must have indicator variables mapped and be parameterized.

        year: An integer, designates the year (ex: 2012).

        month: An integer, designates the month (1-12).

        **kwargs: These are options for which you can specify
        country, state.

    Returns:
        Returns dictionary object representing observed state.
    """

    country = kwargs.get("country", "South Sudan")
    state = kwargs.get("state")

    return {
        n[0]: {
            indicator.name: get_data_value(
                indicator.name, country, state, year, month, indicator.unit
            )
            for indicator in n[1]["indicators"].values()
        }
        for n in G.nodes(data=True)
    }


def set_observed_state_sequence_from_data(
    G,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    **kwargs,
) -> None:
    """ Set the observed state sequence for a given time range from data. See
    get_data_value() for missing data rules. Note: units are automatically set
    according to the parameterization of the given CAG.

    Args:
        G: A CAG, must have indicator variables mapped and be parameterized.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        **kwargs: These are options for which you can specify
        country, state.

    Returns:
        None, just sets the CAGs observed_state_sequence variable.
    """

    n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )
    G.observed_state_sequence = []
    year = start_year
    month = start_month
    for i in range(n_timesteps + 1):
        G.observed_state_sequence.append(
            set_observed_state_from_data(G, year, month, **kwargs)
        )

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1


def set_latent_state_sequence(
    G, start_year: int, start_month: int, end_year: int, end_month: int
) -> None:
    """ Set the latent state sequence for a given time range. WARNING: This
    definition is still under construction and currently runs an alternative
    proxy procedure to set the latent state sequence. The proxy procedure
    requires that G.sample_from_prior() has been ran.

    Args:
        G: A CAG, must have indicator variables mapped and be parameterized. As
        noted G.sample_from_prior() must of been called as well.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

    Returns:
        None, just sets the CAGs latent_state_sequence variable.
    """
    G.s0 = G.construct_default_initial_state()
    for edge in G.edges(data=True):
        G.s0[f"∂({edge[0]})/∂t"] = 0.1 * np.random.rand()

    G.n_timesteps = calculate_timestep(
        start_year, start_month, end_year, end_month
    )

    A = G.transition_matrix_collection[0]

    G.set_latent_state_sequence(A)


# ==========================================================================
# Evaluation and inference functions
# ==========================================================================


def data_to_df(
    variable: str,
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
        vals[j] = get_data_value(variable, country, state, year, month, unit)
        date.append(f"{year}-{month}")

        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1

    return pd.DataFrame(vals, date, columns=[variable + "(True)"])


def train_model(
    G,
    start_year: int = 2012,
    start_month: Optional[int] = None,
    end_year: int = 2017,
    end_month: int = 12,
    res: int = 200,
    **kwargs,
) -> None:
    """ Trains a prediction model given a CAG with indicators.

    Args:
        G: A CAG. It Must have indicators variables mapped to nodes.

        start_year: An integer, designates the starting year (ex: 2012).

        start_month: An integer, starting month (1-12).

        end_year: An integer, ending year.

        end_month: An integer, ending month.

        res: Sampling resolution. Default is 200 samples.

        **kwargs: These are options for which you can specify
        country, state, units, fallback aggregation axes (fallback_aggaxes),
        aggregation function (aggfunc).

    Returns:
        None, sets training variables for CAG.
    """

    country = kwargs.get("country", "South Sudan")
    state = kwargs.get("state")
    units = kwargs.get("units")
    fallback_aggaxes = kwargs.get("fallback_aggaxes", ["year", "month"])
    aggfunc = kwargs.get("aggfunc", np.mean)

    G.res = res
    G.sample_from_prior()
    try:
        G.parameterize(
            country=country,
            state=state,
            year=start_year,
            month=start_month,
            units=units,
            fallback_aggaxes=fallback_aggaxes,
            aggfunc=aggfunc,
        )
    except KeyError:
        print(
            "Key Error most likely caused from passing incomplete",
            "Causal Analysis Graph (CAG). Ensure that the passed CAG",
            "has indicator variables mapped to nodes by calling",
            "<CAG>.map_concepts_to_indicators().",
        )
        raise


    if start_month is None:
        start_month = 1

    set_observed_state_sequence_from_data(
        G, start_year, start_month, end_year, end_month, **kwargs
    )

    set_latent_state_sequence(G, start_year, start_month, end_year, end_month)

    A = G.transition_matrix_collection[0]
    for edge in G.edges(data=True):
        A[f"∂({edge[0]})/∂t"][edge[1]] = 0.0

    n_samples: int = 10000
    for i, _ in enumerate(trange(n_samples)):
        if i >= (n_samples - G.res):
            G.transition_matrix_collection[
                i - (n_samples - G.res)
            ] = G.sample_from_posterior(A).copy()
        else:
            G.sample_from_posterior(A)


def evaluate(
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


# ==========================================================================
# Interventions: This section is under construction
# ==========================================================================


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

    pred = np.zeros(n_timesteps + 1)
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

    pred = np.roll(pred, 1)
    pred[0] = list(G.nodes(data=True)[target_node]["indicators"].values())[
        0
    ].mean
    return pd.DataFrame(pred, columns=[target_indicator + "(Predictions)"])


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
