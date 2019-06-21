import delphi.evaluation as EN
import delphi.AnalysisGraph as AG
import pytest
import numpy as np
import random
from conftest import G_eval
import pickle
import os


# Set seeds
np.random.seed(2019)
random.seed(2019)

# Testing definitions


def test_get_predictions(G_eval):
    # Checking assertion error when putting in len(deltas) != n_timesteps
    deltas = np.random.rand(10)
    n_timesteps = 9
    with pytest.raises(AssertionError) as excinfo:
        pred_df = EN.get_predictions(
            G_eval,
            "UN/entities/natural/crop_technology/product",
            "UN/events/human/human_migration",
            deltas,
            n_timesteps,
        )
    assert "The length of deltas must be equal to n_timesteps." in str(
        excinfo.value
    )

    # Checking number of rows, column titles
    deltas = np.random.rand(10)
    n_timesteps = 10
    pred_df = EN.get_predictions(
        G_eval,
        "UN/entities/natural/crop_technology/product",
        "UN/events/human/human_migration",
        deltas,
        n_timesteps,
    )
    assert len(pred_df) == n_timesteps + 1
    assert pred_df.columns[0] == "Import Quantity, Infant food(Predictions)"


def test_get_true_values(G_eval):
    G_eval.parameterize(year=2011, month=9)
    G_eval.get_timeseries_values_for_indicators()

    # Checking number of rows, column title, and index names
    true_df = EN.get_true_values(
        G_eval, "UN/entities/natural/crop_technology/product", 48, 2011, 9
    )
    assert len(true_df) == 49
    assert true_df.columns[0] == "Import Quantity, Infant food(True)"
    assert true_df.index[6] == "2012-3"

    # Checking aggregation heuristics
    G = AG.AnalysisGraph.from_text(
        "Improved Rainfall causes decreased inflation rates",
        webservice="http://54.84.114.146:9000",
    )
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2013, month=9)
    G.get_timeseries_values_for_indicators()

    true_df = EN.get_true_values(
        G, "UN/entities/human/financial/economic/inflation", 10, 2013, 9
    )
    assert len(true_df) == 11
    assert true_df.columns[0] == "Inflation Rate(True)"
    assert true_df.index[6] == "2014-3"


def test_calculate_timestep():
    # Check Assertion error when putting end_year < start_year
    start_year = 2017
    end_year = 2013
    start_month = 6
    end_month = 7
    with pytest.raises(AssertionError) as excinfo:
        EN.calculate_timestep(start_year, start_month, end_year, end_month)
    assert "Starting date cannot exceed ending date." in str(excinfo.value)

    # Check Assertion error when putting end_month < start_month when start_year == end_year
    start_year = 2017
    end_year = 2017
    start_month = 8
    end_month = 5
    with pytest.raises(AssertionError) as excinfo:
        EN.calculate_timestep(start_year, start_month, end_year, end_month)
    assert "Starting date cannot exceed ending date." in str(excinfo.value)

    # Check to see if it accurately produces timestep
    start_year = 2010
    end_year = 2020
    start_month = 5
    end_month = 2
    assert (
        EN.calculate_timestep(start_year, start_month, end_year, end_month)
        == 117
    )


def test_estimate_deltas(G_eval):
    G_eval.parameterize(year=2011, month=9)
    G_eval.get_timeseries_values_for_indicators()

    # Checking length
    deltas = EN.estimate_deltas(
        G_eval, "UN/entities/natural/crop_technology/product", 48, 2011, 9
    )
    assert len(deltas) == 48

    # Checking aggregation heuristics
    G = AG.AnalysisGraph.from_text(
        "Improved Rainfall causes decreased inflation rates",
        webservice="http://54.84.114.146:9000",
    )
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2013, month=9)
    G.get_timeseries_values_for_indicators()

    deltas = EN.estimate_deltas(
        G, "UN/entities/human/financial/economic/inflation", 10, 2013, 9
    )
    assert len(deltas) == 10


def test_setup_evaluate(G_eval):

    # Check Empty input Assertion Error
    with pytest.raises(AssertionError) as excinfo:
        EN.setup_evaluate(res=200)
    assert (
        "A CAG must be passed to G or a pickle file containing a CAG must be passed to input."
        in str(excinfo.value)
    )

    # Check pickle file upload and warning when also passing G along with pickle file
    with open("test_CAG.pkl", "wb") as f:
        pickle.dump(G_eval, f)
    with pytest.warns(
        UserWarning,
        match="The CAG passed to G will be suppressed by the CAG loaded from the pickle file.",
    ):
        EN.setup_evaluate(G_eval, input="test_CAG.pkl", res=200)
    os.remove("test_CAG.pkl")


def test_evaluate(G_eval):
    target_node = "UN/entities/natural/crop_technology/product"
    intervened_node = "UN/events/human/human_migration"
    start_year = 2013
    start_month = 9
    end_year = 2017
    end_month = 9

    # Check empty input Assertion Error
    with pytest.raises(AssertionError) as excinfo:
        EN.evaluate(
            target_node=target_node,
            intervened_node=intervened_node,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        )
    assert (
        "A CAG must be passed to G or a pickle file containing a CAG must be passed to input."
        in str(excinfo.value)
    )

    # Check pickle file upload and warning when also passing G along with pickle file
    with open("test_CAG.pkl", "wb") as f:
        pickle.dump(G_eval, f)
    with pytest.warns(
        UserWarning,
        match="The CAG passed to G will be suppressed by the CAG loaded from the pickle file.",
    ):
        EN.evaluate(
            target_node=target_node,
            intervened_node=intervened_node,
            G=G_eval,
            input="test_CAG.pkl",
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        )
    os.remove("test_CAG.pkl")

    # Check start_month = None, end_month = None case and plotting option
    start_month = None
    end_month = None
    df = EN.evaluate(
        target_node=target_node,
        intervened_node=intervened_node,
        G=G_eval,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        plot=True,
    )
    assert len(df) == 49
    assert df.index[0] == "2013-1"
    assert df.index[48] == "2017-1"

    # Check plotting Error plot
    EN.evaluate(
        target_node=target_node,
        intervened_node=intervened_node,
        G=G_eval,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        plot=True,
        plot_type="Error",
    )
