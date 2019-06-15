import delphi.evaluation as EN
import delphi.AnalysisGraph as AG
import pytest
import numpy as np
import random
import pandas as pd
import pickle


#Set seeds
np.random.seed(2019)
random.seed(2019)

#Testing definitions

def test_get_predictions():
    G = AG.AnalysisGraph.from_text('Improved migration causes increased product', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2013, month=9)
    G.get_timeseries_values_for_indicators()

    #Checking Assertion error when putting in len(deltas) != n_timesteps
    deltas = np.random.rand(10)
    n_timesteps = 9
    with pytest.raises(AssertionError) as excinfo:
        pred_df = EN.get_predictions(G,"UN/entities/natural/crop_technology/product","UN/events/human/human_migration", deltas, n_timesteps)
    assert "The length of deltas must be equal to n_timesteps." in str(excinfo.value)

    #Checking number of rows, column titles
    deltas = np.random.rand(10)
    n_timesteps = 10
    pred_df = EN.get_predictions(G,"UN/entities/natural/crop_technology/product","UN/events/human/human_migration", deltas, n_timesteps)
    assert len(pred_df) == n_timesteps
    assert pred_df.columns[0] == 'Import Quantity, Infant food(Predictions)'


def test_get_true_values():
    G = AG.AnalysisGraph.from_text('Improved migration causes increased product', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2011, month=9)
    G.get_timeseries_values_for_indicators()

    #Checking number of rows, column title, and index names
    true_df = EN.get_true_values(G,"UN/entities/natural/crop_technology/product",48,2011,9)
    assert len(true_df) == 48
    assert true_df.columns[0] == 'Import Quantity, Infant food(True)'
    assert true_df.index[6] == '2012-4'

    #Checking aggregation heuristics
    G = AG.AnalysisGraph.from_text('Improved Rainfall causes decreased inflation rates', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2013, month=9)
    G.get_timeseries_values_for_indicators()


    true_df = EN.get_true_values(G,"UN/entities/human/financial/economic/inflation",10,2013,9)
    assert len(true_df) == 10
    assert true_df.columns[0] == 'Inflation Rate(True)'
    assert true_df.index[6] == '2014-4'



def test_calculate_timestep():
    #Check Assertion error when putting end_year < start_year
    start_year = 2017
    end_year = 2013
    start_month = 6
    end_month = 7
    with pytest.raises(AssertionError) as excinfo:
       EN.calculate_timestep(start_year,start_month,end_year,end_month)
    assert "Starting date cannot exceed ending date." in str(excinfo.value)

    #Check Assertion error when putting end_month < start_month when start_year == end_year 
    start_year = 2017
    end_year = 2017
    start_month = 8
    end_month = 5
    with pytest.raises(AssertionError) as excinfo:
       EN.calculate_timestep(start_year,start_month,end_year,end_month)
    assert "Starting date cannot exceed ending date." in str(excinfo.value)

    #Check to see if it accurately produces timestep
    start_year = 2010
    end_year = 2020
    start_month = 5
    end_month = 2
    assert EN.calculate_timestep(start_year,start_month,end_year,end_month) == 117


def test_estimate_deltas():
    G = AG.AnalysisGraph.from_text('Improved migration causes increased product', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2011, month=9)
    G.get_timeseries_values_for_indicators()

    #Checking length 
    deltas = EN.estimate_deltas(G,"UN/entities/natural/crop_technology/product",48,2011,9)
    assert len(deltas) == 48

    #Checking aggregation heuristics
    G = AG.AnalysisGraph.from_text('Improved Rainfall causes decreased inflation rates', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()
    G.res = 200
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.parameterize(year=2013, month=9)
    G.get_timeseries_values_for_indicators()


    deltas = EN.estimate_deltas(G,"UN/entities/human/financial/economic/inflation",10,2013,9)
    assert len(deltas) == 10


def test_setup_evaluate():
    G = AG.AnalysisGraph.from_text('Improved migration causes increased product', webservice='http://54.84.114.146:9000')
    G.map_concepts_to_indicators()

    #Check Empty input Assertion Error
    with pytest.raises(AssertionError) as excinfo:
        EN.setup_evaluate(res=200)
    assert "A CAG must be passed to G or a pickle file containing a CAG must be passed to input." in str(excinfo.value)

    #Check pickle file upload and warning when also passing G along with pickle file
    with pytest.warns(UserWarning, match='The CAG passed to G will be suppressed by the CAG loaded from the pickle file.'):
        EN.setup_evaluate(G,input='data/evaluation/test_CAG.pkl',res=200)
