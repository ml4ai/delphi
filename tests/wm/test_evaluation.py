import delphi.evaluation as EN
import delphi.AnalysisGraph as AG
import pytest
import numpy as np
import random
from conftest import G_unit
import pickle
import os


# Set seeds
np.random.seed(2019)
random.seed(2019)

# Testing prediction workflow


def test_pred(G_unit):

    EN.train_model(G_unit, 2015, 1, 2015, 12, 1000, 1000, k=1)
    EN.generate_predictions(G_unit, 2016, 1, 2016, 12)
    pred = EN.pred_to_array(G_unit, "Net migration")
    print(pred)

    pred_df = EN.mean_pred_to_df(G_unit, "Net migration")

    print(pred_df)

    EN.pred_plot(G_unit, "Net migration")


def test_pred_compare(G_unit):

    EN.train_model(G_unit, 2015, 1, 2015, 12, 1000, 1000, k=1)
    EN.generate_predictions(G_unit, 2016, 1, 2016, 12)

    pred_df = EN.mean_pred_to_df(G_unit, "Net migration", true_vals=True)

    print(pred_df)

    EN.pred_plot(G_unit, "Net migration", plot_type="Comparison")


def test_pred_error(G_unit):

    EN.train_model(G_unit, 2015, 1, 2015, 12, 1000, 1000, k=1)
    EN.generate_predictions(G_unit, 2016, 1, 2016, 12)

    EN.pred_plot(G_unit, "Net migration", plot_type="Error")
