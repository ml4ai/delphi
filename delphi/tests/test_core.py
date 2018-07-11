import os
import sys
from delphi.core import *
from delphi.paths import data_dir, adjectiveData, south_sudan_data
from datetime import datetime
from pandas import Series
from pandas.testing import assert_series_equal
import pytest
from pathlib import Path

from indra.statements import Influence, Concept



# Causal analysis graph
# CAG = set_indicators(create_dressed_CAG([statement1, statement2],
                    # adjectiveData))

# indicators = CAG.node['food security']['indicators']
s_index = ["A", "∂A/∂t"]


def test_construct_default_initial_state():
    series = construct_default_initial_state(s_index)
    assert_series_equal(series, Series({"A": 1.0, "∂A/∂t": 0.0}))

# def test_get_latent_state_components():
    # assert set(get_latent_state_components(CAG)) == set(['X', '∂(X)/∂t', 'conflict',
            # '∂(conflict)/∂t', 'food security', '∂(food security)/∂t'])
