from delphi.core import *
from pandas import Series
from pandas.testing import assert_series_equal
import pytest

s_index = ['A', '∂A/∂t']

def test_construct_default_initial_state():
    series = construct_default_initial_state(s_index)
    assert_series_equal(series, Series({'A': 100, '∂A/∂t': 1.0}))
