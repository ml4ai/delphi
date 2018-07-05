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

# Concepts
c1 = Concept("X")
c2 = Concept(
    "conflict",
    db_refs={
        "TEXT": "conflict",
        "UN": [
            ("UN/events/human/conflict", 0.8),
            ("UN/events/crisis", 0.4),
        ],
    },
)

c3 = Concept(
    "food security",
    db_refs={
        "TEXT": "food security",
        "UN": [
            ("UN/entities/food/food_security", 0.8),
        ],
    },
)



# Statements
statement1 = Influence(c2, c3)
statement2 = Influence(c1, c2)


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

# Concept to indicator mapping
concept_to_indicator_mapping="""\
concepts:
    food security:
        indicators:
            average dietary energy supply adequacy:
                source: FAO
                url: http://www.fao.org/economic/ess/ess-fs/ess-fadata/en/#.Wx7h1y2ZP3Y
            average value of food production:
                source: FAO
                url: http://www.fao.org/economic/ess/ess-fs/ess-fadata/en/#.Wx7h1y2ZP3Y
"""
yaml = YAML()
mapping = yaml.load(concept_to_indicator_mapping)
