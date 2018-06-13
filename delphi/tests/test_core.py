from delphi.core import *
from pandas import Series
from pandas.testing import assert_series_equal
import pytest

from indra.statements import Influence, Concept

c1 = Concept("X")
c2 = Concept(
    "conflict",
    db_refs={
        "TEXT": "conflict",
        "UN": [
            ("UN/events/human/conflict", 0.8020367824862633),
            ("UN/events/crisis", 0.4484265805127715),
        ],
    },
)

c3 = Concept(
    "precipitation",
    db_refs={
        "TEXT": "precipitation",
        "UN": [
            ("UN/events/natural/weather/precipitation", 0.8586376448119921),
            ("UN/events/natural/natural_disaster/storm", 0.5535723072294313),
        ],
    },
)


relevant_concepts = ["precipitation"]

statement = Influence(c2, c3)
s_index = ["A", "∂A/∂t"]


def test_construct_default_initial_state():
    series = construct_default_initial_state(s_index)
    assert_series_equal(series, Series({"A": 100, "∂A/∂t": 1.0}))


def test_is_grounded_concept():
    assert not is_grounded_concept(c1)
    assert is_grounded_concept(c2)


def test_top_grounding_score():
    assert top_grounding_score(c2) == 0.8020367824862633


def test_is_well_grounded_concept():
    assert not is_well_grounded_concept(c3, cutoff=0.9)
    assert is_well_grounded_concept(c3, cutoff=0.4)


def test_is_well_grounded_statement():
    assert is_well_grounded_statement(statement, cutoff=0.5)


def test_is_grounded_to_name():
    assert is_grounded_to_name(c2, "conflict")


def test_contains_concept():
    assert contains_concept(statement, "conflict")


def test_contains_relevant_concept():
    assert contains_relevant_concept(statement, relevant_concepts)
