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


relevant_concepts = ["food_security"]

statement1 = Influence(c2, c3)
statement2 = Influence(c1, c2)
s_index = ["A", "∂A/∂t"]


def test_construct_default_initial_state():
    series = construct_default_initial_state(s_index)
    assert_series_equal(series, Series({"A": 100, "∂A/∂t": 1.0}))


def test_is_grounded():
    assert not is_grounded(c1)
    assert is_grounded(c2)
    assert is_grounded(statement1)


def test_top_grounding_score():
    assert top_grounding_score(c2) == 0.8


def test_is_well_grounded():
    assert not is_well_grounded(c3, cutoff=0.9)
    assert is_well_grounded(c3, cutoff=0.4)
    assert is_well_grounded(statement1, cutoff=0.5)


def test_is_grounded_to_name():
    assert is_grounded_to_name(c2, "conflict")


def test_contains_concept():
    assert contains_concept(statement1, "conflict")


def test_contains_relevant_concept():
    assert contains_relevant_concept(statement1, relevant_concepts)
    assert not contains_relevant_concept(statement2, relevant_concepts)


def test_get_indicators():
    yaml_string="""\
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
    mapping = yaml.load(yaml_string)

    assert (
        list(get_indicators("food security", mapping).keys())[0]
        == "average dietary energy supply adequacy"
    )
    CAG = set_indicators(create_dressed_CAG([statement1, statement2],
                        'data/adjectiveData.tsv'))
    indicators = CAG.nodes['food security']['indicators']
    assert('average dietary energy supply adequacy' in indicators)
