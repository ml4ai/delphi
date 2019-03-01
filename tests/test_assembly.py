import numpy as np
import pandas as pd
from conftest import *
from datetime import datetime
from indra.statements import Concept, Influence
from delphi.assembly import *
from future.utils import lfilter
from delphi.utils.indra import *
import pytest


def test_deltas():
    assert deltas(s1) == (
        {"adjectives": ["large"], "polarity": 1},
        {"adjectives": ["small"], "polarity": -1},
    )


def test_nameTuple():
    assert nameTuple(s1) == (conflict_string, food_security_string)


def test_top_grounding():
    assert top_grounding(conflict) == conflict_string


def test_top_grounding_score():
    assert top_grounding_score(conflict) == 0.8


def test_scope():
    assert STS[0].subj.name == conflict_string


def test_get_concepts():
    assert get_concepts(STS) == set(
        [conflict_string, food_security_string, "precipitation", "flooding"]
    )


def test_is_simulable():
    assert is_simulable(s1)
    assert not is_simulable(s2)


def test_is_grounded():
    assert not is_grounded(precipitation)
    assert is_grounded(conflict)
    assert is_grounded(s1)


def test_is_well_grounded():
    assert not is_well_grounded(food_security, cutoff=0.9)
    assert is_well_grounded(food_security, cutoff=0.4)
    assert is_well_grounded(s1, cutoff=0.5)


def test_is_grounded_to_name():
    assert is_grounded_to_name(food_security, food_security_string)


def test_contains_concept():
    assert contains_concept(s1, conflict_string)


def test_contains_relevant_concept():
    relevant_concepts = [food_security_string]
    assert contains_relevant_concept(s1, relevant_concepts)
    assert not contains_relevant_concept(s3, relevant_concepts)


def test_get_indicator_data():
    indicator = Indicator(
        "Value, Political stability and absence of violence/terrorism (index)"
    )
    assert get_indicator_value(indicator, year = 2012, month = 1)[0] == -1.961666666666667
