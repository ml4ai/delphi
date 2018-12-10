import numpy as np
import pandas as pd
from conftest import *
from datetime import datetime
from indra.statements import Concept, Influence
from delphi.assembly import *
from delphi.paths import adjectiveData, south_sudan_data
from future.utils import lfilter
from delphi.utils.indra import *
import pytest

gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby("adjective")


def test_make_edge():
    assert make_edge(STS, ("conflict", "food_security")) == (
        "conflict",
        "food_security",
        {"InfluenceStatements": [STS[0]]},
    )


def test_deltas():
    assert deltas(s1) == (
        {"adjectives": ["large"], "polarity": 1},
        {"adjectives": ["small"], "polarity": -1},
    )


# Not testing get_respdevs


def test_nameTuple():
    assert nameTuple(s1) == ("conflict", "food_security")


def test_top_grounding():
    assert top_grounding(conflict) == "conflict"


def test_top_grounding_score():
    assert top_grounding_score(conflict) == 0.8


def test_scope():
    assert STS[0].subj.name == "conflict"


def test_get_concepts():
    assert get_concepts(STS) == set(
        ["conflict", "food_security", "precipitation", "flooding"]
    )


def test_process_concept_name():
    assert process_concept_name("food_security") == "food security"


# Testing preassembly functions


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
    assert is_grounded_to_name(food_security, "food_security")


def test_contains_concept():
    assert contains_concept(s1, "conflict")


def test_contains_relevant_concept():
    relevant_concepts = ["food_security"]
    assert contains_relevant_concept(s1, relevant_concepts)
    assert not contains_relevant_concept(s3, relevant_concepts)


indicator_data = get_data(south_sudan_data)


@pytest.mark.skip
def test_get_indicator_data():
    indicator = Indicator(
        "Political stability and absence of violence/terrorism (index), Value",
        "FAO/WDI",
    )
    t = datetime(2012, 1, 1)
    assert get_indicator_value(indicator, t, indicator_data)[0] == -1.2
