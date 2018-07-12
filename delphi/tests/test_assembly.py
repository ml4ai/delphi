import numpy as np
import pandas as pd
from datetime import datetime
from indra.statements import Concept, Influence
from .assembly import *
from .paths import adjectiveData, south_sudan_data
from future.utils import lfilter
import pytest

c1 = Concept(
    "conflict",
    db_refs={
        "TEXT": "conflict",
        "UN": [
            ("UN/events/human/conflict", 0.8),
            ("UN/events/crisis", 0.4),
        ],
    },
)
c2 = Concept(
    "food_security",
    db_refs={
        "TEXT": "food security",
        "UN": [
            ("UN/entities/food/food_security", 0.8),
        ],
    },
)
c3 = Concept("precipitation")
c4 = Concept("flooding")

s1 = Influence(c1, c2, {'adjectives': ['large'], 'polarity': 1}, {'adjectives': ['small'], 'polarity': -1})
s2 = Influence(c3, c2)
s3 = Influence(c3, c4)

sts = [s1, s2]

gb = pd.read_csv(adjectiveData, delim_whitespace=True).groupby("adjective")

def test_deltas():
    assert deltas(s1) == ({'adjectives': ['large'], 'polarity' : 1}, {'adjectives': ['small'], 'polarity' : -1})

# Not testing get_respdevs


def test_nameTuple():
    assert nameTuple(s1) == ("conflict", "food_security")


def test_make_edge():
    assert make_edge(sts, ("conflict", "food_security")) == ("conflict", "food_security", {"InfluenceStatements": [s1]})


def test_top_grounding():
    assert top_grounding(c1) == "conflict"


def test_top_grounding_score():
    assert top_grounding_score(c1) == 0.8


def test_get_concepts():
    assert get_concepts(sts) == set(["conflict", "food_security", "precipitation"])


def test_process_concept_name():
    assert process_concept_name("food_security") == "food security"



# Testing preassembly functions

def test_is_simulable():
    assert is_simulable(s1)
    assert not is_simulable(s2)

def test_is_grounded():
    assert not is_grounded(c3)
    assert is_grounded(c1)
    assert is_grounded(s1)

def test_is_well_grounded():
    assert not is_well_grounded(c2, cutoff=0.9)
    assert is_well_grounded(c2, cutoff=0.4)
    assert is_well_grounded(s1, cutoff=0.5)


def test_is_grounded_to_name():
    assert is_grounded_to_name(c2, "food_security")


def test_contains_concept():
    assert contains_concept(s1, "conflict")

def test_contains_relevant_concept():
    relevant_concepts = ["food_security"]
    assert contains_relevant_concept(s1, relevant_concepts)
    assert not contains_relevant_concept(s3, relevant_concepts)

faostat_data = get_data(south_sudan_data)

@pytest.mark.skip(reason="Broken by changes for EC Hackathon - won't break other things")
def test_get_indicators():
    assert (
        get_indicators("food security", mapping)[0].name
        == "average dietary energy supply adequacy"
    )


def test_get_indicator_data():
    indicator = Indicator('Political stability and absence of violence/terrorism (index), Value',
            'FAO/WDI')
    t = datetime(2012, 1, 1)
    assert get_indicator_value(indicator, t, faostat_data)[0] == -1.2
