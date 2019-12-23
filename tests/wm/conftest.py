import pickle
import pytest
from datetime import date
from indra.statements import (
    Concept,
    Influence,
    Evidence,
    Event,
    QualitativeDelta,
)
from delphi.cpp.DelphiPython import AnalysisGraph
from delphi.utils.indra import *
from delphi.utils.shell import cd

concepts = {
    "conflict": {
        "grounding": "wm/concept/causal_factor/social_and_political/conflict/war",
        "delta": {"polarity": 1, "adjective": ["small"]},
    },
    "food security": {
        "grounding": "wm/concept/causal_factor/condition/food_security",
        "delta": {"polarity": -1, "adjective": ["large"]},
    },
    "migration": {
        "grounding": "wm/concept/causal_factor/social_and_political/migration/human_migration",
        "delta": {"polarity": 1, "adjective": ["small"]},
    },
    "product": {
        "grounding": "wm/concept/causal_factor/agriculture/crop_storage",
        "delta": {"polarity": 1, "adjective": ["large"]},
    },
    "economic crisis": {
        "grounding": "wm/concept/causal_factor/crisis_and_disaster/crisis/economic_crisis/poverty",
        "delta": {"polarity": 1, "adjective": ["large"]},
    },
    "precipitation": {
        "grounding": "wm/concept/indicator_and_reported_property/weather/rainfall",
        "delta": {"polarity": 1, "adjective": []},
    },
    "inflation": {
        "grounding": "wm/concept/causal_factor/economic_and_commerce/economic activity/market/inflation",
        "delta": {"polarity": -1, "adjective": []},
    },
}


def make_event(concept, attrs):
    return Event(
        Concept(
            attrs["grounding"],
            db_refs={"TEXT": concept, "UN": [(attrs["grounding"], 0.8)]},
        ),
        delta=QualitativeDelta(
            attrs["delta"]["polarity"], attrs["delta"]["adjective"]
        ),
    )


def make_statement(event1, event2):
    return Influence(
        event1,
        event2,
        evidence=Evidence(
            annotations={
                "subj_adjectives": event1.delta.adjectives,
                "obj_adjectives": event2.delta.adjectives,
            }
        ),
    )


events = {
    concept: make_event(concept, attrs) for concept, attrs in concepts.items()
}

precipitation = Event(Concept("precipitation"))

s1 = make_statement(events["conflict"], events["food security"])
s2 = make_statement(events["migration"], events["product"])
s3 = make_statement(events["migration"], events["economic crisis"])
s4 = make_statement(events["precipitation"], events["inflation"])

STS = [s1]


@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(STS))
    G.res = 200
    G.sample_from_prior()
    G.map_concepts_to_indicators()
    G.parameterize(year=2017, month=9)
    yield G


@pytest.fixture(scope="session")
def G_eval():
    G = AnalysisGraph.from_statements([s2])
    G.map_concepts_to_indicators()
    G.res = 200
    G.sample_from_prior()
    G.parameterize(year=2017, month=9)
    G.get_timeseries_values_for_indicators()
    yield G


@pytest.fixture(scope="session")
def G_unit():
    G = AnalysisGraph.from_statements([s3])
    G.map_concepts_to_indicators()
    G.res = 200
    G.sample_from_prior()
    yield G
