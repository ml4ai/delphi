from indra.statements import Influence, Concept
from delphi.types import Indicator
from delphi.AnalysisGraph import *

conflict = Concept(
    "conflict",
    db_refs={
        "TEXT": "conflict",
        "UN": [("UN/events/human/conflict", 0.8), ("UN/events/crisis", 0.4)],
    },
)
food_security = Concept(
    "food_security",
    db_refs={
        "TEXT": "food security",
        "UN": [("UN/entities/food/food_security", 0.8)],
    },
)
precipitation = Concept("precipitation")
flooding = Concept("flooding")

s1 = Influence(
    conflict,
    food_security,
    {"adjectives": ["large"], "polarity": 1},
    {"adjectives": ["small"], "polarity": -1},
)
s2 = Influence(precipitation, food_security)
s3 = Influence(precipitation, flooding)

sts = [s1, s2, s3]
G = AnalysisGraph.from_statements(sts)

def test_make_edge():
    assert make_edge(sts, ("conflict", "food_security")) == ("conflict", "food_security", {"InfluenceStatements": [s1]})

def test_get_subgraph_for_concept():
    concept_of_interest = "food_security"
    sg = G.get_subgraph_for_concept(concept_of_interest)
    assert set(sg.nodes()) == set(["conflict", "food_security"])


def test_get_subgraph_for_concept_pair():
    concept_pair = ("conflict", "food_security")
    sg = G.get_subgraph_for_concept_pair(*concept_pair)
    assert set(sg.nodes()) == set(concept_pair)


def test_map_concepts_to_indicators():
    G.map_concepts_to_indicators()
    indicator = Indicator(
        name="Number of severely food insecure people Value",
        source="FAO/WDI",
        mean=None,
        stdev=None,
        time=None,
    )
    assert G.nodes["food_security"]["indicators"][0].name == indicator.name


def test_infer_transition_model():
    G.infer_transition_model()
    assert (
        len(G.get_edge_data("conflict", "food_security")["InfluenceStatements"])
        == 1
    )
