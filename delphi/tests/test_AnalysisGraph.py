from indra.statements import Influence, Concept
from delphi.random_variables import Indicator
from delphi.AnalysisGraph import *
from delphi.quantification import *
from delphi.subgraphs import (
    get_subgraph_for_concept,
    get_subgraph_for_concept_pair,
)
import pytest
from delphi.tests.conftest import *


def test_make_edge():
    assert make_edge(sts, ("conflict", "food_security")) == (
        "conflict",
        "food_security",
        {"InfluenceStatements": [sts[0]]},
    )


def test_get_subgraph_for_concept(G):
    concept_of_interest = "food_security"
    sg = get_subgraph_for_concept(G, concept_of_interest)
    assert set(sg.nodes()) == set(["conflict", "food_security"])


def test_get_subgraph_for_concept_pair(G):
    concept_pair = ("conflict", "food_security")
    sg = get_subgraph_for_concept_pair(G, *concept_pair)
    assert set(sg.nodes()) == set(concept_pair)


def test_map_concepts_to_indicators(G):
    map_concepts_to_indicators(G)
    indicator = Indicator(
        name="Number of severely food insecure people Value",
        source="FAO/WDI",
        mean=None,
        stdev=None,
        time=None,
    )
    assert G.nodes["food_security"]["indicators"][0].name == indicator.name


def test_infer_transition_model(G):
    G.infer_transition_model()
    assert (
        len(
            G.get_edge_data("conflict", "food_security")["InfluenceStatements"]
        )
        == 1
    )
