import os
from indra.statements import Influence, Concept
from delphi.random_variables import Indicator
from delphi.AnalysisGraph import *
from delphi.quantification import *
from delphi.subgraphs import (
    get_subgraph_for_concept,
    get_subgraph_for_concept_pair,
)
import pickle
import pytest
from delphi.tests.conftest import *

# Testing constructors


def test_make_edge():
    assert make_edge(sts, ("conflict", "food_security")) == (
        "conflict",
        "food_security",
        {"InfluenceStatements": [sts[0]]},
    )


def test_from_statements():
    G = AnalysisGraph.from_statements(sts)
    assert set(G.nodes()) == set(["conflict", "food_security"])
    assert set(G.edges()) == set([("conflict", "food_security")])


def test_from_statements_file(test_statements_file):
    with open(test_statements_file, "rb") as f:
        sts_from_file = pickle.load(f)
    G = AnalysisGraph.from_statements(sts_from_file)
    assert set(G.nodes()) == set(["conflict", "food_security"])
    assert set(G.edges()) == set([("conflict", "food_security")])
    os.remove(test_statements_file)


def test_from_pickle(G, test_model_file):
    with open(test_model_file, "wb") as f:
        pickle.dump(G, f)
    with open(test_model_file, "rb") as f:
        M = pickle.load(f)
    assert set(M.nodes()) == set(["conflict", "food_security"])
    assert set(M.edges()) == set([("conflict", "food_security")])
    os.remove(test_model_file)


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
