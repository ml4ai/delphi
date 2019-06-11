import os
from conftest import *
from delphi.random_variables import Indicator
from delphi.AnalysisGraph import AnalysisGraph
import pickle
import pytest

# Testing constructors


def test_from_statements():
    G = AnalysisGraph.from_statements(STS, assign_default_polarities=False)
    assert set(G.nodes()) == set([conflict_string, food_security_string])
    assert set(G.edges()) == set([(conflict_string, food_security_string)])


def test_from_statements_file():
    test_statements_file = "test_statements.pkl"
    with open(test_statements_file, "wb") as f:
        pickle.dump(STS, f)
    with open(test_statements_file, "rb") as f:
        sts_from_file = pickle.load(f)
    G = AnalysisGraph.from_statements(sts_from_file, assign_default_polarities=False)
    assert set(G.nodes()) == set([conflict_string, food_security_string])
    assert set(G.edges()) == set([(conflict_string, food_security_string)])
    os.remove(test_statements_file)


def test_get_subgraph_for_concept(G):
    concept_of_interest = food_security_string
    sg = G.get_subgraph_for_concept(concept_of_interest, reverse=True)
    assert set(sg.nodes()) == set([conflict_string, food_security_string])


def test_get_subgraph_for_concept_pair(G):
    concept_pair = (conflict_string, food_security_string)
    sg = G.get_subgraph_for_concept_pair(*concept_pair)
    assert set(sg.nodes()) == set(concept_pair)


def test_map_concepts_to_indicators(G):
    G.map_concepts_to_indicators()
    indicator = Indicator(
        name="Import Value, Infant food",
        source="WB",
        mean=None,
        stdev=None,
        time=None,
    )
    assert indicator.name in G.nodes[food_security_string]["indicators"]
