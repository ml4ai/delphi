from delphi.export import *
from conftest import *


def test_to_dict(G):
    d = G.to_dict()
    assert (
        d["name"] == "Linear Dynamical System with Stochastic Transition Model"
    )
    variables = [x["name"] for x in d["variables"]]
    assert d["timeStep"] == "1.0"
    assert set(variables) == set([conflict_string, food_security_string])
