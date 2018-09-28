from delphi.export import *


def test_to_dict(G):
    d = to_dict(G)
    assert (
        d["name"] == "Linear Dynamical System with Stochastic Transition Model"
    )
    variables = [x["name"] for x in d["variables"]]
    assert d["timeStep"] == "1.0"
    assert set(variables) == set(["conflict", "food_security"])
