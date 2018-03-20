from indra.statements import Influence, Agent
from delphi.core import *

statements = [Influence(
        Agent('X'),
        Agent('Y'),
        {'adjectives': ["small"], 'polarity': 1},
        {'adjectives': ["large"], 'polarity': -1},
    ),
    Influence(
        Agent('Y'),
        Agent('Z'),
        {'adjectives': ["big"], 'polarity': 1},
        {'adjectives': ["huge"], 'polarity': 1},
    ),
]

def test_construct_CAG_skeleton():
    G = construct_CAG_skeleton(statements)
    assert len(G) == 3
    assert len(G.edges()) == 2

def test_export_model():
    export_model(statements)
