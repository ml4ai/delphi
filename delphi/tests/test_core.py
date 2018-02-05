from indra.statements import Influence, Agent
from delphi.core import construct_CAG

statements = [Influence(
        Agent('X'),
        Agent('Y'),
        {'adjectives': [], 'polarity': None},
        {'adjectives': [], 'polarity': None},
    ),
    Influence(
        Agent('Y'),
        Agent('Z'),
        {'adjectives': [], 'polarity': None},
        {'adjectives': [], 'polarity': None},
    ),
]

def test_construct_CAG_skeleton():
    G = construct_CAG_skeleton(['X', 'Y', 'Z'], statements)
    assert len(G) == 3
    assert len(G.edges()) == 2
