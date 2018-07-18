from indra.statements import Concept, Influence
import pytest
from delphi.AnalysisGraph import AnalysisGraph

conflict = Concept(
        "conflict",
        db_refs={
            "TEXT": "conflict",
            "UN": [("UN/events/human/conflict", 0.8), ("UN/events/crisis", 0.4)],
        },
    )

food_security =Concept(
        "food_security",
        db_refs={
            "TEXT": "food security",
            "UN": [("UN/entities/food/food_security", 0.8)],
        },
    )

precipitation=Concept("precipitation")


flooding = Concept("flooding")
s1=Influence(
        conflict,
        food_security,
        {"adjectives": ["large"], "polarity": 1},
        {"adjectives": ["small"], "polarity": -1},
    )

s2=Influence(precipitation, food_security)
s3=Influence(precipitation, flooding)

sts = [s1, s2, s3]

@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(sts)
    G.infer_transition_model()
    return G
