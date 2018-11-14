import pickle
import pytest
from indra.statements import Concept, Influence, Evidence
from delphi.AnalysisGraph import AnalysisGraph
from delphi.assembly import get_valid_statements_for_modeling


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
    evidence=Evidence(
        annotations={"subj_adjectives": ["large"], "obj_adjectives": ["small"]}
    ),
)

default_annotations = {"subj_adjectives": [], "obj_adjectives": []}

s2 = Influence(
    precipitation,
    food_security,
    evidence=Evidence(annotations=default_annotations),
)
s3 = Influence(
    precipitation, flooding, evidence=Evidence(annotations=default_annotations)
)

sts = [s1, s2, s3]


@pytest.fixture(scope="session")
def test_statements_file():
    return "test_statements.pkl"


@pytest.fixture(scope="session")
def test_model_file():
    return "test_model.pkl"


with open(test_statements_file(), "wb") as f:
    pickle.dump(sts, f)


@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(sts))
    G.infer_transition_model()
    return G
