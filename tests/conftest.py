import pickle
import pytest
from datetime import date
from indra.statements import Concept, Influence, Evidence
from delphi.AnalysisGraph import AnalysisGraph
from delphi.utils.indra import *
from delphi.utils.shell import cd

conflict_string = "UN/events/human/conflict"
food_security_string = "UN/entities/human/food/food_security"

conflict = Concept(
    conflict_string,
    db_refs={
        "TEXT": "conflict",
        "UN": [(conflict_string, 0.8), ("UN/events/crisis", 0.4)],
    },
)

food_security = Concept(
    food_security_string,
    db_refs={
        "TEXT": "food security",
        "UN": [(food_security_string, 0.8)],
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

STS = [s1, s2, s3]


@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(STS))
    G.assemble_transition_model_from_gradable_adjectives()
    G.map_concepts_to_indicators()
    G.parameterize(date(2014, 12, 1))
    G.to_pickle()
    G.create_bmi_config_file()
    yield G
