import pickle
import pytest
from datetime import date
from indra.statements import Concept, Influence, Evidence, Event, QualitativeDelta
from delphi.AnalysisGraph import AnalysisGraph
from delphi.utils.indra import *
from delphi.utils.shell import cd

conflict_string = "UN/events/human/conflict"
human_migration_string = "UN/events/human/human_migration"
food_security_string = "UN/entities/human/food/food_security"

conflict = Event(
    Concept(
        conflict_string,
        db_refs={
            "TEXT": "conflict",
            "UN": [(conflict_string, 0.8), ("UN/events/crisis", 0.4)],
        },
    ),
    delta=QualitativeDelta(1, ["large"]),
)

food_security = Event(
    Concept(
        food_security_string,
        db_refs={"TEXT": "food security", "UN": [(food_security_string, 0.8)]},
    ),
    delta=QualitativeDelta(-1, ["small"]),
)

precipitation = Event(Concept("precipitation"))
human_migration = Event(
    Concept(
        human_migration_string,
        db_refs={"TEXT": "migration", "UN": [(human_migration_string, 0.8)]},
    ),
    delta=QualitativeDelta(1, ["large"]),
)


flooding = Event(Concept("flooding"))

s1 = Influence(
    conflict,
    food_security,
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
s4 = Influence(conflict, human_migration, evidence=Evidence(annotations =
    default_annotations))

STS = [s1, s2, s3, s4]


@pytest.fixture(scope="session")
def G():
    G = AnalysisGraph.from_statements(get_valid_statements_for_modeling(STS))
    G.sample_from_prior()
    G.map_concepts_to_indicators()
    G.parameterize(year=2014, month=12)
    G.to_pickle()
    yield G
