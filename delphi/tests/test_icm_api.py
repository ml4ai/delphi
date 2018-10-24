import json
import pytest
from delphi.icm_api import create_app
from delphi.icm_api.models import *
from delphi.tests.conftest import *
from datetime import date


@pytest.fixture
def delphi_model(G):
    return DelphiModel(id=G.id, model=G)


@pytest.fixture
def causal_primitives(G):
    return CausalVariable(model_id=G.id)


@pytest.fixture
def icm_metadata(G):
    metadata = ICMMetadata(
        id=G.id,
        created=date.today().isoformat(),
        estimatedNumberOfPrimitives=len(G.nodes) + len(G.edges),
        createdByUser_id=1,
        lastAccessedByUser_id=1,
        lastUpdatedByUser_id=1,
    )
    return metadata


@pytest.fixture
def app(icm_metadata, delphi_model):
    app = create_app()
    app.testing = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
    with app.app_context():
        db.create_all()
        db.session.add(icm_metadata)
        db.session.add(delphi_model)
        db.session.commit()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_listAllICMs(G, client):
    rv = client.get("/icm")
    assert G.id in rv.json


def test_getICMByUUID(G, client):
    rv = client.get(f"/icm/{G.id}")
    assert G.id == rv.json["id"]


def test_getICMPrimitives(G, client):
    rv = client.get(f"/icm/{G.id}/primitive")


def test_forwardProjection_and_getExperiment(G, client):
    for n in G.nodes(data=True):
        print(n[0], n[1]["id"])
    post_url = "/".join(["icm", G.id, "experiment", "forwardProjection"])

    timestamp = "2018-11-01"
    post_data = {
        "interventions": [
            {
                "id": G.nodes["conflict"]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": 0.77,
                },
            },
            {
                "id": G.nodes["food_security"]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": 0.01,
                },
            },
        ],
        "projection": {"numSteps": 4, "stepSize": "MONTH"},
        "options": {"timeout": 3600},
    }
    rv = client.post(post_url, json=post_data)
    print(rv.json)
    assert b"Forward projection sent successfully" in rv.data
    experiment = Experiment.query.first()
    url = "/".join(["icm", G.id, "experiment", experiment.id])
    rv = client.get(url)
    assert True
