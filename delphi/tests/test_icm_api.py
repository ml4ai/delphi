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
    CausalVariable(model_id=G.id)


@pytest.fixture
def icm_metadata(G):
    metadata = ICMMetadata(
        id=G.id,
        icmProvider="",
        title="",
        version="",
        created=date.today().isoformat(),
        createdByUser="",
        lastAccessed="",
        lastAccessedByUser="",
        lastUpdated="",
        lastUpdatedByUser="",
        estimatedNumberOfPrimitives=len(G.nodes) + len(G.edges),
        lifecycleState="",
        derivation="",
    )
    return metadata


@pytest.fixture
def app(icm_metadata, delphi_model):
    app = create_app()
    app.testing = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
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


def test_forwardProjection(G, client):
    post_url = "/".join(["icm", G.id, "experiment", "forwardProjection"])

    timestamp = "2018-11-01"
    post_data = {
        "interventions": [
            {
                "id": "precipitation",
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": 0.77,
                },
            },
            {
                "id": "<Some other causal Variable id/hash>",
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
    print(rv.data)
