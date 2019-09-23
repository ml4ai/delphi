import json
from datetime import date
from uuid import uuid4

import numpy as np
import pytest

from conftest import G, concepts
from delphi.apps.rest_api import create_app, db
from delphi.apps.rest_api.models import (
    CausalRelationship,
    CausalVariable,
    DelphiModel,
    Evidence,
    ForwardProjection,
    ICMMetadata,
)
from delphi.random_variables import LatentVar


@pytest.fixture(scope="module")
def app(G):
    app = create_app(debug=True)
    app.testing = True

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

    with app.app_context():
        G.to_sql(app)
        yield app
        db.drop_all()


@pytest.fixture(scope="module")
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.mark.skip
def test_listAllICMs(G, client):
    rv = client.get("/icm")
    assert G.id in rv.json


def test_getICMByUUID(G, client):
    rv = client.get(f"/icm/{G.id}")
    assert G.id == rv.json["id"]


def test_getICMPrimitives(G, client):
    rv = client.get(f"/icm/{G.id}/primitive")
    assert len(rv.json) == 3


def test_createExperiment(G, client):
    timestamp = "2018-11-01"
    post_data = {
        "interventions": [
            {
                "id": G.nodes[concepts["conflict"]["grounding"]]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": {"baseType": "FloatValue", "value": 0.77},
                },
            },
            {
                "id": G.nodes[concepts["food security"]["grounding"]]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": {"baseType": "FloatValue", "value": 0.01},
                },
            },
        ],
        "projection": {
            "numSteps": 4,
            "stepSize": "MONTH",
            "startTime": "2018-10-25T15:10:37.419Z",
        },
        "options": {"timeout": 3600},
    }
    rv = client.post(f"icm/{G.id}/experiment", json=post_data)
    assert b"Forward projection sent successfully" in rv.data


def test_getExperiment(G, client):
    experiment = ForwardProjection.query.first()
    url = "/".join(["icm", G.id, "experiment", experiment.id])
    rv = client.get(url)
    assert rv.json["id"] == experiment.id


@pytest.mark.skip
def test_getAllModels(G, client):
    rv = client.get("/delphi/models")
    assert G.id in rv.json


def test_createModel(client):
    with open("tests/data/delphi_create_model_payload.json", encoding="utf-8") as f:
        data = json.load(f)
    rv = client.post(f"/delphi/create-model", json=data)
    post_data = {
        "startTime": {"year": 2017, "month": 4},
        "perturbations": [
            {"concept": "UN/entities/human/food/food_price", "value": 0.2}
        ],
        "timeStepsInMonths": 3,
    }

    rv = client.post(f"/delphi/models/{data['id']}/projection", json=post_data)
    experimentId = rv.json["experimentId"]
    url = f"delphi/models/{data['id']}/experiment/{experimentId}"
    rv = client.get(url)
    print(json.dumps(rv.json["results"]["UN/events/human/famine"], indent=2))


def test_getIndicators(client):
    with open("tests/data/causemos_cag.json", "r") as f:
        data = json.load(f)

    rv = client.post(f"/delphi/models", json=data)

    indicator_get_request_params = {
        "start": 2012,
        "end": 2016,
        "geolocation": None,
        "func": "mean",
    }
    rv = client.get(
        f"/delphi/models/{data['model_id']}/indicators?start=2012&end=2016&func=mean"
    )
    assert True
