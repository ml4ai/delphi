import json
import pytest
from conftest import *
from uuid import uuid4
from delphi.icm_api import create_app, db
from delphi.icm_api.models import (
    Evidence,
    ICMMetadata,
    CausalVariable,
    CausalRelationship,
    DelphiModel,
    ForwardProjection,
)
from datetime import date
from delphi.random_variables import LatentVar
import numpy as np


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
    post_url = "/".join(["icm", G.id, "experiment"])

    timestamp = "2018-11-01"
    post_data = {
        "interventions": [
            {
                "id": G.nodes[conflict_string]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": {"baseType": "FloatValue", "value": 0.77},
                },
            },
            {
                "id": G.nodes[food_security_string]["id"],
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
    rv = client.post(post_url, json=post_data)
    assert b"Forward projection sent successfully" in rv.data


def test_getExperiment(G, client):
    experiment = ForwardProjection.query.first()
    url = "/".join(["icm", G.id, "experiment", experiment.id])
    rv = client.get(url)
    assert rv.json["id"] == experiment.id
