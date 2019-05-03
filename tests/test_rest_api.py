import json
import pytest
from conftest import *
from uuid import uuid4
from delphi.apps.rest_api import create_app, db
from delphi.apps.rest_api.models import (
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


def test_getAllModels(G, client):
    rv = client.get("/delphi/models")
    assert G.id in rv.json


def test_getIndicators(G, client):
    with open("tests/data/causemos_cag.json", "r") as f:
        data = json.load(f)
    data.update(
        {
            "start": 2012,
            "end": 2016,
            "geolocation": None,
            "concept": "UN/events/human/conflict",
            "func": "mean",
        }
    )
    rv = client.post(f"/delphi/models/{G.id}/indicators", json=data)
    assert rv.json == {
        "UN/events/human/conflict": [
            {
                "name": "Conflict incidences",
                "score": 0.725169,
                "unit": "number of cases",
                "value": 12.63076923076923,
            },
            {
                "name": "Internally displaced persons, total displaced by conflict and violence",
                "score": 0.6580004,
                "unit": "number of people",
                "value": 1280166.6666666667,
            },
            {
                "name": "Conflict fatalities",
                "score": 0.64682186,
                "unit": "number of cases",
                "value": 59.042307692307695,
            },
            {
                "name": "Internally displaced persons, new displacement associated with conflict and violence",
                "score": 0.60062087,
                "unit": "number of cases",
                "value": 535666.6666666666,
            },
            {
                "name": "Legislation exists on domestic violence",
                "score": 0.56590945,
                "unit": "1=yes; 0=no",
                "value": 0.0,
            },
            {
                "name": "Percentage of livestock migrating due to conflict / insecurity",
                "score": 0.5439555,
                "unit": "%",
                "value": 29.4,
            },
            {
                "name": "Value, Political stability and absence of violence/terrorism (index)",
                "score": 0.5019650999999999,
                "unit": "index",
                "value": -1.961666666666667,
            },
            {
                "name": "Average number of cattle died/slaughtered/lost per  household during last 4 weeks",
                "score": 0.47536009999999995,
                "unit": None,
                "value": 73.71428571428571,
            },
            {
                "name": "Refugee population by country or territory of asylum",
                "score": 0.4751459,
                "unit": None,
                "value": 248216.0,
            },
            {
                "name": "Refugee population by country or territory of origin",
                "score": 0.47174284,
                "unit": None,
                "value": 912165.8333333334,
            },
        ]
    }
