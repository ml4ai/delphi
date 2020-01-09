import json
from datetime import date
from uuid import uuid4

import numpy as np
import pytest

from delphi.cpp.DelphiPython import AnalysisGraph

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
def app():
    app = create_app(debug=True)
    app.testing = True

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope="module")
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_createModel(client):
    with open("tests/data/delphi_create_model_payload.json", encoding="utf-8") as f:
        data = json.load(f)
    rv = client.post(f"/delphi/create-model", json=data)
    post_data = {
        "startTime": {"year": 2017, "month": 4},
        "perturbations": [
            {"concept": "wm/concept/indicator_and_reported_property/weather/rainfall", "value": 0.2}
        ],
        "timeStepsInMonths": 3,
    }

    rv = client.post(f"/delphi/models/{data['id']}/projection", json=post_data)
    experimentId = rv.json["experimentId"]
    url = f"delphi/models/{data['id']}/experiment/{experimentId}"
    rv = client.get(url)
    assert True


@pytest.mark.skip
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
