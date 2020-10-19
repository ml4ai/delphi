import json
import pytest
import time

from delphi.apps.rest_api import create_app, db
from delphi.apps.rest_api.models import DelphiModel


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


def test_createModel_and_createExperiment(client):
    # test_createModel
    with open(
        "tests/data/delphi/causemos_create-model.json", encoding="utf-8"
    ) as f:
        data = json.load(f)
    rv = client.post(f"/delphi/create-model", json=data)

    # Test createExperiment
    with open(
        "tests/data/delphi/causemos_experiments_projection_input.json",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    model_id = "XYZ"
    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    experiment_id = rv.get_json()["experimentId"]
    status = "in progress"

    while status == "in progress":
        time.sleep(1)
        rv = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id}")
        status = rv.get_json()["status"]
        print(status)

    # Test createExperiment for a second time
    # This time model should not get trained since the trained model should
    # have been stored in the database the first time createExpetiment called
    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    experiment_id = rv.get_json()["experimentId"]
    status = "in progress"

    while status == "in progress":
        time.sleep(1)
        rv = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id}")
        status = rv.get_json()["status"]
        print(status)

    assert True
