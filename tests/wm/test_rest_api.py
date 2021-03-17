import json
import pytest
import time

from delphi.apps.rest_api import create_app, db
from delphi.apps.rest_api.models import DelphiModel, CauseMosAsyncExperimentResult


@pytest.fixture(scope="module")
def app():
    app = create_app(debug=True)
    app.testing = True

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

    with app.app_context():
        db.create_all()
        yield app
        #db.drop_all()


@pytest.fixture(scope="module")
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_createModel_and_createExperiment(client):
    # Test createModel
    print('\n\nCreating model\n--------------')

    with open(
        "tests/data/delphi/create_model_input_2.json", encoding="utf-8"
    ) as f:
        data = json.load(f)

    model_id = data['id']
    print('\nmodel id: ', model_id)

    rv = client.post(f"/delphi/create-model", json=data)
    print('\ncreate-model response:\n', rv.get_json())

    rv = client.get(f"/delphi/models/{model_id}")
    status = rv.get_json()['status']
    print('\nmodel status initial response:\n', rv.get_json())

    if status == 'training error':
        print('\nServer error: training process cannot be forked!!')
        assert False
        return

    count = 1
    while status == 'training':
        rv = client.get(f"/delphi/models/{model_id}")
        status = rv.get_json()['status']
        print('\n\t', '--'*count, status)
        count += 1
        time.sleep(5)

    print('\nmodel status final response:\n', rv.get_json())

    # Test createExperiment
    print('\n\nCreating experiment 1\n----------------------')

    with open(
        "tests/data/delphi/experiments_projection_input_2.json",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    experiment_id1 = rv.get_json()["experimentId"]
    print('\nexperiment 1 id: ', experiment_id1)

    status = "in progress"
    count = 1
    while status == "in progress":
        rv11 = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id1}")
        status = rv11.get_json()["status"]
        print('\n\t', '--'*count, status)
        count += 1
        time.sleep(1)

    status = rv11.get_json()["status"]
    print('\nexperiment 1 final status: ', status)

    time.sleep(1)

    # Test createExperiment for a second time
    print('\n\nCreating experiment 2\n----------------------')
    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    experiment_id2 = rv.get_json()["experimentId"]
    print('\nexperiment 2 id: ', experiment_id2)

    status = "in progress"
    count = 1
    while status == "in progress":
        rv21 = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id1}")
        status = rv21.get_json()["status"]
        print('\n\t', '--'*count, status)
        count += 1
        time.sleep(1)

    status = rv21.get_json()["status"]
    print('\nexperiment 2 final status: ', status)

    # Request results for a previous experiment. The results we got earlier and
    # the results we are getting now must be identical.
    print('\n\nRetrieving experiment 1 results again\n-------------------------------------')
    rv12 = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id1}")
    assert rv11.get_json() == rv12.get_json()

    status = rv12.get_json()["status"]
    print('\nexperiment 1 final status 2nd retrieval: ', status)

    # Delete the rows added to the database by testing code
    CauseMosAsyncExperimentResult.query.filter_by(id=experiment_id1).delete()
    CauseMosAsyncExperimentResult.query.filter_by(id=experiment_id2).delete()
    DelphiModel.query.filter_by(id=model_id).delete()
    db.session.commit()

    assert True
