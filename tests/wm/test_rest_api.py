import json
import pytest
import time

from delphi.apps.rest_api import create_app, db
from delphi.apps.rest_api.models import DelphiModel, CauseMosAsyncExperimentResult, ExperimentResult


@pytest.fixture(scope="module")
def app():
    app = create_app(debug=True)
    app.testing = True

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True

    with app.app_context():
        #db.create_all()
        yield app
        #db.drop_all()


@pytest.fixture(scope="module")
def client(app):
    """A test client for the app."""
    return app.test_client()


def create_model(client, requrst_json_fle, model_no):
    print(f'\n\nCreating model {model_no}\n--------------')

    with open(
            requrst_json_fle, encoding="utf-8"
    ) as f:
        data = json.load(f)

    model_id = data['id']
    print(f'\nmodel {model_no} id: ', model_id)

    rv = client.post(f"/delphi/create-model", json=data)
    print(f'\ncreate-model {model_no} response:\n', rv.get_json())

    rv = client.get(f"/delphi/models/{model_id}")
    status = rv.get_json()['status']
    print(f'\nmodel {model_no} status initial response:\n', rv.get_json())

    if status == 'invalid model id':
        print(f'\nGet Model {model_no} Status Error: Invalid model id!!')
        assert False
        return
    elif status == 'server error: training':
        print(f'\nModel {model_no} - Server error: training process cannot be forked!!')
        assert False
        return

    count = 1
    while status == 'training':
        rv = client.get(f"/delphi/models/{model_id}")
        status = rv.get_json()['status']
        print('\n\t', '--'*count, status)
        count += 1
        time.sleep(5)

    print(f'\nmodel {model_no} status final response:\n', rv.get_json())

    return model_id


def create_experiment(client, request_json_fle, model_id, experiment_no):
    print(f'\n\nCreating experiment {experiment_no}\n----------------------')

    with open(
            request_json_fle, encoding="utf-8"
    ) as f:
        data = json.load(f)

    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    experiment_id = rv.get_json()["experimentId"]
    print(f'\nexperiment {experiment_no} id: ', experiment_id)

    if experiment_id == 'invalid model id':
        print(f'\nCreate Experiment {experiment_no} Error: Invalid model id!!')
        assert False
        return
    elif experiment_id == 'model not trained':
        print(f'\nCreate Experiment {experiment_no} Error: Model not trained. Cannot run experiment!!')
        assert False
        return

    return rv


def get_experiment_results(client, model_id, experiment_id):
    rv = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id}")
    status = rv.get_json()["status"]
    if status == 'invalid experiment id':
        print(f'\n\nGet Experiment {experiment_id} Results Error: Invalid experiment id!!')
        assert False
        return

    status = "in progress"
    count = 1
    while status == "in progress":
        rv = client.get(f"/delphi/models/{model_id}/experiments/{experiment_id}")
        status = rv.get_json()["status"]
        print('\n\t', '--'*count, status)
        count += 1
        time.sleep(1)

    status = rv.get_json()["status"]
    print(f'\nexperiment {experiment_id} final status: ', status)

    return rv


def test_createModel_and_createExperiment(client):
    # Test create Model
    model_id1 = create_model(client, "tests/data/delphi/create_model_input_2.json", 1)

    # Test create Experiment
    rv = create_experiment(client, "tests/data/delphi/experiments_projection_input_2.json",
                           model_id1, 1)
    experiment_id1 = rv.get_json()["experimentId"]

    # Test get Experiment Results
    rv11 = get_experiment_results(client, model_id1, experiment_id1)

    time.sleep(1)

    # Test create Experiment for a second time
    rv = create_experiment(client, "tests/data/delphi/experiments_projection_input_2.json", model_id1, 2)
    experiment_id2 = rv.get_json()["experimentId"]

    get_experiment_results(client, model_id1, experiment_id2)

    # Request results for a previous experiment. The results we got earlier and
    # the results we are getting now must be identical.
    print('\n\nRetrieving experiment 1 results again\n-------------------------------------')
    rv12 = get_experiment_results(client, model_id1, experiment_id1)
    assert rv11.get_json() == rv12.get_json()

    status = rv12.get_json()["status"]
    print('\nexperiment 1 final status 2nd retrieval: ', status)

    # Test create another Model
    model_id2 = create_model(client, "tests/data/delphi/create_model_input_1.json", 2)

    # Delete the rows added to the database by testing code
    CauseMosAsyncExperimentResult.query.filter_by(id=experiment_id1).delete()
    CauseMosAsyncExperimentResult.query.filter_by(id=experiment_id2).delete()
    ExperimentResult.query.filter_by(id=experiment_id1).delete()
    ExperimentResult.query.filter_by(id=experiment_id2).delete()
    DelphiModel.query.filter_by(id=model_id1).delete()
    DelphiModel.query.filter_by(id=model_id2).delete()
    db.session.commit()

    assert True