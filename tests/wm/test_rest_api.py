import json
import pytest
import time
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
from time import sleep
from matplotlib import pyplot as plt


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


def test_createModelNew(client):
    with open("tests/data/delphi/causemos_create-model.json", encoding="utf-8") as f:
        data = json.load(f)
    rv = client.post(f"/delphi/create-model", json=data)
    assert True

def test_createExperiment(client):
    with open("tests/data/delphi/causemos_experiments_projection_input.json", encoding="utf-8") as f: data = json.load(f)
    model_id="XYZ"
    rv = client.post(f"/delphi/models/{model_id}/experiments", json=data)
    print(rv.get_json())

    # An ides to test the responwe from flask
    '''
    # Retrieve experiment results
    # Extract the experiment uuid
    experiment_id = "this_is_a_dummy"
    status = "not completed" # Replace this with returned json value
    while status != "completed":
        rv = client.post(f"/delphi/models/{model_id}/experiments/{experiment_id}")
        print(rv.get_json())
        time.sleep(1)
        # Extract returned json and read status
        status = "completed" # Replace this with returned json value
    '''
    assert True


@pytest.mark.skip
def test_createModel(client):
    with open("tests/data/delphi_create_model_payload.json", encoding="utf-8") as f:
        data = json.load(f)
    rv = client.post(f"/delphi/create-model", json=data)
    post_data = {
        "startTime": {"year": 2017, "month": 4},
        "perturbations": [
            {"concept": "wm/concept/indicator_and_reported_property/weather/rainfall", "value": 0.1}
        ],
        "timeStepsInMonths": 6,
    }

    rv = client.post(f"/delphi/models/{data['id']}/projection", json=post_data)
    experimentId = rv.json["experimentId"]
    url = f"delphi/models/{data['id']}/experiment/{experimentId}"
    print("Waiting 10 seconds to query for results")
    sleep(10)
    rv = client.get(url)
    output = rv.json['results']
    # This chunk of code is for plotting outputs to compare with CauseMos views
    # and debug. Set plot_figs=True to create plots.

    plot_figs = False
    if plot_figs:
        for concept, results in output.items():
            xs = []
            ys = []
            for datapoint in results['values']:
                xs.append(datapoint['month'])
                ys.append(datapoint['value'])
            fig, ax = plt.subplots()
            ax.plot(xs, ys, label=concept)
            ax.legend()
            plt.savefig(f"{concept.replace('/','_')}.pdf")


    # Test overwriting
    rv = client.post(f"/delphi/create-model", json=data)
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
