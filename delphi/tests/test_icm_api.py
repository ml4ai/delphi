import pytest
from delphi.icm_api import create_app


@pytest.fixture
def test_model_uuid():
    yield "49d53f7a-26d1-4bdd-ab49-98eea61a0345"

@pytest.fixture
def app():
    app = create_app()
    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_icm(client):
    rv = client.get("/icm")
    assert b'["ea9bf1b4-4f88-4598-a927-09d1ff7b51e5"]' in rv.data


def test_forwardProjection(client, test_model_uuid):
    post_url = "/".join(
        [
            "icm",
            test_model_uuid,
            "experiment",
            "forwardProjection",
        ]
    )
    post_data = {
        "interventions": [
            {
                "id": "49d53f7a-26d1-4bdd-ab49-98eea61a0345",
                "values": {
                    "active": "ACTIVE",
                    "time": "2018-11-01",
                    "value": 0.77,
                },
            },
            {
                "id": "<Some other causal Variable id/hash>",
                "values": {
                    "active": "ACTIVE",
                    "time": "2018-10-01",
                    "value": 0.01,
                },
            },
        ],
        "projection": {"numSteps": 4, "stepSize": "MONTH"},
        "options": {"timeout": 3600},
    }
    rv = client.post(post_url, json=post_data)
    print(rv.data)
