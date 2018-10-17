import pytest
from delphi.icm_api import create_app


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
    assert b'["49d53f7a-26d1-4bdd-ab49-98eea61a0345"]' in rv.data


def test_forwardProjection(client):
    client.post(
        "/".join(
            [
                "icm",
                "49d53f7a-26d1-4bdd-ab49-98eea61a0345",
                "experiment",
                "forwardProjection",
            ]
        ),
        data={
            "interventions": [
                {
                    "id": " < Causal Variable id/hash >",
                    "values": {
                        "active": "ACTIVE",
                        "time": "2018-11-01",
                        "value": 0.77,
                    },
                },
                {
                    "id": " <Some other causal Variable id/hash >",
                    "values": {
                        "active": "ACTIVE",
                        "time": "2018-10-01",
                        "value": 0.01,
                    },
                },
            ],
            "projection": {"numSteps": 4, "stepSize": "MONTH"},
            "options": {"timeout": 3600},
        },
    )
