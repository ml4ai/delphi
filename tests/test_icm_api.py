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
def delphi_model(G):
    return DelphiModel(id=G.id, model=G)


@pytest.fixture(scope="module")
def icm_metadata(G):
    metadata = ICMMetadata(
        id=G.id,
        created=date.today().isoformat(),
        estimatedNumberOfPrimitives=len(G.nodes) + len(G.edges),
        createdByUser_id=1,
        lastAccessedByUser_id=1,
        lastUpdatedByUser_id=1,
    )
    return metadata


@pytest.fixture(scope="module")
def causal_primitives(G):
    today = date.today().isoformat()
    default_latent_var_value = 1.0
    causal_primitives = []
    for n in G.nodes(data=True):
        n[1]["rv"] = LatentVar(n[0])
        n[1]["update_function"] = G.default_update_function
        rv = n[1]["rv"]
        rv.dataset = [default_latent_var_value for _ in range(G.res)]

        if n[1].get("indicators") is not None:
            for ind in n[1]["indicators"].values():
                ind.dataset = np.ones(G.res) * ind.mean

        causal_variable = CausalVariable(
            id=n[1]["id"],
            model_id=G.id,
            units="",
            namespaces={},
            auxiliaryProperties=[],
            label=n[0],
            description=f"Long description of {n[0]}.",
            lastUpdated=today,
            confidence=1.0,
            lastKnownValue={
                "active": "ACTIVE",
                "trend": None,
                "time": today,
                "value": {
                    "baseType": "FloatValue",
                    "value": n[1]["rv"].dataset[0],
                },
            },
            range={
                "baseType": "FloatRange",
                "range": {"min": 0, "max": 10, "step": 0.1},
            },
        )
        causal_primitives.append(causal_variable)

    max_evidences = max(
        [
            sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
            for e in G.edges(data=True)
        ]
    )
    max_mean_betas = max(
        [abs(np.median(e[2]["betas"])) for e in G.edges(data=True)]
    )
    for e in G.edges(data=True):
        # TODO: Have AnalysisGraph automatically assign uuids to edges.

        causal_relationship_id = e[2]['id']
        causal_relationship = CausalRelationship(
            id=e[2]['id'],
            namespaces={},
            source={"id": G.nodes[e[0]]["id"], "baseType": "CausalVariable"},
            target={"id": G.nodes[e[1]]["id"], "baseType": "CausalVariable"},
            model_id=G.id,
            auxiliaryProperties=[],
            lastUpdated=today,
            types=["causal"],
            description=f"{e[0]} influences {e[1]}.",
            confidence=np.mean(
                [s.belief for s in e[2]["InfluenceStatements"]]
            ),
            label=f"{e[0]} influences {e[1]}.",
            strength=abs(np.median(e[2]["betas"]) / max_mean_betas),
            reinforcement=(
                True
                if np.mean(
                    [
                        stmt.subj_delta["polarity"]
                        * stmt.obj_delta["polarity"]
                        for stmt in e[2]["InfluenceStatements"]
                    ]
                )
                > 0
                else False
            ),
        )
        causal_primitives.append(causal_relationship)
    return causal_primitives


@pytest.fixture(scope="module")
def evidences(G):
    evidences = []
    for edge in G.edges(data=True):
        for stmt in edge[2]["InfluenceStatements"]:
            for ev in stmt.evidence:
                evidence = Evidence(
                    id = str(uuid4()),
                    causalrelationship_id = edge[2]['id'],
                    description = ev.text
                )
                evidences.append(evidence)
    return evidences


@pytest.fixture(scope="module")
def app(icm_metadata, delphi_model, causal_primitives, evidences):
    app = create_app()
    app.testing = True

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"

    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
    with app.app_context():
        db.create_all()
        db.session.add(icm_metadata)
        db.session.add(delphi_model)
        for causal_primitive in causal_primitives:
            db.session.add(causal_primitive)
        for evidence in evidences:
            db.session.add(evidence)
        db.session.commit()
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
                "id": G.nodes["conflict"]["id"],
                "values": {
                    "active": "ACTIVE",
                    "time": timestamp,
                    "value": {"baseType": "FloatValue", "value": 0.77},
                },
            },
            {
                "id": G.nodes["food_security"]["id"],
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
