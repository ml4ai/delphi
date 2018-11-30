from uuid import uuid4
from delphi.icm_api.models import (
    Evidence,
    ICMMetadata,
    CausalVariable,
    CausalRelationship,
    DelphiModel,
)
from delphi.icm_api import create_app, db
from datetime import date
from delphi.random_variables import LatentVar
import numpy as np
import os


def write_model_to_database(G):

    G.assemble_transition_model_from_gradable_adjectives()
    delphi_model = DelphiModel(id=G.id, model=G)
    icm_metadata = ICMMetadata(
        id=G.id,
        created=date.today().isoformat(),
        estimatedNumberOfPrimitives=len(G.nodes) + len(G.edges),
        createdByUser_id=1,
        lastAccessedByUser_id=1,
        lastUpdatedByUser_id=1,
    )
    today = date.today().isoformat()
    default_latent_var_value = 1.0
    causal_primitives = []
    for n in G.nodes(data=True):
        n[1]["rv"] = LatentVar(n[0])
        n[1]["update_function"] = G.default_update_function
        rv = n[1]["rv"]
        rv.dataset = [default_latent_var_value for _ in range(G.res)]

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
                "range": {"min": -2, "max": 2, "step": 0.1},
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
    evidences = []
    for e in G.edges(data=True):
        causal_relationship = CausalRelationship(
            id=e[2]["id"],
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
            label="influences",
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
        for stmt in e[2]["InfluenceStatements"]:
            for ev in stmt.evidence:
                evidence = Evidence(
                    id=str(uuid4()),
                    causalrelationship_id=e[2]["id"],
                    description=ev.text,
                )
                evidences.append(evidence)

    app = create_app()
    app.testing = True

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///delphi.db"
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
