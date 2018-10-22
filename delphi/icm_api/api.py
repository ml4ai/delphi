from uuid import uuid4
import pickle
from datetime import datetime, date
from typing import Optional, List
from delphi.bmi import initialize
from delphi.utils import flatten
from flask import jsonify, request, Blueprint
from delphi.icm_api.models import *
from delphi.paths import data_dir
import numpy as np

bp = Blueprint("icm_api", __name__)


def dress_model_for_icm_api(model):
    initialize(model, data_dir / "variables.csv")
    today = date.today().isoformat()
    for n in model.nodes(data=True):
        n[1]["id"] = uuid4()
        n[1]["units"] = ""
        n[1]["namespaces"] = []
        n[1]["label"] = n[0]
        n[1]["description"] = f"Long description of {n[0]}."
        n[1]["lastUpdated"] = today
        n[1]["lastKnownValue"] = {
            "timestep": 0,
            "value": {
                "baseType": "FloatValue",
                "value": n[1]["rv"].dataset[0],
            },
        }
        n[1]["range"] = {
            "baseType": "FloatRange",
            "range": {"min": 0, "max": 10, "step": 0.1},
        }
    max_evidences = max(
        [
            sum([len(s.evidence) for s in e[2]["InfluenceStatements"]])
            for e in model.edges(data=True)
        ]
    )
    max_mean_betas = max(
        [abs(np.median(e[2]["betas"])) for e in model.edges(data=True)]
    )
    for e in model.edges(data=True):
        e[2]["id"] = uuid4()
        e[2]["namespaces"] = []
        e[2]["source"] = model.nodes[e[0]]["id"]
        e[2]["target"] = model.nodes[e[1]]["id"]
        e[2]["lastUpdated"] = today
        e[2]["types"] = ["causal"]
        e[2]["description"] = f"{e[0]} influences {e[1]}."
        e[2]["confidence"] = np.mean(
            [s.belief for s in e[2]["InfluenceStatements"]]
        )
        e[2]["label"] = f"{e[0]} influences {e[1]}."
        e[2]["strength"] = abs(np.median(e[2]["betas"]) / max_mean_betas)
        e[2]["reinforcement"] = np.mean(
            [
                stmt.subj_delta["polarity"] * stmt.obj_delta["polarity"]
                for stmt in e[2]["InfluenceStatements"]
            ]
        )

    return model


@bp.route("/icm", methods=["POST"])
def createNewICM():
    """ Create a new ICM"""
    return ('', 415)


@bp.route("/icm", methods=["GET"])
def listAllICMs():
    """ List all ICMs"""
    return jsonify([metadata.id for metadata in ICMMetadata.query.all()])


@bp.route("/icm/<string:uuid>", methods=["GET"])
def getICMByUUID(uuid: str):
    """ Fetch an ICM by UUID"""
    return jsonify(ICMMetadata.query.filter_by(id=uuid).first().serialize())


@bp.route("/icm/<string:uuid>", methods=["DELETE"])
def deleteICM(uuid: str):
    """ Deletes an ICM"""
    model = ICMMetadata.query.filter_by(id=uuid).first()
    db.session.delete(model)
    db.session.commit()
    return ("", 204)


@bp.route("/icm/<string:uuid>", methods=["PATCH"])
def updateICMMetadata(uuid: str):
    """ Update the metadata for an existing ICM"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/primitive", methods=["GET"])
def getICMPrimitives(uuid: str):
    """ returns all ICM primitives (TODO - needs filter support)"""
    G = DelphiModel.query.filter_by(id=uuid).first()
    print(G.model.nodes())
    return "ok"


@bp.route("/icm/<string:uuid>/primitive", methods=["POST"])
def createICMPrimitive(uuid: str):
    """ create a new causal primitive"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["GET"])
def getICMPrimitive(uuid: str, prim_id: str):
    """ returns a specific causal primitive"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["PATCH"])
def updateICMPrimitive(uuid: str, prim_id: str):
    """ update an existing ICM primitive (can use this for disable?)"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["DELETE"])
def deleteICMPrimitive(uuid: str, prim_id: str):
    """ delete (disable) this ICM primitive"""
    return ('', 415)


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["GET"]
)
def getEvidenceForID(uuid: str, prim_id: str):
    """ returns evidence for a causal primitive (needs pagination support)"""
    return ('', 415)


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["POST"]
)
def attachEvidence(uuid: str, prim_id: str):
    """ attach evidence to a primitive"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["GET"])
def getEvidenceByID(uuid: str, evid_id: str):
    """ returns an individual piece of evidence"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["PATCH"])
def updateEvidence(uuid: str, evid_id: str):
    """ update evidence item"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["DELETE"])
def deleteEvidence(uuid: str, evid_id: str):
    """ delete evidence item"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/recalculate", methods=["POST"])
def recalculateICM(uuid: str):
    """ indication that it is safe to recalculate/recompose model after performing some number of CRUD operations"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/archive", methods=["POST"])
def archiveICM(uuid: str):
    """ archive an ICM"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/branch", methods=["POST"])
def branchICM(uuid: str):
    """ branch an ICM"""
    return ('', 415)


@bp.route("/icm/fuse", methods=["POST"])
def fuseICMs():
    """ fuse two ICMs"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/sparql", methods=["POST"])
def query(uuid: str):
    """ Query the ICM using SPARQL"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/experiment/forwardProjection", methods=["POST"])
def forwardProjection(uuid: str):
    """ Execute a "what if" projection over the model"""
    return ("", 204)


@bp.route("/icm/<string:uuid>/experiment", methods=["GET"])
def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["GET"])
def getExperiment(uuid: str, exp_id: str):
    """ Fetch experiment results"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["DELETE"])
def deleteExperiment(uuid: str, exp_id: str):
    """ Delete experiment"""
    return ('', 415)


@bp.route("/icm/<string:uuid>/traverse/<string:prim_id>", methods=["POST"])
def traverse(uuid: str, prim_id: str):
    """ traverse through the ICM using a breadth-first search"""
    return ('', 415)


@bp.route("/version", methods=["GET"])
def getVersion():
    """ Get the version of the ICM API supported"""
    return ('', 415)


@bp.route("/ping", methods=["GET"])
def ping():
    """ Get the health status of the ICM server"""
    return ('', 415)
