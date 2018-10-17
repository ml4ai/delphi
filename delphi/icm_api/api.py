from uuid import uuid4
import pickle
from datetime import datetime, date
from typing import Optional, List
from delphi.bmi import initialize
from delphi.utils import flatten
from flask import jsonify, request, Blueprint
from delphi.icm_api.models import *
from delphi.paths import data_dir
from pprint import pprint
import numpy as np

bp = Blueprint('icm_api', __name__)

def dress_model_for_icm_api(model):
    initialize(model, data_dir/"test_data"/"variables.csv")
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


with open(data_dir/"test_data"/"delphi_cag.pkl", "rb") as f:
    model = dress_model_for_icm_api(pickle.load(f))

models = {str(model.id): model}


@bp.route("/icm", methods=["POST"])
def createNewICM():
    """ Create a new ICM"""
    pass


@bp.route("/icm", methods=["GET"])
def listAllICMs():
    """ List all ICMs"""
    return jsonify(list(models.keys()))


@bp.route("/icm/<string:uuid>", methods=["GET"])
def getICMByUUID(uuid: str):
    """ Fetch an ICM by UUID"""
    if uuid in models:
        model = models[uuid]
        return jsonify(
            asdict(
                ICMMetadata(
                    id=uuid,
                    estimatedNumberOfPrimitives=len(model)
                    + len(model.edges()),
                )
            )
        )


@bp.route("/icm/<string:uuid>", methods=["DELETE"])
def deleteICM(uuid: str):
    """ Deletes an ICM"""
    if uuid in models:
        del models[uuid]


@bp.route("/icm/<string:uuid>", methods=["PATCH"])
def updateICMMetadata(uuid: str):
    """ Update the metadata for an existing ICM"""
    pass


@bp.route("/icm/<string:uuid>/primitive", methods=["GET"])
def getICMPrimitives(uuid: str):
    """ returns all ICM primitives (TODO - needs filter support)"""
    model = models[uuid]
    nodes = [
        CausalVariable(
            id=n[1]["id"],
            units=n[1]["units"],
            label=n[1]["label"],
            description=n[1]["description"],
            lastUpdated=n[1]["lastUpdated"],
            lastKnownValue=n[1]["lastKnownValue"],
            range=n[1]["range"],
        )
        for n in model.nodes(data=True)
    ]
    edges = [
        CausalRelationship(
            id=e[2]["id"],
            source=e[2]["source"],
            target=e[2]["target"],
            namespaces=e[2]["namespaces"],
            confidence=e[2]["confidence"],
            types=e[2]["types"],
            strength=e[2]["strength"],
            lastUpdated=e[2]["lastUpdated"],
            description=e[2]["description"],
            label=e[2]["label"],
            reinforcement=e[2]["reinforcement"],
        )
        for e in model.edges(data=True)
    ]
    return jsonify([asdict(x) for x in nodes + edges])


@bp.route("/icm/<string:uuid>/primitive", methods=["POST"])
def createICMPrimitive(uuid: str):
    """ create a new causal primitive"""
    pass


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["GET"])
def getICMPrimitive(uuid: str, prim_id: str):
    """ returns a specific causal primitive"""
    pass


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["PATCH"])
def updateICMPrimitive(uuid: str, prim_id: str):
    """ update an existing ICM primitive (can use this for disable?)"""
    pass


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["DELETE"])
def deleteICMPrimitive(uuid: str, prim_id: str):
    """ delete (disable) this ICM primitive"""
    pass


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["GET"]
)
def getEvidenceForID(uuid: str, prim_id: str):
    """ returns evidence for a causal primitive (needs pagination support)"""
    pass


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["POST"]
)
def attachEvidence(uuid: str, prim_id: str):
    """ attach evidence to a primitive"""
    pass


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["GET"])
def getEvidenceByID(uuid: str, evid_id: str):
    """ returns an individual piece of evidence"""
    pass


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["PATCH"])
def updateEvidence(uuid: str, evid_id: str):
    """ update evidence item"""
    pass


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["DELETE"])
def deleteEvidence(uuid: str, evid_id: str):
    """ delete evidence item"""
    pass


@bp.route("/icm/<string:uuid>/recalculate", methods=["POST"])
def recalculateICM(uuid: str):
    """ indication that it is safe to recalculate/recompose model after performing some number of CRUD operations"""
    pass


@bp.route("/icm/<string:uuid>/archive", methods=["POST"])
def archiveICM(uuid: str):
    """ archive an ICM"""
    pass


@bp.route("/icm/<string:uuid>/branch", methods=["POST"])
def branchICM(uuid: str):
    """ branch an ICM"""
    pass


@bp.route("/icm/fuse", methods=["POST"])
def fuseICMs():
    """ fuse two ICMs"""
    pass


@bp.route("/icm/<string:uuid>/sparql", methods=["POST"])
def query(uuid: str):
    """ Query the ICM using SPARQL"""
    pass


@bp.route("/icm/<string:uuid>/experiment/forwardProjection", methods=["POST"])
def forwardProjection(uuid: str):
    """ Execute a "what if" projection over the model"""
    pass


@bp.route("/icm/<string:uuid>/experiment", methods=["GET"])
def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    pass


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["GET"])
def getExperiment(uuid: str, exp_id: str):
    """ Fetch experiment results"""
    pass


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["DELETE"])
def deleteExperiment(uuid: str, exp_id: str):
    """ Delete experiment"""
    pass


@bp.route("/icm/<string:uuid>/traverse/<string:prim_id>", methods=["POST"])
def traverse(uuid: str, prim_id: str):
    """ traverse through the ICM using a breadth-first search"""
    pass


@bp.route("/version", methods=["GET"])
def getVersion():
    """ Get the version of the ICM API supported"""
    pass


@bp.route("/ping", methods=["GET"])
def ping():
    """ Get the health status of the ICM server"""
    pass
