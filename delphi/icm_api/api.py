import json
from uuid import uuid4
import pickle
from datetime import date, timedelta
from typing import Optional, List
from itertools import product
from delphi.random_variables import LatentVar
from delphi.bmi import initialize, update
from delphi.execution import default_update_function
from delphi.utils import flatten
from flask import jsonify, request, Blueprint
from delphi.icm_api.models import *
from delphi.paths import data_dir
import numpy as np

bp = Blueprint("icm_api", __name__)


@bp.route("/icm", methods=["POST"])
def createNewICM():
    """ Create a new ICM"""
    return "", 415


@bp.route("/icm", methods=["GET"])
def listAllICMs():
    """ List all ICMs"""
    return jsonify([metadata.id for metadata in ICMMetadata.query.all()])


@bp.route("/icm/<string:uuid>", methods=["GET"])
def getICMByUUID(uuid: str):
    """ Fetch an ICM by UUID"""
    return jsonify(ICMMetadata.query.filter_by(id=uuid).first().deserialize())


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
    return "", 415


@bp.route("/icm/<string:uuid>/primitive", methods=["GET"])
def getICMPrimitives(uuid: str):
    """ returns all ICM primitives (TODO - needs filter support)"""
    # G = DelphiModel.query.filter_by(id=uuid).first()
    primitives = CausalPrimitive.query.filter_by(model_id=uuid).all()
    return jsonify([p.deserialize() for p in primitives])


@bp.route("/icm/<string:uuid>/primitive", methods=["POST"])
def createICMPrimitive(uuid: str):
    """ create a new causal primitive"""
    return "", 415


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["GET"])
def getICMPrimitive(uuid: str, prim_id: str):
    """ returns a specific causal primitive"""
    return "", 415


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["PATCH"])
def updateICMPrimitive(uuid: str, prim_id: str):
    """ update an existing ICM primitive (can use this for disable?)"""
    return "", 415


@bp.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["DELETE"])
def deleteICMPrimitive(uuid: str, prim_id: str):
    """ delete (disable) this ICM primitive"""
    return "", 415


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["GET"]
)
def getEvidenceForID(uuid: str, prim_id: str):
    """ returns evidence for a causal primitive (needs pagination support)"""
    return "", 415


@bp.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["POST"]
)
def attachEvidence(uuid: str, prim_id: str):
    """ attach evidence to a primitive"""
    return "", 415


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["GET"])
def getEvidenceByID(uuid: str, evid_id: str):
    """ returns an individual piece of evidence"""
    return "", 415


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["PATCH"])
def updateEvidence(uuid: str, evid_id: str):
    """ update evidence item"""
    return "", 415


@bp.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["DELETE"])
def deleteEvidence(uuid: str, evid_id: str):
    """ delete evidence item"""
    return "", 415


@bp.route("/icm/<string:uuid>/recalculate", methods=["POST"])
def recalculateICM(uuid: str):
    """ indication that it is safe to recalculate/recompose model after performing some number of CRUD operations"""
    return "", 415


@bp.route("/icm/<string:uuid>/archive", methods=["POST"])
def archiveICM(uuid: str):
    """ archive an ICM"""
    return "", 415


@bp.route("/icm/<string:uuid>/branch", methods=["POST"])
def branchICM(uuid: str):
    """ branch an ICM"""
    return "", 415


@bp.route("/icm/fuse", methods=["POST"])
def fuseICMs():
    """ fuse two ICMs"""
    return "", 415


@bp.route("/icm/<string:uuid>/sparql", methods=["POST"])
def query(uuid: str):
    """ Query the ICM using SPARQL"""
    return "", 415


@bp.route("/icm/<string:uuid>/experiment", methods=["POST"])
def createExperiment(uuid: str):
    """ Execute an experiment over the model"""
    G = DelphiModel.query.filter_by(id=uuid).first().model
    data = json.loads(request.data)
    default_latent_var_value = 1.0
    for n in G.nodes(data=True):
        n[1]["rv"] = LatentVar(n[0])
        n[1]["update_function"] = default_update_function
        rv = n[1]["rv"]
        rv.dataset = [default_latent_var_value for _ in range(G.res)]
        if n[1].get("indicators") is not None:
            for ind in n[1]["indicators"]:
                ind.dataset = np.ones(G.res) * ind.mean

        rv.partial_t = 0.0
        for variable in data["interventions"]:
            if n[1]["id"] == variable["id"]:
                rv.partial_t = variable["values"]["value"]
                break

    id = str(uuid4())
    experiment = ForwardProjection(baseType="ForwardProjection", id=id)
    db.session.add(experiment)
    db.session.commit()

    result = ForwardProjectionResult(id=id, baseType="ForwardProjectionResult")
    db.session.add(result)
    db.session.commit()

    # TODO Right now, we just take the timestamp of the first intervention -
    # we might need a more generalizable approach.

    d = date(*map(int, data["interventions"][0]["values"]["time"].split("-")))

    increment_month = lambda m: (m + 1) % 12 if (m + 1) % 12 != 0 else 12

    for i in range(data["projection"]["numSteps"]):
        if data["projection"]["stepSize"] == "MONTH":
            # TODO Right now, the day is hardcoded by default to be 1. Need to
            # figure out later what it means to step forward by one month (i.e.
            # increment month by one or number of days by ~30, or ..?
            d = date(d.year, increment_month(d.month), 1)
        elif data["projection"]["stepSize"] == "YEAR":
            d = date(d.year + 1, d.month, d.day)

        update(G)

        for n in G.nodes(data=True):
            CausalVariable.query.filter_by(
                id=n[1]["id"]
            ).first().lastUpdated = d.isoformat()
            result.results.append(
                {
                    "id": n[1]["id"],
                    "baseline": {
                        "active": "ACTIVE",
                        "time": d.isoformat(),
                        "value": 1,
                    },
                    "intervened": {
                        "active": "ACTIVE",
                        "time": d.isoformat(),
                        "value": np.mean(n[1]["rv"].dataset),
                    },
                }
            )
    db.session.add(result)
    db.session.commit()

    return jsonify(
        {
            "id": experiment.id,
            "message": "Forward projection sent successfully",
        }
    )


@bp.route("/icm/<string:uuid>/experiment", methods=["GET"])
def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    return jsonify([x.deserialize() for x in Experiment.query.all()])


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["GET"])
def getExperiment(uuid: str, exp_id: str):
    """ Fetch experiment results"""
    experimentResult = ForwardProjectionResult.query.filter_by(
        id=exp_id
    ).first()
    return jsonify(experimentResult.deserialize())


@bp.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["DELETE"])
def deleteExperiment(uuid: str, exp_id: str):
    """ Delete experiment"""
    return "", 415


@bp.route("/icm/<string:uuid>/traverse/<string:prim_id>", methods=["POST"])
def traverse(uuid: str, prim_id: str):
    """ traverse through the ICM using a breadth-first search"""
    return "", 415


@bp.route("/version", methods=["GET"])
def getVersion():
    """ Get the version of the ICM API supported"""
    return "", 415


@bp.route("/ping", methods=["GET"])
def ping():
    """ Get the health status of the ICM server"""
    return "", 415
