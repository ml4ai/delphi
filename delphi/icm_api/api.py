import json
from math import exp
from uuid import uuid4
import pickle
from datetime import date, timedelta, datetime
import dateutil
from dateutil.relativedelta import relativedelta
from typing import Optional, List
from itertools import product
from delphi.random_variables import LatentVar
from delphi.utils import flatten
from flask import jsonify, request, Blueprint
from delphi.icm_api import db
from delphi.icm_api.models import *
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
    _metadata = ICMMetadata.query.filter_by(id=uuid).first().deserialize()
    del _metadata["model_id"]
    return jsonify(_metadata)


@bp.route("/icm/<string:uuid>", methods=["DELETE"])
def deleteICM(uuid: str):
    """ Deletes an ICM"""
    _metadata = ICMMetadata.query.filter_by(id=uuid).first()
    db.session.delete(_metadata)
    db.session.commit()
    return ("", 204)


@bp.route("/icm/<string:uuid>", methods=["PATCH"])
def updateICMMetadata(uuid: str):
    """ Update the metadata for an existing ICM"""
    return "", 415


@bp.route("/icm/<string:uuid>/primitive", methods=["GET"])
def getICMPrimitives(uuid: str):
    """ returns all ICM primitives (TODO - needs filter support)"""
    primitives = [
        p.deserialize()
        for p in CausalPrimitive.query.filter_by(model_id=uuid).all()
    ]
    for p in primitives:
        del p["model_id"]
    return jsonify(primitives)


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
    evidences = [
        evidence.deserialize()
        for evidence in Evidence.query.filter_by(
            causalrelationship_id=prim_id
        ).all()
    ]
    for evidence in evidences:
        del evidence["causalrelationship_id"]

    return jsonify(evidences)


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
    data = json.loads(request.data)
    G = DelphiModel.query.filter_by(id=uuid).first().model
    G.initialize(initialize_indicators = False)
    for n in G.nodes(data=True):
        rv = n[1]["rv"]
        rv.partial_t = 0.0
        for variable in data["interventions"]:
            if n[1]["id"] == variable["id"]:
                # TODO : Right now, we are only taking the first value in the
                # "values" list. Need to generalize this so that you can have
                # multiple interventions at different times.

                # TODO : The subtraction of 1 is a TEMPORARY PATCH to address
                # the mismatch in semantics between the ICM API and the Delphi
                # model. MUST FIX ASAP.
                rv.partial_t = variable["values"]["value"]["value"] - 1
                for s0 in G.s0:
                    s0[f"∂({n[0]})/∂t"] = rv.partial_t
                break

    id = str(uuid4())
    experiment = ForwardProjection(baseType="ForwardProjection", id=id)
    db.session.add(experiment)
    db.session.commit()

    result = ForwardProjectionResult(id=id, baseType="ForwardProjectionResult")
    db.session.add(result)
    db.session.commit()

    d = dateutil.parser.parse(data["projection"]["startTime"])

    n_timesteps = data["projection"]["numSteps"]

    for i in range(n_timesteps):
        if data["projection"]["stepSize"] == "MONTH":
            d = d + relativedelta(months=1)
        elif data["projection"]["stepSize"] == "YEAR":
            d = d + relativedelta(years=1)

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
                        "value": {"baseType": "FloatValue", "value": 1.0},
                    },
                    "intervened": {
                        "active": "ACTIVE",
                        "time": d.isoformat(),
                        "value": {
                            "baseType": "FloatValue",
                            "value": np.median([s[n[0]] for s in G.s0]),
                        },
                    },
                }
            )

        G.update(update_indicators=False)

        # Hack for 12-month evaluation - have the partial derivative decay over
        # time to restore equilibrium

        tau = 1.0 # Time constant to control the rate of the decay
        for n in G.nodes(data=True):
            for variable in data["interventions"]:
                if n[1]["id"] == variable["id"]:
                    rv = n[1]["rv"]
                    for s0 in G.s0:
                        s0[f"∂({n[0]})/∂t"] = rv.partial_t * exp(-tau*i)

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
