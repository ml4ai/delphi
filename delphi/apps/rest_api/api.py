# -*- coding: utf-8 -*-
import os
import json
from math import exp
from uuid import uuid4
import pickle
from datetime import date, timedelta, datetime
import dateutil
from dateutil.relativedelta import relativedelta
from typing import Optional, List
from itertools import product
from delphi.AnalysisGraph import AnalysisGraph
from delphi.random_variables import LatentVar
from delphi.utils import flatten, lmap
from flask import jsonify, request, Blueprint
from delphi.db import engine
from delphi.apps.rest_api import db
from delphi.apps.rest_api.models import *
from delphi.random_variables import Indicator
import numpy as np
from flask import current_app

bp = Blueprint("rest_api", __name__)

# ============
# CauseMos API
# ============


@bp.route("/delphi/models", methods=["GET"])
def listAllModels():
    """ Return UUIDs for all the models in the database. """
    if (
        list(
            engine.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='icmmetadata'"
            )
        )
        == []
    ):
        return jsonify([])
    else:
        return jsonify([metadata.id for metadata in ICMMetadata.query.all()])


@bp.route("/delphi/models", methods=["POST"])
def createNewModel():
    """ Create a new Delphi model. """
    data = json.loads(request.data)
    G = AnalysisGraph.from_uncharted_json_serialized_dict(data)
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.id = data["model_id"]
    G.to_sql(app=current_app)
    return jsonify({"status": "success"})


@bp.route("/delphi/models/<string:model_id>/indicators", methods=["GET"])
def getIndicators(model_id: str):
    """ Search for indicator candidates pertaining to :model_id.

    The search options include:
        - start/end: To specify search criteria in years (YYYY)
        - geolocation: To match indicators with matching geolocation
        - func: To apply a transform function onto the raw indicator values
        - concept: To search for specific concept, if omitted search across all concepts within the model

    The search will return a listing of matching indicators, sorted by
    similarity score. For each concept a maximum of 10 indicator matches will
    be returned. If there are no matches for a given concept an empty array is
    returned.
    """
    concept = request.args.get("concept")
    func_dict = {
        "mean": np.mean,
        "median": np.median,
        "max": max,
        "min": min,
        "raw": lambda x: x,
    }

    if concept is not None:
        concepts = [concept]
    else:
        concepts = [
            v.deserialize()["description"]
            for v in CausalVariable.query.filter_by(model_id=model_id).all()
        ]

    output_dict = {}
    for concept in concepts:
        output_dict[concept] = []
        query_parts = [
            "select `Concept`, `Source`, `Indicator`, `Score`",
            "from concept_to_indicator_mapping",
            f"where `Concept` like '{concept}'",
        ]
        for indicator_mapping in engine.execute(" ".join(query_parts)):
            query = (
                f"select * from indicator"
                f" where `Variable` like '{indicator_mapping['Indicator']}'"
            )
            records = list(engine.execute(query))
            func = request.args.get("func", "raw")
            value_dict = {}
            if func == "raw":
                for r in records:
                    unit, value, year, source = r["Unit"], r["Value"], r["Year"], r["Source"]
                    if unit not in value_dict:
                        value_dict[unit] = [
                            {"year": year, "value": float(value), "source": source}
                        ]
                    else:
                        value_dict[unit].append(
                            {"year": year, "value": float(value), "source": source}
                        )
                value = value_dict
            else:
                for r in records:
                    unit, value, source = r["Unit"], r["Value"], r["Source"]
                    # HACK! if the variables have the same names but different
                    # sources, this will only give the most recent source
                    if unit not in value_dict:
                        value_dict[unit] = [value]
                    else:
                        value_dict[unit].append(value)

                value = {
                    unit: func_dict[func](lmap(float, values))
                    for unit, values in value_dict.items()
                }

            output_dict[concept].append(
                {
                    "name": indicator_mapping["Indicator"],
                    "score": indicator_mapping["Score"],
                    "value": value,
                    "source": source
                }
            )

    return jsonify(output_dict)


@bp.route("/delphi/models/<string:model_id>/export", methods=["POST"])
def exportModel(model_id: str):
    """ Sends quantification information to be attached to an existing model.
    Sends concept-to-indicator mappings.

    Input: A map object of concepts and their associated
           indicator/indicator-values
    Output: Request status
    """
    data = json.loads(request.data)
    G = DelphiModel.query.filter_by(id=model_id).first().model
    # Tag this model for display in the CauseWorks interface
    G.tag_for_CX = True
    for concept, indicator in data["concept_indicator_map"].items():
        G.nodes[concept]["indicators"] = {
            concept: Indicator(
                concept, indicator["source"], value=indicator["value"]
            )
        }

    return jsonify({"status": "success"})


# ============
# ICM API
# ============


@bp.route("/icm", methods=["POST"])
def createNewICM():
    """ Create a new ICM"""
    data = json.loads(request.data)
    G = AnalysisGraph.from_uncharted_json_serialized_dict(data)
    G.assemble_transition_model_from_gradable_adjectives()
    G.sample_from_prior()
    G.to_sql(app=current_app)
    _metadata = ICMMetadata.query.filter_by(id=G.id).first().deserialize()
    del _metadata["model_id"]
    return jsonify(_metadata)


@bp.route("/icm", methods=["GET"])
def listAllICMs():
    """ List all ICMs"""
    if (
        list(
            engine.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='icmmetadata'"
            )
        )
        == []
    ):
        return jsonify([])
    else:
        ids = [metadata.id for metadata in ICMMetadata.query.all()]
        ids.reverse()
        return jsonify(ids)



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
    G = DelphiModel.query.filter_by(id=uuid).first()
    for primitive in CausalPrimitive.query.filter_by(model_id=uuid).all():
        db.session.delete(primitive)
    db.session.delete(_metadata)
    db.session.delete(G)
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
    data = request.get_json()
    G = DelphiModel.query.filter_by(id=uuid).first().model
    if os.environ.get("TRAVIS") is not None:
        config_file = "bmi_config.txt"
    else:
        if not os.path.exists("/tmp/delphi"):
            os.makedirs("/tmp/delphi", exist_ok=True)
        config_file = "/tmp/delphi/bmi_config.txt"

    G.create_bmi_config_file(config_file)
    G.initialize(initialize_indicators=False, config_file=config_file)
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

        tau = 1.0  # Time constant to control the rate of the decay
        for n in G.nodes(data=True):
            for variable in data["interventions"]:
                if n[1]["id"] == variable["id"]:
                    rv = n[1]["rv"]
                    for s0 in G.s0:
                        s0[f"∂({n[0]})/∂t"] = rv.partial_t * exp(-tau * i)

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
