# -*- coding: utf-8 -*-
import os
import re
import json
from math import exp, sqrt
from uuid import uuid4
import pickle
from datetime import date, timedelta, datetime
import dateutil
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from typing import Optional, List
from itertools import product
from statistics import median, mean
from delphi.cpp.DelphiPython import AnalysisGraph
from delphi.random_variables import LatentVar
from delphi.utils import flatten, lmap
from flask import jsonify, request, Blueprint
from delphi.db import engine
from delphi.apps.rest_api import db
from delphi.apps.rest_api.models import *
from delphi.random_variables import Indicator
import numpy as np
from flask import current_app
import scipy.stats

bp = Blueprint("rest_api", __name__)

# ============
# CauseMos API
# ============


PLACEHOLDER_UNIT = "No units specified."


@bp.route("/delphi/models", methods=["GET"])
def listAllModels():
    """ Return UUIDs for all the models in the database. """

    query = (
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='icmmetadata'"
    )
    if list(engine.execute(query)) == []:
        return jsonify([])
    else:
        return jsonify([metadata.id for metadata in ICMMetadata.query.all()])


@bp.route("/delphi/create-model", methods=["POST"])
def createNewModel():
    """ Create a new Delphi model. """
    data = json.loads(request.data)
    G = AnalysisGraph.from_uncharted_json_string(request.data)
    G.id = data["id"]
    model=DelphiModel(id=data["id"], model = G.to_json_string())
    db.session.add(model)
    db.session.commit()
    return jsonify({"status": "success"})


@bp.route("/delphi/search", methods=["POST"])
def getIndicators():
    """
    Given a list of concepts, this endpoint returns their respective matching
    indicators. The search parameters are:
    - start/end: To specify search criteria in years (YYYY)
    - geolocation: To match indicators with matching geolocation
    - func: To apply a transform function onto the raw indicator values
    - concepts: List of concepts
    - outputResolution: month/year

    The search will return a listing of matching indicators, sorted by
    similarity score. For each concept a maximum of 10 indicator matches will
    be returned. If there are no matches for a given concept an empty array is
    returned.
    """

    # args = request.args
    args = request.get_json()

    func_dict = {
        "mean": mean,
        "median": median,
        "max": max,
        "min": min,
        "raw": lambda x: x,
    }

    output_dict = {}
    for concept in args.get("concepts"):
        output_dict[concept] = []
        query = (
            "select `Concept`, `Source`, `Indicator`, `Score` "
            "from concept_to_indicator_mapping "
            f"where `Concept` like '{concept}'"
        )
        for indicator_mapping in engine.execute(query):
            variable_name = indicator_mapping["Indicator"].replace("'", "''")
            query_parts = [
                f"select * from indicator",
                f"where `Variable` like '{variable_name}'",
            ]
            outputResolution = args.get("outputResolution")
            start = args.get("start")
            end = args.get("end")
            func = args.get("func", "raw")

            if outputResolution is not None:
                query_parts.append(f"and `{outputResolution}` is not null")
            if start is not None:
                query_parts.append(f"and `Year` > {start}")
            if end is not None:
                query_parts.append(f"and `Year` < {end}")

            records = list(engine.execute(" ".join(query_parts)))
            value_dict = {}
            source = "Unknown"
            if func == "raw":
                for r in records:
                    unit, value, year, month, source = (
                        r["Unit"],
                        r["Value"],
                        r["Year"],
                        r["Month"],
                        r["Source"],
                    )

                    value = float(re.findall(r"-?\d+\.?\d*", value)[0])

                    # Sort of a hack - some of the variables in the tables we
                    # process don't have units specified, so we put a
                    # placeholder string to get it to work with CauseMos.
                    if unit is None:
                        unit = PLACEHOLDER_UNIT
                    _dict = {
                        "year": year,
                        "month": month,
                        "value": float(value),
                        "source": source,
                    }

                    if unit not in value_dict:
                        value_dict[unit] = [_dict]
                    else:
                        value_dict[unit].append(_dict)
                value = value_dict
            else:
                for r in records:
                    unit, value, source = r["Unit"], r["Value"], r["Source"]

                    if unit is None:
                        unit = PLACEHOLDER_UNIT

                    value = float(re.findall(r"-?\d+\.?\d*", value)[0])

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
                    "source": source,
                }
            )

    return jsonify(output_dict)


@bp.route("/delphi/models/<string:modelID>/projection", methods=["POST"])
def createProjection(modelID):

    model = DelphiModel.query.filter_by(id=modelID).first().model
    G = AnalysisGraph.from_json_string(model)

    projection_result = G.generate_projection(request.data, resolution = 200)
    print(projection_result)

    id = str(uuid4())

    result = CauseMosForwardProjectionResult(
        id=id, baseType="CauseMosForwardProjectionResult"
    )
    result.results = {
        G[n].name: {
            "values": [],
            "confidenceInterval": {"upper": [], "lower": []},
        }
        for n in G
    }
    db.session.add(result)

    data = json.loads(request.data)
    startTime = data["startTime"]
    #d = dateutil.parser.parse(f"{startTime['year']} {startTime['month']}")
    d = parse(f"{startTime['year']} {startTime['month']}")

    τ = 1.0  # Time constant to control the rate of the decay

    # # From https://www.ucl.ac.uk/child-health/short-courses-events/
    # #     about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
    n = G.res
    lower_rank = int((n - 1.96 * sqrt(n)) / 2)
    upper_rank = int((2 + n + 1.96 * sqrt(n)) / 2)

    lower_rank = 0 if lower_rank < 0 else lower_rank
    upper_rank = n-1 if upper_rank >= n else upper_rank

    for concept, samples in projection_result.items():
        for ts in range(int(data["timeStepsInMonths"])):
            d = d + relativedelta(months=1)

            median_value = median(samples[ts])
            lower_limit = samples[ts][lower_rank]
            upper_limit = samples[ts][upper_rank]

            value_dict = {
                "year": d.year,
                "month": d.month,
                "value": median_value,
            }

            result.results[concept]["values"].append(value_dict.copy())
            value_dict.update({"value": lower_limit})
            result.results[concept]["confidenceInterval"]["lower"].append(
                value_dict.copy()
            )
            value_dict.update({"value": upper_limit})
            result.results[concept]["confidenceInterval"]["upper"].append(
                value_dict.copy()
            )

    db.session.add(result)
    db.session.commit()

    return jsonify({"experimentId": id})

    # What does this do?
    #G.update(update_indicators=False, dampen=True, τ=τ)


@bp.route(
    "/delphi/models/<string:modelID>/experiment/<string:experimentID>",
    methods=["GET"],
)
def getExperimentResults(modelID: str, experimentID: str):
    """ Fetch experiment results"""
    experimentResult = CauseMosForwardProjectionResult.query.filter_by(
        id=experimentID
    ).first()
    return jsonify(experimentResult.deserialize())


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
    G.initialize(initialize_indicators=False)
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

    τ = 1.0  # Time constant to control the rate of the decay
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
                            "value": median([s[n[0]] for s in G.s0]),
                        },
                    },
                }
            )

        G.update(update_indicators=False, dampen=True, τ=τ)

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
