# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
from math import exp, sqrt
from uuid import uuid4
import pickle
from datetime import date, timedelta, datetime
import dateutil
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from statistics import median, mean
from delphi.cpp.DelphiPython import AnalysisGraph
from delphi.utils import lmap
from flask import jsonify, request, Blueprint, current_app
from delphi.db import engine
from delphi.apps.rest_api import db, executor
from delphi.apps.rest_api.models import *
from flask import current_app

bp = Blueprint("rest_api", __name__)

# ============
# CauseMos API
# ============


@bp.route("/delphi/create-model", methods=["POST"])
def createNewModel():
    """ Create a new Delphi model. """
    data = json.loads(request.data)

    if os.environ.get("CI") == "true":
        # When running in a continuous integration run, we set the sampling
        # resolution to be small to prevent timeouts.
        res = 5
    elif os.environ.get("DELPHI_N_SAMPLES") is not None:
        # We also enable setting the sampling resolution through the
        # environment variable "DELPHI_N_SAMPLES", for development and testing
        # purposes.
        res = int(os.environ["DELPHI_N_SAMPLES"])
    else:
        # If neither "CI" or "DELPHI_N_SAMPLES" is set, we default to a
        # sampling resolution of 1000.

        # TODO - we might want to set the default sampling resolution with some
        # kind of heuristic, based on the number of nodes and edges. - Adarsh
        res = 1000

    G = AnalysisGraph.from_causemos_json_string(request.data, res)
    model = DelphiModel(
        id=data["id"], model=G.serialize_to_json_string(verbose=False)
    )
    db.session.merge(model)
    db.session.commit()
    # edge_weights = G.get_edge_weights_for_causemos_viz()
    # return jsonify({"status": "success", "relations": edge_weights})
    response = G.generate_create_model_response()
    return jsonify(response)


@bp.route("/delphi/models/<string:modelID>/experiments", methods=["POST"])
def createCausemosExperiment(modelID):
    model = DelphiModel.query.filter_by(id=modelID).first().model
    G = AnalysisGraph.deserialize_from_json_string(model, verbose=False)
    request_body = request.get_json()
    experiment_type = request_body["experimentType"]
    experiment_id = str(uuid4())

    result = CauseMosAsyncExperimentResult(
        id=experiment_id,
        baseType="CauseMosAsyncExperimentResult",
        experimentType="PROJECTION",
        status = "in progress",
    )


    db.session.add(result)
    db.session.commit()
    def runExperiment():
        if experiment_type == "PROJECTION":
            causemos_experiment_result = G.run_causemos_projection_experiment(
                request.data
            )


        result = CauseMosAsyncExperimentResult.query.filter_by(
            id=experiment_id
        ).first()
        startTime = causemos_experiment_result[0]
        endTime = causemos_experiment_result[1]
        numTimesteps = causemos_experiment_result[2]

        # A rudimentary test to see if the projection failed. We check whether
        # the number time steps is equal to the number of elements in the first
        # concept's time series.
        if len(list(causemos_experiment_result[3].values())[0]) < numTimesteps:
            result.status = "failed"
            result.results = {}
        else:
            result.status = "completed"

            timesteps_nparr = np.round(
                np.linspace(startTime, endTime, numTimesteps)
            )

            # The calculation of the 95% confidence interval about the median is
            # taken from:
            # https://www.ucl.ac.uk/child-health/short-courses-events/ \
            #     about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
            n = G.get_res()
            lower_rank = int((n - 1.96 * sqrt(n)) / 2)
            upper_rank = int((2 + n + 1.96 * sqrt(n)) / 2)

            lower_rank = 0 if lower_rank < 0 else lower_rank
            upper_rank = n - 1 if upper_rank >= n else upper_rank
            result.results = {"data": []}
            for (
                conceptname,
                timestamp_sample_matrix,
            ) in causemos_experiment_result[3].items():
                data_dict = {}
                data_dict["concept"] = conceptname
                data_dict["values"] = []
                data_dict["confidenceInterval"] = {"upper": [], "lower": []}
                for i, time_step in enumerate(timestamp_sample_matrix):
                    time_step.sort()
                    l = len(time_step) // 2
                    median_value = (
                        time_step[l]
                        if len(time_step) % 2
                        else (time_step[l] + time_step[l - 1]) / 2
                    )
                    lower_limit = time_step[lower_rank]
                    upper_limit = time_step[upper_rank]

                    value_dict = {
                        "timestamp": timesteps_nparr[i],
                        "value": median_value,
                    }

                    data_dict["values"].append(value_dict.copy())
                    value_dict.update({"value": lower_limit})
                    data_dict["confidenceInterval"]["lower"].append(
                        value_dict.copy()
                    )
                    value_dict.update({"value": upper_limit})
                    data_dict["confidenceInterval"]["upper"].append(
                        value_dict.copy()
                    )
                result.results["data"].append(data_dict)

        db.session.add(result)
        db.session.commit()

    executor.submit_stored(experiment_id, runExperiment)

    return jsonify({"experimentId": experiment_id})


@bp.route(
    "/delphi/models/<string:modelID>/experiments/<string:experimentID>",
    methods=["GET"],
)
def getExperimentResults(modelID: str, experimentID: str):
    """ Fetch experiment results"""
    result = CauseMosAsyncExperimentResult.query.filter_by(
        id=experimentID
    ).first()

    response = {
        "modelId": modelID,
        "experimentId": experimentID,
        "experimentType": result.experimentType,
        "status": result.status,
        "results": result.results
    }
    return jsonify(response)
