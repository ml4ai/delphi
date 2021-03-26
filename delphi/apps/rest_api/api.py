# -*- coding: utf-8 -*-
import os
import numpy as np
from math import sqrt
from delphi.cpp.DelphiPython import AnalysisGraph, InitialBeta, InitialDerivative
from flask import jsonify, request, Blueprint, current_app
from delphi.apps.rest_api.models import *
import multiprocessing
multiprocessing.set_start_method('fork')

bp = Blueprint("rest_api", __name__)

# ============
# CauseMos API
# ============


def train_model(G, modelID, sampling_resolution, burn, app):
    G.run_train_model(sampling_resolution,
                      burn,
                      InitialBeta.ZERO,
                      InitialDerivative.DERI_ZERO)
    #G.write_model_to_db(modelID)
    with app.app_context():
        model = DelphiModel(
            id=modelID, model=G.serialize_to_json_string(verbose=False)
        )
        db.session.merge(model)
        db.session.commit()


@bp.route("/delphi/create-model", methods=["POST"])
def createNewModel():
    """ Create a new Delphi model. """
    if os.environ.get("CI") == "true":
        # When running in a continuous integration run, we set the sampling
        # resolution to be small to prevent timeouts.
        kde_kernels = 5
        sampling_resolution = 5
        burn = 5
    elif os.environ.get("DELPHI_N_SAMPLES") is not None:
        # We also enable setting the sampling resolution through the
        # environment variable "DELPHI_N_SAMPLES", for development and testing
        # purposes.
        # NOTE: When creating, this environment variable is named incorrectly!
        kde_kernels = int(os.environ["DELPHI_N_SAMPLES"])
        sampling_resolution = 100
        burn = 100
    else:
        # If neither "CI" or "DELPHI_N_SAMPLES" is set, we default to a
        # sampling resolution of 1000.

        # TODO - we might want to set the default sampling resolution with some
        # kind of heuristic, based on the number of nodes and edges. - Adarsh
        kde_kernels = 1000
        sampling_resolution = 1000
        burn = 10000

    data = json.loads(request.data)

    G = AnalysisGraph.from_causemos_json_string(request.data,
                                                belief_score_cutoff=0,
                                                grounding_score_cutoff=0,
                                                kde_kernels=kde_kernels)

    model = DelphiModel(
        id=data["id"], model=G.serialize_to_json_string(verbose=False)
    )
    db.session.merge(model)
    db.session.commit()

    response = json.loads(G.generate_create_model_response())

    # executor.submit_stored(data["id"], train_model, G, data["id"])
    try:
        proc = multiprocessing.Process(target=train_model,
                                       args=(G, data["id"], sampling_resolution, burn, current_app),
                                       name='training')
        proc.start()
    except multiprocessing.ProcessError:
        print("Error: unable to start training process")
        response['status'] = 'server error: training'

    return jsonify(response)


@bp.route("/delphi/models/<string:modelID>", methods=["GET"])
def getModelStatus(modelID):
    query_result = DelphiModel.query.filter_by(id=modelID).first()

    if not query_result:
        return jsonify(json.loads('{"status": "invalid model id"}'))

    model = query_result.model
    G = AnalysisGraph.deserialize_from_json_string(model, verbose=False)

    response = json.loads(G.generate_create_model_response())
    return jsonify(response)


def runProjectionExperiment(request, experiment_id, G, app):
    request_body = request.get_json()

    startTime = request_body["experimentParam"]["startTime"]
    endTime = request_body["experimentParam"]["endTime"]
    numTimesteps = request_body["experimentParam"]["numTimesteps"]

    causemos_experiment_result = G.run_causemos_projection_experiment_from_json_string(
        request.data
    )

    with app.app_context():
        result = CauseMosAsyncExperimentResult.query.filter_by(
            id=experiment_id
        ).first()

    # A rudimentary test to see if the projection failed. We check whether
    # the number time steps is equal to the number of elements in the first
    # concept's time series.
    if len(list(causemos_experiment_result.values())[0]) < numTimesteps:
        result.status = "failed"
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
        ) in causemos_experiment_result.items():
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

    with app.app_context():
        db.session.merge(result)
        db.session.commit()


def runExperiment(request, experiment_id, experiment_type, model, app):
    G = AnalysisGraph.deserialize_from_json_string(model, verbose=False)

    if experiment_type == "PROJECTION":
        runProjectionExperiment(request, experiment_id, G, app)
    elif experiment_type == "GOAL_OPTIMIZATION":
        # Not yet implemented
        pass
    elif experiment_type == "SENSITIVITY_ANALYSIS":
        # Not yet implemented
        pass
    elif experiment_type == "MODEL_VALIDATION":
        # Not yet implemented
        pass
    elif experiment_type == "BACKCASTING":
        # Not yet implemented
        pass
    else:
        # Unknown experiment type
        pass


@bp.route("/delphi/models/<string:modelID>/experiments", methods=["POST"])
def createCausemosExperiment(modelID):
    request_body = request.get_json()

    query_result = DelphiModel.query.filter_by(id=modelID).first()

    if not query_result:
        return jsonify({"experimentId": "invalid model id"})

    model = query_result.model
    trained = json.loads(model)["trained"]

    if not trained:
        return jsonify({"experimentId": "model not trained"})

    experiment_type = request_body["experimentType"]
    experiment_id = str(uuid4())

    result = CauseMosAsyncExperimentResult(
        id=experiment_id,
        baseType="CauseMosAsyncExperimentResult",
        experimentType=experiment_type,
        status="in progress",
        results={}
    )

    db.session.add(result)
    db.session.commit()

    # executor.submit_stored(experiment_id, runExperiment, request,
    #                        experiment_id, experiment_type, model)
    try:
        proc = multiprocessing.Process(target=runExperiment,
                                       args=(request,
                                             experiment_id, experiment_type, model, current_app),
                                       name='experiment')
        proc.start()
    except multiprocessing.ProcessError:
        print("Error: unable to start experiment process")
        return jsonify({"experimentId": 'server error: experiment'})

    return jsonify({"experimentId": experiment_id})


@bp.route(
    "/delphi/models/<string:modelID>/experiments/<string:experimentID>",
    methods=["GET"],
)
def getExperimentResults(modelID: str, experimentID: str):
    """ Fetch experiment results"""
    # NOTE: I saw some weird behavior when we request results for an invalid
    # experiment ID just after running an experiment. The trained model seemed
    # to be not saved to the database. The model got re-trained from scratch
    # on a subsequent experiment after the initial experiment and the invalid
    # experiment result request. When I added a sleep between the initial
    # create experiment and the invalid result request this re-training did not
    # occur.
    result = CauseMosAsyncExperimentResult.query.filter_by(
        id=experimentID
    ).first()

    if result:
        experimentType = result.experimentType
        status = result.status
        results = result.results
    else:
        # experimentID not in database. Should be an incorrect experimentID
        experimentType = "UNKNOWN"
        status = "invalid experiment id"
        results = {}

    response = {
        "modelId": modelID,
        "experimentId": experimentID,
        "experimentType": experimentType,
        "status": status,
        "results": results
    }
    return jsonify(response)
