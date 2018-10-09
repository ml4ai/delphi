from uuid import uuid4
import pickle
from datetime import datetime, date
from typing import Optional, List
from delphi.bmi import initialize
from flask import Flask, jsonify, request
from delphi.icm_api.models import *

app = Flask(__name__)

with open("delphi_cag.pkl", "rb") as f:
    model = pickle.load(f)
    initialize(model, "variables.csv")
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
    for e in model.edges(data=True):
        e[2]["id"] = uuid4()
        e[2]["namespaces"] = []
        e[2]["source"] = model.nodes[e[0]]["id"]
        e[2]["target"] = model.nodes[e[1]]["id"]
        e[2]["lastUpdated"] = today
        e[2]["types"] = ["causal"]
        e[2]["description"] = f"{e[0]} influences {e[1]}."
        e[2]["confidence"] = 1
        e[2]["label"] = f"{e[0]} influences {e[1]}."
        e[2]["strength"] = 1
        e[2]["reinforcement"] = "placeholder reinforcement"

models = {str(model.id): model}


@app.route("/icm", methods=["POST"])
def createNewICM():
    """ Create a new ICM"""
    pass


@app.route("/icm", methods=["GET"])
def listAllICMs():
    """ List all ICMs"""
    return jsonify(list(models.keys()))


@app.route("/icm/<string:uuid>", methods=["GET"])
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


@app.route("/icm/<string:uuid>", methods=["DELETE"])
def deleteICM(uuid: str):
    """ Deletes an ICM"""
    if uuid in models:
        del models[uuid]


@app.route("/icm/<string:uuid>", methods=["PATCH"])
def updateICMMetadata(uuid: str):
    """ Update the metadata for an existing ICM"""
    pass


@app.route("/icm/<string:uuid>/primitive", methods=["GET"])
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


@app.route("/icm/<string:uuid>/primitive", methods=["POST"])
def createICMPrimitive(uuid: str):
    """ create a new causal primitive"""
    pass


@app.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["GET"])
def getICMPrimitive(uuid: str, prim_id: str):
    """ returns a specific causal primitive"""
    pass


@app.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["PATCH"])
def updateICMPrimitive(uuid: str, prim_id: str):
    """ update an existing ICM primitive (can use this for disable?)"""
    pass


@app.route("/icm/<string:uuid>/primitive/<string:prim_id>", methods=["DELETE"])
def deleteICMPrimitive(uuid: str, prim_id: str):
    """ delete (disable) this ICM primitive"""
    pass


@app.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["GET"]
)
def getEvidenceForID(uuid: str, prim_id: str):
    """ returns evidence for a causal primitive (needs pagination support)"""
    pass


@app.route(
    "/icm/<string:uuid>/primitive/<string:prim_id>/evidence", methods=["POST"]
)
def attachEvidence(uuid: str, prim_id: str):
    """ attach evidence to a primitive"""
    pass


@app.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["GET"])
def getEvidenceByID(uuid: str, evid_id: str):
    """ returns an individual piece of evidence"""
    pass


@app.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["PATCH"])
def updateEvidence(uuid: str, evid_id: str):
    """ update evidence item"""
    pass


@app.route("/icm/<string:uuid>/evidence/<string:evid_id>", methods=["DELETE"])
def deleteEvidence(uuid: str, evid_id: str):
    """ delete evidence item"""
    pass


@app.route("/icm/<string:uuid>/recalculate", methods=["POST"])
def recalculateICM(uuid: str):
    """ indication that it is safe to recalculate/recompose model after performing some number of CRUD operations"""
    pass


@app.route("/icm/<string:uuid>/archive", methods=["POST"])
def archiveICM(uuid: str):
    """ archive an ICM"""
    pass


@app.route("/icm/<string:uuid>/branch", methods=["POST"])
def branchICM(uuid: str):
    """ branch an ICM"""
    pass


@app.route("/icm/fuse", methods=["POST"])
def fuseICMs():
    """ fuse two ICMs"""
    pass


@app.route("/icm/<string:uuid>/sparql", methods=["POST"])
def query(uuid: str):
    """ Query the ICM using SPARQL"""
    pass


@app.route("/icm/<string:uuid>/experiment/forwardProjection", methods=["POST"])
def forwardProjection(uuid: str):
    """ Execute a "what if" projection over the model"""
    pass


@app.route("/icm/<string:uuid>/experiment", methods=["GET"])
def getExperiments(uuid: str):
    """ list active (running or completed) experiments"""
    pass


@app.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["GET"])
def getExperiment(uuid: str, exp_id: str):
    """ Fetch experiment results"""
    pass


@app.route("/icm/<string:uuid>/experiment/<string:exp_id>", methods=["DELETE"])
def deleteExperiment(uuid: str, exp_id: str):
    """ Delete experiment"""
    pass


@app.route("/icm/<string:uuid>/traverse/<string:prim_id>", methods=["POST"])
def traverse(uuid: str, prim_id: str):
    """ traverse through the ICM using a breadth-first search"""
    pass


@app.route("/version", methods=["GET"])
def getVersion():
    """ Get the version of the ICM API supported"""
    pass


@app.route("/ping", methods=["GET"])
def ping():
    """ Get the health status of the ICM server"""
    pass


if __name__ == "__main__":
    app.run(debug=True)
