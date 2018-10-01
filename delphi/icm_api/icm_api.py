import uuid
import pickle
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from flask import Flask, jsonify, request

app = Flask(__name__)


@dataclass
class ICMMetadata:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    icmProvider: Optional[str] = "UAZ"
    title: Optional[str] = "Delphi Model"
    estimatedNumberOfPrimitives: Optional[int] = 0
    version: Optional[int] = 1
    created: Optional[str] = field(default_factory=lambda: str(datetime.now()))
    createdByUser: Optional[str] = None


@dataclass
class CausalPrimitive:
    """ Top level object that contains common properties that would apply to any
        causal primitive (variable, relationship, etc.) """

    id: str
    editable: bool = True
    baseType: str = "string"


@dataclass
class Range:
    """Top level range object used in a CausalVariable. """

    baseType: str = "string"


@dataclass
class CausalVariable(CausalPrimitive):
    """ API definition of a causal variable. """

    range: Range = Range()

@dataclass
class Evidence:
    """Object that holds a reference to evidence 
    (either KO from TA1 or human provided)."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    link: str
    description: str = "description placeholder"
    category: str
    rank: int

@dataclass
class CausalRelationship(CausalPrimitive):
    """ API definition of a causal relationship. """

    source: str = "source"
    target: str = "target"


with open("delphi_cag.pkl", "rb") as f:
    model = pickle.load(f)

models = {str(model.id): model}


@app.route("/icm")
def listAllICMs():
    """ List all ICMs """
    return jsonify(list(models.keys()))


@app.route("/icm/<string:uuid>", methods=["GET"])
def getICMByUUID(uuid):
    """ Returns the metadata associated with an ICM. """
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
def deleteICM(uuid):
    """ Delete an ICM by UUID. """
    if uuid in models:
        del models[uuid]


@app.route("/icm/<string:uuid>/primitive")
def getICMPrimitives(uuid):
    """returns all ICM primitives (TODO - needs filter support)"""
    model = models[uuid]
    nodes = [CausalVariable(id=n) for n in model.nodes()]
    edges = [
        CausalRelationship(id="__".join(e), source=e[0], target=e[1])
        for e in model.edges()
    ]
    return jsonify([asdict(x) for x in nodes + edges])


if __name__ == "__main__":
    app.run(debug=True)
