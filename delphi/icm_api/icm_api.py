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
    icmProvider: Optional[str] = 'UAZ'
    title: Optional[str] = "Delphi Model"
    version: Optional[int] = 1
    created: Optional[str] = field(default_factory=lambda:str(datetime.now()))
    createdByUser: Optional[str] = None

@dataclass
class CausalPrimitive:
    label: str

@dataclass
class CausalVariable(CausalPrimitive):
    """ API definition of a causal variable. """
    label: str="CausalVariable"

@dataclass
class CausalRelationship(CausalPrimitive):
    """ API definition of a causal relationship. """
    source: str = "source"
    target: str = "target"
    label:str ="CausalRelationship"

@dataclass
class ICM:
    metadata: ICMMetadata
    primitives: List[CausalPrimitive]

metadatas = {str(x):ICMMetadata(id=str(x)) for x in range(5)}

A = [CausalVariable(label="X"), CausalVariable(label="Y"),
        CausalRelationship(source="X", target="Y")]

@app.route('/icm')
def icm():
    """ List all ICMs """
    return jsonify(list(metadatas.keys()))

@app.route('/icm/<string:uuid>', methods=["GET", "DELETE"])
def icm_uuid(uuid):
    if request.method == "GET":
        return jsonify(asdict(metadatas[uuid]))
    elif request.method == "DELETE":
        if uuid in metadatas:
            del metadatas[uuid]

@app.route('/icm/<string:uuid>/primitive', methods=["GET", "POST"])
def icm_uuid_primitive(uuid):
    if request.method == "POST":
        pass
    else:
        return jsonify([asdict(x) for x in A])


if __name__ == '__main__':
    app.run(debug=True)
