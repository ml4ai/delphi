import json
from uuid import uuid4
from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from flask_sqlalchemy import SQLAlchemy
from delphi.apps.rest_api import db
from sqlalchemy import PickleType
from sqlalchemy.inspection import inspect
from sqlalchemy.ext import mutable
from sqlalchemy.sql import operators
from sqlalchemy.types import TypeDecorator


class JsonEncodedList(db.TypeDecorator):
    """Enables list storage by encoding and decoding on the fly."""

    impl = db.Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return "[]"
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        else:
            return json.loads(value.replace("'", '"'))


mutable.MutableList.associate_with(JsonEncodedList)


class JsonEncodedDict(db.TypeDecorator):
    """Enables JsonEncodedDict storage by encoding and decoding on the fly."""

    impl = db.Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return "{}"
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        else:
            return json.loads(value)


mutable.MutableDict.associate_with(JsonEncodedDict)


class Serializable(object):
    def deserialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def deserialize_list(l):
        return [m.serialize() for m in l]

class DelphiModel(db.Model, Serializable):
    """ Delphi AnalysisGraph Model """
    __tablename__ = "delphimodel"
    id = db.Column(db.String, primary_key=True)
    icm_metadata = db.relationship(
        "ICMMetadata", backref="delphimodel", lazy=True, uselist=False
    )
    model = db.Column(db.String)


class CauseMosAsyncExperimentResult(ExperimentResult):
    """ Placeholder docstring for class CauseMosAsyncExperimentResult. """

    __tablename__ = "causemosasyncexperimentresult"
    id = db.Column(
        db.String,
        db.ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    status = db.Column(db.String, nullable=True)
    experimentType = db.Column(db.String, nullable=True)
    results = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CauseMosAsyncExperimentResult"}
