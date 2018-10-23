import json
from uuid import uuid4
from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from flask_sqlalchemy import SQLAlchemy
from delphi.icm_api import db
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
            return json.loads(value)


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
    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]


class DelphiModel(db.Model):
    __tablename__ = "delphimodel"
    id = db.Column(db.String, primary_key=True)
    icm_metadata = db.relationship(
        "ICMMetadata", backref="delphimodel", lazy=True, uselist=False
    )
    model = db.Column(db.PickleType)


class ICMMetadata(db.Model, Serializable):
    """ Placeholder docstring for class ICMMetadata. """

    __tablename__ = "icmmetadata"
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    icmProvider = db.Column(
        db.Enum("BAE", "BBN", "STR", "DUMMY"), nullable=True
    )
    title = db.Column(db.String, nullable=True)
    version = db.Column(db.Integer, nullable=True)
    created = db.Column(db.String, nullable=True)
    createdByUser_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )
    createdByUser = db.relationship("User", foreign_keys=[createdByUser_id])
    lastAccessed = db.Column(db.String, nullable=True)
    lastAccessedByUser_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )
    lastAccessedByUser = db.relationship(
        "User", foreign_keys=[lastAccessedByUser_id]
    )
    lastUpdated = db.Column(db.String, nullable=True)
    lastUpdatedByUser_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )
    lastUpdatedByUser = db.relationship(
        "User", foreign_keys=[lastUpdatedByUser_id]
    )
    estimatedNumberOfPrimitives = db.Column(db.Integer, nullable=True)
    lifecycleState = db.Column(
        db.Enum(
            "PROPOSED",
            "APPROVED",
            "EXPERIMENTAL",
            "OPERATIONAL",
            "SUSPENDED",
            "ARCHIVED",
            "CREATED",
        ),
        nullable=True,
    )
    derivation = db.Column(JsonEncodedList, nullable=True)
    model_id = db.Column(db.String, db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "icmmetadata"}


class User(db.Model, Serializable):
    """ Placeholder docstring for class User. """

    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, nullable=True)
    firstName = db.Column(db.String, nullable=True)
    lastName = db.Column(db.String, nullable=True)
    email = db.Column(db.String, nullable=True)
    password = db.Column(db.String, nullable=True)
    phone = db.Column(db.String, nullable=True)
    userStatus = db.Column(db.Integer, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "user"}


class ServerResponse(db.Model, Serializable):
    """ Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = db.Column(
        db.String, primary_key=True, default=str(uuid4()), nullable=True
    )
    message = db.Column(db.String, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "serverresponse"}


class CausalPrimitive(db.Model, Serializable):
    """ Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    baseType = db.Column(db.String)
    namespaces = db.Column(JsonEncodedDict, nullable=True)
    types = db.Column(JsonEncodedList, nullable=True)
    editable = db.Column(db.Boolean, nullable=True)
    disableable = db.Column(db.Boolean, nullable=True)
    disabled = db.Column(db.Boolean, nullable=True)
    id = db.Column(
        db.String, primary_key=True, default=str(uuid4()), nullable=True
    )
    label = db.Column(db.String, nullable=True)
    description = db.Column(db.String, nullable=True)
    lastUpdated = db.Column(db.String, nullable=True)
    __mapper_args__ = {
        "polymorphic_identity": "causalprimitive",
        "polymorphic_on": baseType,
    }


class Evidence(db.Model, Serializable):
    """ Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = db.Column(
        db.String, primary_key=True, default=str(uuid4()), nullable=True
    )
    link = db.Column(db.String, nullable=True)
    description = db.Column(db.String, nullable=True)
    category = db.Column(db.String, nullable=True)
    rank = db.Column(db.Integer, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "evidence"}


class Experiment(db.Model, Serializable):
    """ structure used for experimentation """

    __tablename__ = "experiment"
    id = db.Column(
        db.String, primary_key=True, default=str(uuid4()), nullable=True
    )
    label = db.Column(db.String, nullable=True)
    options = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "experiment"}


class ExperimentResult(db.Model, Serializable):
    """ Notional model of experiment results """

    __tablename__ = "experimentresult"
    id = db.Column(
        db.String, primary_key=True, default=str(uuid4()), nullable=True
    )
    __mapper_args__ = {"polymorphic_identity": "experimentresult"}
