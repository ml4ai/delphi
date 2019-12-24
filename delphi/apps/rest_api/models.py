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
    __mapper_args__ = {"polymorphic_identity": "ICMMetadata"}


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
    __mapper_args__ = {"polymorphic_identity": "User"}


class ServerResponse(db.Model, Serializable):
    """ Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    message = db.Column(db.String, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ServerResponse"}


class CausalPrimitive(db.Model, Serializable):
    """ Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    baseType = db.Column(db.String)
    namespaces = db.Column(JsonEncodedDict, nullable=True)
    types = db.Column(JsonEncodedList, nullable=True)
    editable = db.Column(db.Boolean, nullable=True, default=True)
    disableable = db.Column(db.Boolean, nullable=True, default=True)
    disabled = db.Column(db.Boolean, nullable=True, default=False)
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    label = db.Column(db.String, nullable=True)
    description = db.Column(db.String, nullable=True)
    lastUpdated = db.Column(db.String, nullable=True)
    auxiliaryProperties = db.Column(JsonEncodedList, nullable=True)
    model_id = db.Column(db.String, db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {
        "polymorphic_identity": "CausalPrimitive",
        "polymorphic_on": baseType,
    }


class Entity(CausalPrimitive):
    """ Placeholder docstring for class Entity. """

    __tablename__ = "entity"
    id = db.Column(
        db.String,
        db.ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    confidence = db.Column(db.Float, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "Entity"}


class CausalVariable(CausalPrimitive):
    """ Placeholder docstring for class CausalVariable. """

    __tablename__ = "causalvariable"
    id = db.Column(
        db.String,
        db.ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    units = db.Column(db.String, nullable=True)
    backingEntities = db.Column(JsonEncodedList, nullable=True)
    lastKnownValue = db.Column(JsonEncodedDict, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    range = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CausalVariable"}


class ConfigurationVariable(CausalPrimitive):
    """ Placeholder docstring for class ConfigurationVariable. """

    __tablename__ = "configurationvariable"
    id = db.Column(
        db.String,
        db.ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    units = db.Column(db.String, nullable=True)
    lastKnownValue = db.Column(JsonEncodedDict, nullable=True)
    range = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ConfigurationVariable"}


class CausalRelationship(CausalPrimitive):
    """ Placeholder docstring for class CausalRelationship. """

    __tablename__ = "causalrelationship"
    id = db.Column(
        db.String,
        db.ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    source = db.Column(JsonEncodedDict, nullable=True)
    target = db.Column(JsonEncodedDict, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    strength = db.Column(db.Float, nullable=True)
    reinforcement = db.Column(db.Boolean, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CausalRelationship"}


class Relationship(CausalPrimitive):
    """ Placeholder docstring for class Relationship. """

    __tablename__ = "relationship"
    id = db.Column(
        db.String,
        db.ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    source = db.Column(JsonEncodedDict, nullable=True)
    target = db.Column(JsonEncodedDict, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "Relationship"}


class Evidence(db.Model, Serializable):
    """ Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    link = db.Column(db.String, nullable=True)
    description = db.Column(db.String, nullable=True)
    category = db.Column(db.String, nullable=True)
    rank = db.Column(db.Integer, nullable=True)
    causalrelationship_id = db.Column(
        db.String, db.ForeignKey("causalrelationship.id")
    )
    __mapper_args__ = {"polymorphic_identity": "Evidence"}


class Experiment(db.Model, Serializable):
    """ structure used for experimentation """

    __tablename__ = "experiment"
    baseType = db.Column(db.String)
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    label = db.Column(db.String, nullable=True)
    options = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {
        "polymorphic_identity": "Experiment",
        "polymorphic_on": baseType,
    }


class ForwardProjection(Experiment):
    """ Placeholder docstring for class ForwardProjection. """

    __tablename__ = "forwardprojection"
    id = db.Column(
        db.String,
        db.ForeignKey("experiment.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    interventions = db.Column(JsonEncodedList, nullable=True)
    projection = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ForwardProjection"}


class SensitivityAnalysis(Experiment):
    """ Placeholder docstring for class SensitivityAnalysis. """

    __tablename__ = "sensitivityanalysis"
    id = db.Column(
        db.String,
        db.ForeignKey("experiment.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    nodeOfInterest = db.Column(db.String, nullable=True)
    numSteps = db.Column(db.Integer, nullable=True)
    sensitivityVariables = db.Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "SensitivityAnalysis"}


class ExperimentResult(db.Model, Serializable):
    """ Notional model of experiment results """

    __tablename__ = "experimentresult"
    baseType = db.Column(db.String)
    id = db.Column(db.String, primary_key=True, default=str(uuid4()))
    __mapper_args__ = {
        "polymorphic_identity": "ExperimentResult",
        "polymorphic_on": baseType,
    }


class CauseMosForwardProjectionResult(ExperimentResult):
    """ Placeholder docstring for class CauseMosForwardProjectionResult. """

    __tablename__ = "causemosforwardprojectionresult"
    id = db.Column(
        db.String,
        db.ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    results = db.Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CauseMosForwardProjectionResult"}


class ForwardProjectionResult(ExperimentResult):
    """ Placeholder docstring for class ForwardProjectionResult. """

    __tablename__ = "forwardprojectionresult"
    id = db.Column(
        db.String,
        db.ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    projection = db.Column(JsonEncodedDict, nullable=True)
    results = db.Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ForwardProjectionResult"}


class SensitivityAnalysisResult(ExperimentResult):
    """ Placeholder docstring for class SensitivityAnalysisResult. """

    __tablename__ = "sensitivityanalysisresult"
    id = db.Column(
        db.String,
        db.ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    results = db.Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "SensitivityAnalysisResult"}
