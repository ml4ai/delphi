import json
from uuid import uuid4
from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from delphi.db import Base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    TypeDecorator,
    PickleType,
    Enum,
    Boolean,
    Float,
)
from sqlalchemy.inspection import inspect
from sqlalchemy.ext import mutable
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import operators


Base = declarative_base()


class JsonEncodedList(TypeDecorator):
    """Enables list storage by encoding and decoding on the fly."""

    impl = Text

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


class JsonEncodedDict(TypeDecorator):
    """Enables JsonEncodedDict storage by encoding and decoding on the fly."""

    impl = Text

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


class DelphiModel(Base, Serializable):
    __tablename__ = "delphimodel"
    id = Column(String, primary_key=True)
    icm_metadata = relationship(
        "ICMMetadata", backref="delphimodel", lazy=True, uselist=False
    )
    model = Column(PickleType)


class ICMMetadata(Base, Serializable):
    """ Placeholder docstring for class ICMMetadata. """

    __tablename__ = "icmmetadata"
    id = Column(String, primary_key=True, default=str(uuid4()))
    icmProvider = Column(Enum("BAE", "BBN", "STR", "DUMMY"), nullable=True)
    title = Column(String, nullable=True)
    version = Column(Integer, nullable=True)
    created = Column(String, nullable=True)
    createdByUser_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    createdByUser = relationship("User", foreign_keys=[createdByUser_id])
    lastAccessed = Column(String, nullable=True)
    lastAccessedByUser_id = Column(
        Integer, ForeignKey("user.id"), nullable=True
    )
    lastAccessedByUser = relationship(
        "User", foreign_keys=[lastAccessedByUser_id]
    )
    lastUpdated = Column(String, nullable=True)
    lastUpdatedByUser_id = Column(
        Integer, ForeignKey("user.id"), nullable=True
    )
    lastUpdatedByUser = relationship(
        "User", foreign_keys=[lastUpdatedByUser_id]
    )
    estimatedNumberOfPrimitives = Column(Integer, nullable=True)
    lifecycleState = Column(
        Enum(
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
    derivation = Column(JsonEncodedList, nullable=True)
    model_id = Column(String, ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "ICMMetadata"}


class User(Base, Serializable):
    """ Placeholder docstring for class User. """

    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=True)
    firstName = Column(String, nullable=True)
    lastName = Column(String, nullable=True)
    email = Column(String, nullable=True)
    password = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    userStatus = Column(Integer, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "User"}


class ServerResponse(Base, Serializable):
    """ Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = Column(String, primary_key=True, default=str(uuid4()))
    message = Column(String, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ServerResponse"}


class CausalPrimitive(Base, Serializable):
    """ Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    baseType = Column(String)
    namespaces = Column(JsonEncodedDict, nullable=True)
    types = Column(JsonEncodedList, nullable=True)
    editable = Column(Boolean, nullable=True, default=True)
    disableable = Column(Boolean, nullable=True, default=True)
    disabled = Column(Boolean, nullable=True, default=False)
    id = Column(String, primary_key=True, default=str(uuid4()))
    label = Column(String, nullable=True)
    description = Column(String, nullable=True)
    lastUpdated = Column(String, nullable=True)
    auxiliaryProperties = Column(JsonEncodedList, nullable=True)
    model_id = Column(String, ForeignKey("delphimodel.id"))
    __mapper_args__ = {
        "polymorphic_identity": "CausalPrimitive",
        "polymorphic_on": baseType,
    }


class Entity(CausalPrimitive):
    """ Placeholder docstring for class Entity. """

    __tablename__ = "entity"
    id = Column(
        String,
        ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    confidence = Column(Float, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "Entity"}


class CausalVariable(CausalPrimitive):
    """ Placeholder docstring for class CausalVariable. """

    __tablename__ = "causalvariable"
    id = Column(
        String,
        ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    units = Column(String, nullable=True)
    backingEntities = Column(JsonEncodedList, nullable=True)
    lastKnownValue = Column(JsonEncodedDict, nullable=True)
    confidence = Column(Float, nullable=True)
    range = Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CausalVariable"}


class ConfigurationVariable(CausalPrimitive):
    """ Placeholder docstring for class ConfigurationVariable. """

    __tablename__ = "configurationvariable"
    id = Column(
        String,
        ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    units = Column(String, nullable=True)
    lastKnownValue = Column(JsonEncodedDict, nullable=True)
    range = Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ConfigurationVariable"}


class CausalRelationship(CausalPrimitive):
    """ Placeholder docstring for class CausalRelationship. """

    __tablename__ = "causalrelationship"
    id = Column(
        String,
        ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    source = Column(JsonEncodedDict, nullable=True)
    target = Column(JsonEncodedDict, nullable=True)
    confidence = Column(Float, nullable=True)
    strength = Column(Float, nullable=True)
    reinforcement = Column(Boolean, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "CausalRelationship"}


class Relationship(CausalPrimitive):
    """ Placeholder docstring for class Relationship. """

    __tablename__ = "relationship"
    id = Column(
        String,
        ForeignKey("causalprimitive.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    source = Column(JsonEncodedDict, nullable=True)
    target = Column(JsonEncodedDict, nullable=True)
    confidence = Column(Float, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "Relationship"}


class Evidence(Base, Serializable):
    """ Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = Column(String, primary_key=True, default=str(uuid4()))
    link = Column(String, nullable=True)
    description = Column(String, nullable=True)
    category = Column(String, nullable=True)
    rank = Column(Integer, nullable=True)
    causalrelationship_id = Column(String, ForeignKey("causalrelationship.id"))
    __mapper_args__ = {"polymorphic_identity": "Evidence"}


class Experiment(Base, Serializable):
    """ structure used for experimentation """

    __tablename__ = "experiment"
    baseType = Column(String)
    id = Column(String, primary_key=True, default=str(uuid4()))
    label = Column(String, nullable=True)
    options = Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {
        "polymorphic_identity": "Experiment",
        "polymorphic_on": baseType,
    }


class ForwardProjection(Experiment):
    """ Placeholder docstring for class ForwardProjection. """

    __tablename__ = "forwardprojection"
    id = Column(
        String,
        ForeignKey("experiment.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    interventions = Column(JsonEncodedList, nullable=True)
    projection = Column(JsonEncodedDict, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ForwardProjection"}


class SensitivityAnalysis(Experiment):
    """ Placeholder docstring for class SensitivityAnalysis. """

    __tablename__ = "sensitivityanalysis"
    id = Column(
        String,
        ForeignKey("experiment.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    nodeOfInterest = Column(String, nullable=True)
    numSteps = Column(Integer, nullable=True)
    sensitivityVariables = Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "SensitivityAnalysis"}


class ExperimentResult(Base, Serializable):
    """ Notional model of experiment results """

    __tablename__ = "experimentresult"
    baseType = Column(String)
    id = Column(String, primary_key=True, default=str(uuid4()))
    __mapper_args__ = {
        "polymorphic_identity": "ExperimentResult",
        "polymorphic_on": baseType,
    }


class ForwardProjectionResult(ExperimentResult):
    """ Placeholder docstring for class ForwardProjectionResult. """

    __tablename__ = "forwardprojectionresult"
    id = Column(
        String,
        ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    projection = Column(JsonEncodedDict, nullable=True)
    results = Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "ForwardProjectionResult"}


class SensitivityAnalysisResult(ExperimentResult):
    """ Placeholder docstring for class SensitivityAnalysisResult. """

    __tablename__ = "sensitivityanalysisresult"
    id = Column(
        String,
        ForeignKey("experimentresult.id"),
        primary_key=True,
        default=str(uuid4()),
    )
    results = Column(JsonEncodedList, nullable=True)
    __mapper_args__ = {"polymorphic_identity": "SensitivityAnalysisResult"}
