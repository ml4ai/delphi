from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from sqlalchemy import Table, Column, Integer, String, ForeignKey, PickleType
from flask_sqlalchemy import SQLAlchemy
from delphi.icm_api import db
from sqlalchemy.inspection import inspect


class Serializable(object):
    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]


class DelphiModel(db.Model):
    __tablename__ = "delphimodel"
    id = db.Column(db.String(120), primary_key=True)
    icm_metadata = db.relationship(
        "ICMMetadata", backref="delphimodel", lazy=True, uselist=False
    )
    model = db.Column(db.PickleType)


@unique
class ICMProvider(Enum):
    BAE = "BAE"
    BBN = "BBN"
    STR = "STR"
    DUMMY = "DUMMY"


@unique
class LifecycleState(Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    EXPERIMENTAL = "EXPERIMENTAL"
    OPERATIONAL = "OPERATIONAL"
    SUSPENDED = "SUSPENDED"
    ARCHIVED = "ARCHIVED"
    CREATED = "CREATED"


class User(db.Model, Serializable):
    """ Placeholder docstring for class User. """

    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=False)
    firstName = db.Column(db.String, unique=False)
    lastName = db.Column(db.String, unique=False)
    email = db.Column(db.String, unique=False)
    password = db.Column(db.String, unique=False)
    phone = db.Column(db.String, unique=False)
    userStatus = db.Column(db.Integer, unique=False)
    __mapper_args__ = {"polymorphic_identity": "user"}


class ICMMetadata(db.Model, Serializable):
    """ Placeholder docstring for class ICMMetadata. """

    __tablename__ = "icmmetadata"
    id = db.Column(db.String, primary_key=True)
    icmProvider = db.Column(db.String, unique=False)
    title = db.Column(db.String, unique=False)
    version = db.Column(db.Integer, unique=False)
    created = db.Column(db.String, unique=False)
    createdByUser = db.Column(db.String, unique=False)
    lastAccessed = db.Column(db.String, unique=False)
    lastAccessedByUser = db.Column(db.String, unique=False)
    lastUpdated = db.Column(db.String, unique=False)
    lastUpdatedByUser = db.Column(db.String, unique=False)
    estimatedNumberOfPrimitives = db.Column(db.Integer, unique=False)
    lifecycleState = db.Column(db.String, unique=False)
    derivation = db.Column(db.String, unique=False)
    model_id = db.Column(db.String(120), db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "icmmetadata"}


class ServerResponse(db.Model, Serializable):
    """ Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = db.Column(db.String, primary_key=True)
    message = db.Column(db.String, unique=False)
    __mapper_args__ = {"polymorphic_identity": "serverresponse"}


class Range(object):
    """ Top level range object used in a CausalVariable """

    baseType = db.Column(db.String, unique=False)


class IntegerRange(Range):
    """ The range for an integer value """

    range = db.Column(db.PickleType, unique=False)


class FloatRange(Range):
    """ The range for a floating point value """

    range = db.Column(db.PickleType, unique=False)


class BooleanRange(Range):
    """ Denotes a boolean range """

    range = db.Column(db.PickleType, unique=False)


class EnumRange(Range):
    """ The values of an enumeration """

    range = db.Column(db.String, unique=False)


class DistributionEnumRange(Range):
    """ The range of classifications that can be reported in a DistributionEnumValue """

    range = db.Column(db.String, unique=False)


class Value(object):
    """ Top level value object used in a TimeSeriesValue """

    baseType = db.Column(db.String, unique=False)


class IntegerValue(Value):
    """ An integer value """

    value = db.Column(db.Integer, unique=False)


class FloatValue(Value):
    """ A floating point value """

    value = db.Column(db.Float, unique=False)


class BooleanValue(Value):
    """ A boolean value """

    value = db.Column(db.Boolean, unique=False)


class EnumValue(Value):
    """ An enumeration value defined in the EnumRange for the CausalVariable """

    value = db.Column(db.String, unique=False)


class DistributionEnumValue(Value):
    """ A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    value = db.Column(db.PickleType, unique=False)


@unique
class TimeSeriesState(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"


@unique
class TimeSeriesTrend(Enum):
    LARGE_INCREASE = "LARGE_INCREASE"
    SMALL_INCREASE = "SMALL_INCREASE"
    NO_CHANGE = "NO_CHANGE"
    SMALL_DECREASE = "SMALL_DECREASE"
    LARGE_DECREASE = "LARGE_DECREASE"


class TimeSeriesValue(object):
    """ Time series value at a particular time """

    time = db.Column(db.String, unique=False)
    value = db.Column(db.String, unique=False)
    active = db.Column(db.String, unique=False)
    trend = db.Column(db.String, unique=False)


class CausalPrimitive(db.Model, Serializable):
    """ Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    baseType = db.Column(db.String, unique=False)
    namespaces = db.Column(db.PickleType, unique=False)
    types = db.Column(db.String, unique=False)
    editable = db.Column(db.Boolean, default=True, unique=False)
    disableable = db.Column(db.Boolean, default=True, unique=False)
    disabled = db.Column(db.Boolean, default=False, unique=False)
    id = db.Column(db.String, primary_key=True)
    label = db.Column(db.String, unique=False)
    description = db.Column(db.String, unique=False)
    lastUpdated = db.Column(db.String, unique=False)
    __mapper_args__ = {
        "polymorphic_identity": "causalprimitive",
        "polymorphic_on": baseType,
    }


class Entity(CausalPrimitive):
    """ API definition of an entity.  """

    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "entity"}


class CausalVariable(CausalPrimitive):
    """ API definition of a causal variable.  """

    range = db.Column(db.String, unique=False)
    units = db.Column(db.String, unique=False)
    backingEntities = db.Column(db.String, unique=False)
    lastKnownValue = db.Column(db.String, unique=False)
    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    model_id = db.Column(db.String(120), db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "causalvariable"}


class ConfigurationVariable(CausalPrimitive):
    """ Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    units = db.Column(db.String, unique=False)
    lastKnownValue = db.Column(db.String, unique=False)
    range = db.Column(db.String, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "configurationvariable"}


class CausalRelationship(CausalPrimitive):
    """ API defintion of a causal relationship. Indicates causality between two causal variables. """

    source = db.Column(db.PickleType, unique=False)
    target = db.Column(db.PickleType, unique=False)
    confidence = db.Column(db.Float, unique=False)
    strength = db.Column(db.Float, unique=False)
    reinforcement = db.Column(db.Boolean, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    model_id = db.Column(db.String(120), db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "causalrelationship"}


class Relationship(CausalPrimitive):
    """ API definition of a generic relationship between two primitives """

    source = db.Column(db.PickleType, unique=False)
    target = db.Column(db.PickleType, unique=False)
    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "relationship"}


class Evidence(db.Model, Serializable):
    """ Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = db.Column(db.String, primary_key=True)
    link = db.Column(db.String, unique=False)
    description = db.Column(db.String, unique=False)
    category = db.Column(db.String, unique=False)
    rank = db.Column(db.Integer, unique=False)
    __mapper_args__ = {"polymorphic_identity": "evidence"}


class Projection(object):
    """ Placeholder docstring for class Projection. """

    numSteps = db.Column(db.Integer, unique=False)
    stepSize = db.Column(db.String, unique=False)
    startTime = db.Column(db.String, unique=False)


class Experiment(db.Model, Serializable):
    """ structure used for experimentation """

    __tablename__ = "experiment"
    id = db.Column(db.String, primary_key=True)
    label = db.Column(db.String, unique=False)
    options = db.Column(db.PickleType, unique=False)
    __mapper_args__ = {"polymorphic_identity": "experiment"}


class ForwardProjection(Experiment):
    """ a foward projection experiment """

    interventions = db.Column(db.PickleType, unique=False)
    projection = db.Column(db.String, unique=False)
    id = db.Column(db.String, ForeignKey("experiment.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "forwardprojection"}


class SensitivityAnalysis(Experiment):
    """ a sensitivity analysis experiment """

    variables = db.Column(db.String, unique=False)
    id = db.Column(db.String, ForeignKey("experiment.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "sensitivityanalysis"}


class ExperimentResult(db.Model, Serializable):
    """ Notional model of experiment results """

    __tablename__ = "experimentresult"
    id = db.Column(db.String, primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "experimentresult"}


class ForwardProjectionResult(ExperimentResult):
    """ The result of a forward projection experiment """

    projection = db.Column(db.String, unique=False)
    results = db.Column(db.PickleType, unique=False)
    id = db.Column(
        db.String, ForeignKey("experimentresult.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "forwardprojectionresult"}


class SensitivityAnalysisResult(ExperimentResult):
    """ The result of a sensitivity analysis experiment """

    results = db.Column(db.PickleType, unique=False)
    id = db.Column(
        db.String, ForeignKey("experimentresult.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "sensitivityanalysisresult"}


class Traversal(object):
    """ Placeholder docstring for class Traversal. """

    maxDepth = db.Column(db.Integer, unique=False)


class Version(object):
    """ Placeholder docstring for class Version. """

    icmVersion = db.Column(db.String, unique=False)
    icmProviderVersion = db.Column(db.String, unique=False)
