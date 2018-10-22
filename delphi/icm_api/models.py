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
    id = db.Column(db.String, primary_key=True)
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
    id = db.Column(db.Integer, unique=False)
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
    id = db.Column(db.String, unique=False)
    icmProvider = db.relationship("ICMProvider", backref="icmmetadata")
    title = db.Column(db.String, unique=False)
    version = db.Column(db.Integer, unique=False)
    created = db.Column(db.String, unique=False)
    createdByUser = db.relationship("User", backref="icmmetadata")
    lastAccessed = db.Column(db.String, unique=False)
    lastAccessedByUser = db.relationship("User", backref="icmmetadata")
    lastUpdated = db.Column(db.String, unique=False)
    lastUpdatedByUser = db.relationship("User", backref="icmmetadata")
    estimatedNumberOfPrimitives = db.Column(db.Integer, unique=False)
    lifecycleState = db.relationship("LifecycleState", backref="icmmetadata")
    derivation = db.Column("", unique=False)
    model_id = db.Column(db.String, db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "icmmetadata"}


class ServerResponse(db.Model, Serializable):
    """ Placeholder docstring for class ServerResponse. """

    __tablename__ = "serverresponse"
    id = db.Column(db.String, unique=False)
    message = db.Column(db.String, unique=False)
    __mapper_args__ = {"polymorphic_identity": "serverresponse"}


class Range(object):
    """ Top level range object used in a CausalVariable """

    baseType = "Range"


@dataclass
class IntegerRange(Range):
    """ The range for an integer value """

    baseType = "IntegerRange"


@dataclass
class FloatRange(Range):
    """ The range for a floating point value """

    baseType = "FloatRange"


@dataclass
class BooleanRange(Range):
    """ Denotes a boolean range """

    baseType = "BooleanRange"


@dataclass
class EnumRange(Range):
    """ The values of an enumeration """

    baseType = "EnumRange"


@dataclass
class DistributionEnumRange(Range):
    """ The range of classifications that can be reported in a DistributionEnumValue """

    baseType = "DistributionEnumRange"


class Value(object):
    """ Top level value object used in a TimeSeriesValue """

    baseType = "Value"


@dataclass
class IntegerValue(Value):
    """ An integer value """

    baseType = "IntegerValue"


@dataclass
class FloatValue(Value):
    """ A floating point value """

    baseType = "FloatValue"


@dataclass
class BooleanValue(Value):
    """ A boolean value """

    baseType = "BooleanValue"


@dataclass
class EnumValue(Value):
    """ An enumeration value defined in the EnumRange for the CausalVariable """

    baseType = "EnumValue"


@dataclass
class DistributionEnumValue(Value):
    """ A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    baseType = "DistributionEnumValue"


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

    baseType = "TimeSeriesValue"
    baseType = "TimeSeriesValue"
    baseType = "TimeSeriesValue"
    baseType = "TimeSeriesValue"


class CausalPrimitive(db.Model, Serializable):
    """ Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "causalprimitive"
    baseType = db.Column(db.String, unique=False)
    namespaces = db.Column("", unique=False)
    types = db.Column("", unique=False)
    editable = db.Column(db.Boolean, default=True, unique=False)
    disableable = db.Column(db.Boolean, default=True, unique=False)
    disabled = db.Column(db.Boolean, default=False, unique=False)
    id = db.Column(db.String, unique=False)
    label = db.Column(db.String, unique=False)
    description = db.Column(db.String, unique=False)
    lastUpdated = db.Column(db.String, unique=False)
    __mapper_args__ = {
        "polymorphic_identity": "causalprimitive",
        "polymorphic_on": baseType,
    }


@dataclass
class Entity(CausalPrimitive):
    """ API definition of an entity.  """

    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "entity"}


@dataclass
class CausalVariable(CausalPrimitive):
    """ API definition of a causal variable.  """

    range = db.relationship("Range", backref="causalvariable")
    units = db.Column(db.String, unique=False)
    backingEntities = db.Column("", unique=False)
    lastKnownValue = db.relationship(
        "TimeSeriesValue", backref="causalvariable"
    )
    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    model_id = db.Column(db.String, db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "causalvariable"}


@dataclass
class ConfigurationVariable(CausalPrimitive):
    """ Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    units = db.Column(db.String, unique=False)
    lastKnownValue = db.relationship(
        "TimeSeriesValue", backref="configurationvariable"
    )
    range = db.relationship("Range", backref="configurationvariable")
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "configurationvariable"}


@dataclass
class CausalRelationship(CausalPrimitive):
    """ API defintion of a causal relationship. Indicates causality between two causal variables. """

    source = db.Column("", unique=False)
    target = db.Column("", unique=False)
    confidence = db.Column(db.Float, unique=False)
    strength = db.Column(db.Float, unique=False)
    reinforcement = db.Column(db.Boolean, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    model_id = db.Column(db.String, db.ForeignKey("delphimodel.id"))
    __mapper_args__ = {"polymorphic_identity": "causalrelationship"}


@dataclass
class Relationship(CausalPrimitive):
    """ API definition of a generic relationship between two primitives """

    source = db.Column("", unique=False)
    target = db.Column("", unique=False)
    confidence = db.Column(db.Float, unique=False)
    id = db.Column(
        db.String, ForeignKey("causalprimitive.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "relationship"}


class Evidence(db.Model, Serializable):
    """ Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "evidence"
    id = db.Column(db.String, unique=False)
    link = db.Column(db.String, unique=False)
    description = db.Column(db.String, unique=False)
    category = db.Column(db.String, unique=False)
    rank = db.Column(db.Integer, unique=False)
    __mapper_args__ = {"polymorphic_identity": "evidence"}


class Projection(object):
    """ Placeholder docstring for class Projection. """

    baseType = "Projection"
    baseType = "Projection"
    baseType = "Projection"


class Experiment(db.Model, Serializable):
    """ structure used for experimentation """

    __tablename__ = "experiment"
    id = db.Column(db.String, unique=False)
    label = db.Column(db.String, unique=False)
    options = db.Column("", unique=False)
    __mapper_args__ = {"polymorphic_identity": "experiment"}


@dataclass
class ForwardProjection(Experiment):
    """ a foward projection experiment """

    interventions = db.Column("", unique=False)
    projection = db.relationship("Projection", backref="forwardprojection")
    id = db.Column(db.String, ForeignKey("experiment.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "forwardprojection"}


@dataclass
class SensitivityAnalysis(Experiment):
    """ a sensitivity analysis experiment """

    variables = db.Column("", unique=False)
    id = db.Column(db.String, ForeignKey("experiment.id"), primary_key=True)
    __mapper_args__ = {"polymorphic_identity": "sensitivityanalysis"}


class ExperimentResult(db.Model, Serializable):
    """ Notional model of experiment results """

    __tablename__ = "experimentresult"
    id = db.Column(db.String, unique=False)
    __mapper_args__ = {"polymorphic_identity": "experimentresult"}


@dataclass
class ForwardProjectionResult(ExperimentResult):
    """ The result of a forward projection experiment """

    projection = db.relationship(
        "Projection", backref="forwardprojectionresult"
    )
    results = db.Column("", unique=False)
    id = db.Column(
        db.String, ForeignKey("experimentresult.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "forwardprojectionresult"}


@dataclass
class SensitivityAnalysisResult(ExperimentResult):
    """ The result of a sensitivity analysis experiment """

    results = db.Column("", unique=False)
    id = db.Column(
        db.String, ForeignKey("experimentresult.id"), primary_key=True
    )
    __mapper_args__ = {"polymorphic_identity": "sensitivityanalysisresult"}


class Traversal(object):
    """ Placeholder docstring for class Traversal. """

    baseType = "Traversal"


class Version(object):
    """ Placeholder docstring for class Version. """

    baseType = "Version"
    baseType = "Version"
