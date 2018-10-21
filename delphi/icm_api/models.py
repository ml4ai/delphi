from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


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


class User(db.Model):
    """Placeholder docstring for class User. """

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=False)
    firstName = db.Column(db.String(120), unique=False)
    lastName = db.Column(db.String(120), unique=False)
    email = db.Column(db.String(120), unique=False)
    password = db.Column(db.String(120), unique=False)
    phone = db.Column(db.String(120), unique=False)
    userStatus = db.Column(db.Integer, unique=False)


class ICMMetadata(db.Model):
    """Placeholder docstring for class ICMMetadata. """

    id = db.Column(db.String(120), primary_key=True)
    icmProvider = db.Column(db.Text, unique=False)
    title = db.Column(db.String(120), unique=False)
    version = db.Column(db.Integer, unique=False)
    created = db.Column(db.String(120), unique=False)
    createdByUser = db.Column(db.Text, unique=False)
    lastAccessed = db.Column(db.String(120), unique=False)
    lastAccessedByUser = db.Column(db.Text, unique=False)
    lastUpdated = db.Column(db.String(120), unique=False)
    lastUpdatedByUser = db.Column(db.Text, unique=False)
    estimatedNumberOfPrimitives = db.Column(db.Integer, unique=False)
    lifecycleState = db.Column(db.Text, unique=False)
    derivation = db.Column(db.String(120), unique=False)


class ServerResponse(db.Model):
    """Placeholder docstring for class ServerResponse. """

    id = db.Column(db.String(120), primary_key=True)
    message = db.Column(db.String(120), unique=False)


class Range(db.Model):
    """Top level range object used in a CausalVariable """

    id = db.Column(db.Integer, primary_key=True)


class IntegerRange(db.Model):
    """The range for an integer value """

    range_id = db.Column(db.Integer, ForeignKey("range.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class FloatRange(db.Model):
    """The range for a floating point value """

    range_id = db.Column(db.Integer, ForeignKey("range.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class BooleanRange(db.Model):
    """Denotes a boolean range """

    range_id = db.Column(db.Integer, ForeignKey("range.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class EnumRange(db.Model):
    """The values of an enumeration """

    range_id = db.Column(db.Integer, ForeignKey("range.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class DistributionEnumRange(db.Model):
    """The range of classifications that can be reported in a DistributionEnumValue """

    range_id = db.Column(db.Integer, ForeignKey("range.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class Value(db.Model):
    """Top level value object used in a TimeSeriesValue """

    id = db.Column(db.Integer, primary_key=True)


class IntegerValue(db.Model):
    """An integer value """

    value_id = db.Column(db.Integer, ForeignKey("value.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class FloatValue(db.Model):
    """A floating point value """

    value_id = db.Column(db.Integer, ForeignKey("value.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class BooleanValue(db.Model):
    """A boolean value """

    value_id = db.Column(db.Integer, ForeignKey("value.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class EnumValue(db.Model):
    """An enumeration value defined in the EnumRange for the CausalVariable """

    value_id = db.Column(db.Integer, ForeignKey("value.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


class DistributionEnumValue(db.Model):
    """A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    value_id = db.Column(db.Integer, ForeignKey("value.id"), primary_key=True)
    id = db.Column(db.Integer, primary_key=True)


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


class TimeSeriesValue(db.Model):
    """Time series value at a particular time """

    time = db.Column(db.String(120), primary_key=True)
    value_id = db.Column(db.Integer, ForeignKey("value.id"))
    active = db.Column(db.Text, unique=False)
    trend = db.Column(db.Text, unique=False)


class CausalPrimitive(db.Model):
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    namespaces = db.Column(db.Text, unique=False)
    types = db.Column(db.String(120), unique=False)
    editable = db.Column(db.Boolean, default=True, unique=False)
    disableable = db.Column(db.Boolean, default=True, unique=False)
    disabled = db.Column(db.Boolean, default=False, unique=False)
    id = db.Column(db.String(120), primary_key=True)
    label = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    lastUpdated = db.Column(db.String(120), unique=False)


class Entity(db.Model):
    """API definition of an entity.  """

    causalprimitive_id = db.Column(
        db.Integer, ForeignKey("causalprimitive.id"), primary_key=True
    )
    confidence = db.Column(db.Float, primary_key=True)


class CausalVariable(db.Model):
    """API definition of a causal variable.  """

    causalprimitive_id = db.Column(
        db.Integer, ForeignKey("causalprimitive.id"), primary_key=True
    )
    range_id = db.Column(db.Integer, ForeignKey("range.id"))
    units = db.Column(db.String(120), unique=False)
    backingEntities = db.Column(db.String(120), unique=False)
    lastKnownValue = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)


class ConfigurationVariable(db.Model):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    causalprimitive_id = db.Column(
        db.Integer, ForeignKey("causalprimitive.id"), primary_key=True
    )
    units = db.Column(db.String(120), primary_key=True)
    lastKnownValue = db.Column(db.Text, unique=False)
    range_id = db.Column(db.Integer, ForeignKey("range.id"))


class CausalRelationship(db.Model):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """

    causalprimitive_id = db.Column(
        db.Integer, ForeignKey("causalprimitive.id"), primary_key=True
    )
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)
    strength = db.Column(db.Float, unique=False)
    reinforcement = db.Column(db.Boolean, unique=False)


class Relationship(db.Model):
    """API definition of a generic relationship between two primitives """

    causalprimitive_id = db.Column(
        db.Integer, ForeignKey("causalprimitive.id"), primary_key=True
    )
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)


class Evidence(db.Model):
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """

    id = db.Column(db.String(120), primary_key=True)
    link = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    category = db.Column(db.String(120), unique=False)
    rank = db.Column(db.Integer, unique=False)


class Projection(db.Model):
    """Placeholder docstring for class Projection. """

    numSteps = db.Column(db.Integer, primary_key=True)
    stepSize = db.Column(db.String(120), unique=False)
    startTime = db.Column(db.String(120), unique=False)


class Experiment(db.Model):
    """structure used for experimentation """

    id = db.Column(db.String(120), primary_key=True)
    label = db.Column(db.String(120), unique=False)
    options = db.Column(db.Text, unique=False)


class ForwardProjection(db.Model):
    """a foward projection experiment """

    experiment_id = db.Column(
        db.Integer, ForeignKey("experiment.id"), primary_key=True
    )
    interventions = db.Column(db.Text, primary_key=True)
    projection = db.Column(db.Text, unique=False)


class SensitivityAnalysis(db.Model):
    """a sensitivity analysis experiment """

    experiment_id = db.Column(
        db.Integer, ForeignKey("experiment.id"), primary_key=True
    )
    variables = db.Column(db.String(120), primary_key=True)


class ExperimentResult(db.Model):
    """Notional model of experiment results """

    id = db.Column(db.String(120), primary_key=True)


class ForwardProjectionResult(db.Model):
    """The result of a forward projection experiment """

    experimentresult_id = db.Column(
        db.Integer, ForeignKey("experimentresult.id"), primary_key=True
    )
    projection = db.Column(db.Text, primary_key=True)
    results = db.Column(db.Text, unique=False)


class SensitivityAnalysisResult(db.Model):
    """The result of a sensitivity analysis experiment """

    experimentresult_id = db.Column(
        db.Integer, ForeignKey("experimentresult.id"), primary_key=True
    )
    results = db.Column(db.Text, primary_key=True)


class Traversal(db.Model):
    """Placeholder docstring for class Traversal. """

    maxDepth = db.Column(db.Integer, primary_key=True)


class Version(db.Model):
    """Placeholder docstring for class Version. """

    icmVersion = db.Column(db.String(120), primary_key=True)
    icmProviderVersion = db.Column(db.String(120), unique=False)
