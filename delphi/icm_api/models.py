from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()



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


@dataclass
class User(db.Model):
    """Placeholder docstring for class User. """

    __tablename__ = "User"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=False)
    firstName = db.Column(db.String(120), unique=False)
    lastName = db.Column(db.String(120), unique=False)
    email = db.Column(db.String(120), unique=False)
    password = db.Column(db.String(120), unique=False)
    phone = db.Column(db.String(120), unique=False)
    userStatus = db.Column(db.Integer, unique=False)


@dataclass
class ICMMetadata(db.Model):
    """Placeholder docstring for class ICMMetadata. """

    __tablename__ = "ICMMetadata"
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


@dataclass
class ServerResponse(db.Model):
    """Placeholder docstring for class ServerResponse. """

    __tablename__ = "ServerResponse"
    id = db.Column(db.String(120), primary_key=True)
    message = db.Column(db.String(120), unique=False)


@dataclass
class Range(db.Model):
    """Top level range object used in a CausalVariable """

    __tablename__ = "Range"


@dataclass
class IntegerRange(Range):
    """The range for an integer value """

    __tablename__ = "IntegerRange"
    range = db.Column(db.Text, primary_key=True)


@dataclass
class FloatRange(Range):
    """The range for a floating point value """

    __tablename__ = "FloatRange"
    range = db.Column(db.Text, primary_key=True)


@dataclass
class BooleanRange(Range):
    """Denotes a boolean range """

    __tablename__ = "BooleanRange"
    range = db.Column(db.Text, primary_key=True)


@dataclass
class EnumRange(Range):
    """The values of an enumeration """

    __tablename__ = "EnumRange"
    range = db.Column(db.String(120), primary_key=True)


@dataclass
class DistributionEnumRange(Range):
    """The range of classifications that can be reported in a DistributionEnumValue """

    __tablename__ = "DistributionEnumRange"
    range = db.Column(db.String(120), primary_key=True)


@dataclass
class Value(db.Model):
    """Top level value object used in a TimeSeriesValue """

    __tablename__ = "Value"


@dataclass
class IntegerValue(Value):
    """An integer value """

    __tablename__ = "IntegerValue"
    value = db.Column(db.Integer, primary_key=True)


@dataclass
class FloatValue(Value):
    """A floating point value """

    __tablename__ = "FloatValue"
    value = db.Column(db.Float, primary_key=True)


@dataclass
class BooleanValue(Value):
    """A boolean value """

    __tablename__ = "BooleanValue"
    value = db.Column(db.Boolean, primary_key=True)


@dataclass
class EnumValue(Value):
    """An enumeration value defined in the EnumRange for the CausalVariable """

    __tablename__ = "EnumValue"
    value = db.Column(db.String(120), primary_key=True)


@dataclass
class DistributionEnumValue(Value):
    """A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    __tablename__ = "DistributionEnumValue"
    value = db.Column(db.Text, primary_key=True)


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


@dataclass
class TimeSeriesValue(db.Model):
    """Time series value at a particular time """

    __tablename__ = "TimeSeriesValue"
    time = db.Column(db.String(120), primary_key=True)
    value = db.Column(db.Text, unique=False)
    active = db.Column(db.Text, unique=False)
    trend = db.Column(db.Text, unique=False)


@dataclass
class CausalPrimitive(db.Model):
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "CausalPrimitive"
    namespaces = db.Column(db.Text, unique=False)
    types = db.Column(db.String(120), unique=False)
    editable = db.Column(db.Boolean, default=True, unique=False)
    disableable = db.Column(db.Boolean, default=True, unique=False)
    disabled = db.Column(db.Boolean, default=False, unique=False)
    id = db.Column(db.String(120), unique=False)
    label = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    lastUpdated = db.Column(db.String(120), unique=False)


@dataclass
class Entity(CausalPrimitive):
    """API definition of an entity.  """

    __tablename__ = "Entity"
    confidence = db.Column(db.Float, primary_key=True)


@dataclass
class CausalVariable(CausalPrimitive):
    """API definition of a causal variable.  """

    __tablename__ = "CausalVariable"
    range = db.Column(db.Text, primary_key=True)
    units = db.Column(db.String(120), unique=False)
    backingEntities = db.Column(db.String(120), unique=False)
    lastKnownValue = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)


@dataclass
class ConfigurationVariable(CausalPrimitive):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    __tablename__ = "ConfigurationVariable"
    units = db.Column(db.String(120), primary_key=True)
    lastKnownValue = db.Column(db.Text, unique=False)
    range = db.Column(db.Text, unique=False)


@dataclass
class CausalRelationship(CausalPrimitive):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """

    __tablename__ = "CausalRelationship"
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)
    strength = db.Column(db.Float, unique=False)
    reinforcement = db.Column(db.Boolean, unique=False)


@dataclass
class Relationship(CausalPrimitive):
    """API definition of a generic relationship between two primitives """

    __tablename__ = "Relationship"
    source = db.Column(db.Text, primary_key=True)
    target = db.Column(db.Text, unique=False)
    confidence = db.Column(db.Float, unique=False)


@dataclass
class Evidence(db.Model):
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "Evidence"
    id = db.Column(db.String(120), primary_key=True)
    link = db.Column(db.String(120), unique=False)
    description = db.Column(db.String(120), unique=False)
    category = db.Column(db.String(120), unique=False)
    rank = db.Column(db.Integer, unique=False)


@dataclass
class Projection(db.Model):
    """Placeholder docstring for class Projection. """

    __tablename__ = "Projection"
    numSteps = db.Column(db.Integer, primary_key=True)
    stepSize = db.Column(db.String(120), unique=False)
    startTime = db.Column(db.String(120), unique=False)


@dataclass
class Experiment(db.Model):
    """structure used for experimentation """

    __tablename__ = "Experiment"
    id = db.Column(db.String(120), primary_key=True)
    label = db.Column(db.String(120), unique=False)
    options = db.Column(db.Text, unique=False)


@dataclass
class ForwardProjection(Experiment):
    """a foward projection experiment """

    __tablename__ = "ForwardProjection"
    interventions = db.Column(db.Text, primary_key=True)
    projection = db.Column(db.Text, unique=False)


@dataclass
class SensitivityAnalysis(Experiment):
    """a sensitivity analysis experiment """

    __tablename__ = "SensitivityAnalysis"
    variables = db.Column(db.String(120), primary_key=True)


@dataclass
class ExperimentResult(db.Model):
    """Notional model of experiment results """

    __tablename__ = "ExperimentResult"
    id = db.Column(db.String(120), primary_key=True)


@dataclass
class ForwardProjectionResult(ExperimentResult):
    """The result of a forward projection experiment """

    __tablename__ = "ForwardProjectionResult"
    projection = db.Column(db.Text, primary_key=True)
    results = db.Column(db.Text, unique=False)


@dataclass
class SensitivityAnalysisResult(ExperimentResult):
    """The result of a sensitivity analysis experiment """

    __tablename__ = "SensitivityAnalysisResult"
    results = db.Column(db.Text, primary_key=True)


@dataclass
class Traversal(db.Model):
    """Placeholder docstring for class Traversal. """

    __tablename__ = "Traversal"
    maxDepth = db.Column(db.Integer, primary_key=True)


@dataclass
class Version(db.Model):
    """Placeholder docstring for class Version. """

    __tablename__ = "Version"
    icmVersion = db.Column(db.String(120), primary_key=True)
    icmProviderVersion = db.Column(db.String(120), unique=False)