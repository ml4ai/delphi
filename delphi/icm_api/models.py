from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict


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
class User:
    """Placeholder docstring for class User. """

    __tablename__ = "User"
    id = Column(Integer, primary_key=True)
    username = Column(String(120), unique=False)
    firstName = Column(String(120), unique=False)
    lastName = Column(String(120), unique=False)
    email = Column(String(120), unique=False)
    password = Column(String(120), unique=False)
    phone = Column(String(120), unique=False)
    userStatus = Column(Integer, unique=False)


@dataclass
class ICMMetadata:
    """Placeholder docstring for class ICMMetadata. """

    __tablename__ = "ICMMetadata"
    id = Column(String(120), primary_key=True)
    icmProvider = Column(ICMProvider, unique=False)
    title = Column(String(120), unique=False)
    version = Column(Integer, unique=False)
    created = Column(String(120), unique=False)
    createdByUser = Column(User, unique=False)
    lastAccessed = Column(String(120), unique=False)
    lastAccessedByUser = Column(User, unique=False)
    lastUpdated = Column(String(120), unique=False)
    lastUpdatedByUser = Column(User, unique=False)
    estimatedNumberOfPrimitives = Column(Integer, unique=False)
    lifecycleState = Column(LifecycleState, unique=False)
    derivation = Column(String(120), unique=False)


@dataclass
class ServerResponse:
    """Placeholder docstring for class ServerResponse. """

    __tablename__ = "ServerResponse"
    id = Column(String(120), primary_key=True)
    message = Column(String(120), unique=False)


@dataclass
class Range:
    """Top level range object used in a CausalVariable """

    __tablename__ = "Range"


@dataclass
class IntegerRange(Range):
    """The range for an integer value """

    __tablename__ = "IntegerRange"
    range = Column(Object, primary_key=True)


@dataclass
class FloatRange(Range):
    """The range for a floating point value """

    __tablename__ = "FloatRange"
    range = Column(Object, primary_key=True)


@dataclass
class BooleanRange(Range):
    """Denotes a boolean range """

    __tablename__ = "BooleanRange"
    range = Column(Object, primary_key=True)


@dataclass
class EnumRange(Range):
    """The values of an enumeration """

    __tablename__ = "EnumRange"
    range = Column(String(120), primary_key=True)


@dataclass
class DistributionEnumRange(Range):
    """The range of classifications that can be reported in a DistributionEnumValue """

    __tablename__ = "DistributionEnumRange"
    range = Column(String(120), primary_key=True)


@dataclass
class Value:
    """Top level value object used in a TimeSeriesValue """

    __tablename__ = "Value"


@dataclass
class IntegerValue(Value):
    """An integer value """

    __tablename__ = "IntegerValue"
    value = Column(Integer, primary_key=True)


@dataclass
class FloatValue(Value):
    """A floating point value """

    __tablename__ = "FloatValue"
    value = Column(Float, primary_key=True)


@dataclass
class BooleanValue(Value):
    """A boolean value """

    __tablename__ = "BooleanValue"
    value = Column(Boolean, primary_key=True)


@dataclass
class EnumValue(Value):
    """An enumeration value defined in the EnumRange for the CausalVariable """

    __tablename__ = "EnumValue"
    value = Column(String(120), primary_key=True)


@dataclass
class DistributionEnumValue(Value):
    """A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """

    __tablename__ = "DistributionEnumValue"
    value = Column(Object, primary_key=True)


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
class TimeSeriesValue:
    """Time series value at a particular time """

    __tablename__ = "TimeSeriesValue"
    time = Column(String(120), primary_key=True)
    value = Column(Value, unique=False)
    active = Column(TimeSeriesState, unique=False)
    trend = Column(TimeSeriesTrend, unique=False)


@dataclass
class CausalPrimitive:
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """

    __tablename__ = "CausalPrimitive"
    namespaces = Column(Object, primary_key=True)
    types = Column(String(120), unique=False)
    editable = Column(Boolean, unique=False)
    disableable = Column(Boolean, unique=False)
    disabled = Column(Boolean, unique=False)
    id = Column(String(120), unique=False)
    label = Column(String(120), unique=False)
    description = Column(String(120), unique=False)
    lastUpdated = Column(String(120), unique=False)


@dataclass
class Entity(CausalPrimitive):
    """API definition of an entity.  """

    __tablename__ = "Entity"
    confidence = Column(Float, primary_key=True)


@dataclass
class CausalVariable(CausalPrimitive):
    """API definition of a causal variable.  """

    __tablename__ = "CausalVariable"
    range = Column(Range, primary_key=True)
    units = Column(String(120), unique=False)
    backingEntities = Column(String(120), unique=False)
    lastKnownValue = Column(TimeSeriesValue, unique=False)
    confidence = Column(Float, unique=False)


@dataclass
class ConfigurationVariable(CausalPrimitive):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """

    __tablename__ = "ConfigurationVariable"
    units = Column(String(120), primary_key=True)
    lastKnownValue = Column(TimeSeriesValue, unique=False)
    range = Column(Range, unique=False)


@dataclass
class CausalRelationship(CausalPrimitive):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """

    __tablename__ = "CausalRelationship"
    source = Column(Object, primary_key=True)
    target = Column(Object, unique=False)
    confidence = Column(Float, unique=False)
    strength = Column(Float, unique=False)
    reinforcement = Column(Boolean, unique=False)


@dataclass
class Relationship(CausalPrimitive):
    """API definition of a generic relationship between two primitives """

    __tablename__ = "Relationship"
    source = Column(Object, primary_key=True)
    target = Column(Object, unique=False)
    confidence = Column(Float, unique=False)


@dataclass
class Evidence:
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """

    __tablename__ = "Evidence"
    id = Column(String(120), primary_key=True)
    link = Column(String(120), unique=False)
    description = Column(String(120), unique=False)
    category = Column(String(120), unique=False)
    rank = Column(Integer, unique=False)


@dataclass
class Projection:
    """Placeholder docstring for class Projection. """

    __tablename__ = "Projection"
    numSteps = Column(Integer, primary_key=True)
    stepSize = Column(String(120), unique=False)
    startTime = Column(String(120), unique=False)


@dataclass
class Experiment:
    """structure used for experimentation """

    __tablename__ = "Experiment"
    id = Column(String(120), primary_key=True)
    label = Column(String(120), unique=False)
    options = Column(Object, unique=False)


@dataclass
class ForwardProjection(Experiment):
    """a foward projection experiment """

    __tablename__ = "ForwardProjection"
    interventions = Column(Object, primary_key=True)
    projection = Column(Projection, unique=False)


@dataclass
class SensitivityAnalysis(Experiment):
    """a sensitivity analysis experiment """

    __tablename__ = "SensitivityAnalysis"
    variables = Column(String(120), primary_key=True)


@dataclass
class ExperimentResult:
    """Notional model of experiment results """

    __tablename__ = "ExperimentResult"
    id = Column(String(120), primary_key=True)


@dataclass
class ForwardProjectionResult(ExperimentResult):
    """The result of a forward projection experiment """

    __tablename__ = "ForwardProjectionResult"
    projection = Column(Projection, primary_key=True)
    results = Column(Object, unique=False)


@dataclass
class SensitivityAnalysisResult(ExperimentResult):
    """The result of a sensitivity analysis experiment """

    __tablename__ = "SensitivityAnalysisResult"
    results = Column(Object, primary_key=True)


@dataclass
class Traversal:
    """Placeholder docstring for class Traversal. """

    __tablename__ = "Traversal"
    maxDepth = Column(Integer, primary_key=True)


@dataclass
class Version:
    """Placeholder docstring for class Version. """

    __tablename__ = "Version"
    icmVersion = Column(String(120), primary_key=True)
    icmProviderVersion = Column(String(120), unique=False)
