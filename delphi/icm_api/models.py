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
    basetype: str = "User"
    id: int = None
    basetype: str = "User"
    username: Optional[str] = None
    basetype: str = "User"
    firstName: Optional[str] = None
    basetype: str = "User"
    lastName: Optional[str] = None
    basetype: str = "User"
    email: Optional[str] = None
    basetype: str = "User"
    password: Optional[str] = None
    basetype: str = "User"
    phone: Optional[str] = None
    basetype: str = "User"
    userStatus: Optional[int] = None


@dataclass
class ICMMetadata:
    """Placeholder docstring for class ICMMetadata. """
    basetype: str = "ICMMetadata"
    id: str = None
    basetype: str = "ICMMetadata"
    icmProvider: Optional[ICMProvider] = None
    basetype: str = "ICMMetadata"
    title: Optional[str] = None
    basetype: str = "ICMMetadata"
    version: Optional[int] = None
    basetype: str = "ICMMetadata"
    created: Optional[str] = None
    basetype: str = "ICMMetadata"
    createdByUser: Optional[User] = None
    basetype: str = "ICMMetadata"
    lastAccessed: Optional[str] = None
    basetype: str = "ICMMetadata"
    lastAccessedByUser: Optional[User] = None
    basetype: str = "ICMMetadata"
    lastUpdated: Optional[str] = None
    basetype: str = "ICMMetadata"
    lastUpdatedByUser: Optional[User] = None
    basetype: str = "ICMMetadata"
    estimatedNumberOfPrimitives: Optional[int] = None
    basetype: str = "ICMMetadata"
    lifecycleState: Optional[LifecycleState] = None
    basetype: str = "ICMMetadata"
    derivation: Optional[List[str]] = None


@dataclass
class ServerResponse:
    """Placeholder docstring for class ServerResponse. """
    basetype: str = "ServerResponse"
    id: Optional[str] = None
    basetype: str = "ServerResponse"
    message: Optional[str] = None


@dataclass
class Range:
    """Top level range object used in a CausalVariable """
    basetype: str = "Range"


@dataclass
class IntegerRange(Range):
    """The range for an integer value """
    basetype: str = "IntegerRange"
    range: Optional[object] = None


@dataclass
class FloatRange(Range):
    """The range for a floating point value """
    basetype: str = "FloatRange"
    range: Optional[object] = None


@dataclass
class BooleanRange(Range):
    """Denotes a boolean range """
    basetype: str = "BooleanRange"
    range: Optional[object] = None


@dataclass
class EnumRange(Range):
    """The values of an enumeration """
    basetype: str = "EnumRange"
    range: Optional[List[str]] = None


@dataclass
class DistributionEnumRange(Range):
    """The range of classifications that can be reported in a DistributionEnumValue """
    basetype: str = "DistributionEnumRange"
    range: Optional[List[str]] = None


@dataclass
class Value:
    """Top level value object used in a TimeSeriesValue """
    basetype: str = "Value"


@dataclass
class IntegerValue(Value):
    """An integer value """
    basetype: str = "IntegerValue"
    value: Optional[int] = None


@dataclass
class FloatValue(Value):
    """A floating point value """
    basetype: str = "FloatValue"
    value: Optional[float] = None


@dataclass
class BooleanValue(Value):
    """A boolean value """
    basetype: str = "BooleanValue"
    value: Optional[bool] = None


@dataclass
class EnumValue(Value):
    """An enumeration value defined in the EnumRange for the CausalVariable """
    basetype: str = "EnumValue"
    value: Optional[str] = None


@dataclass
class DistributionEnumValue(Value):
    """A distribution of classifications with non-zero probabilities. The classifications must be defined in the DistributionEnumRange of the CausalVariable and the probabilities must add to 1.0. """
    basetype: str = "DistributionEnumValue"
    value: Optional[object] = None


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
    basetype: str = "TimeSeriesValue"
    time: str = None
    basetype: str = "TimeSeriesValue"
    value: Value = None
    basetype: str = "TimeSeriesValue"
    active: Optional[TimeSeriesState] = None
    basetype: str = "TimeSeriesValue"
    trend: Optional[TimeSeriesTrend] = None


@dataclass
class CausalPrimitive:
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """
    basetype: str = "CausalPrimitive"
    basetype: str = "CausalPrimitive"
    namespaces: Optional[object] = None
    basetype: str = "CausalPrimitive"
    types: Optional[List[str]] = None
    basetype: str = "CausalPrimitive"
    editable: Optional[bool] = True
    basetype: str = "CausalPrimitive"
    disableable: Optional[bool] = True
    basetype: str = "CausalPrimitive"
    disabled: Optional[bool] = False
    basetype: str = "CausalPrimitive"
    id: Optional[str] = None
    basetype: str = "CausalPrimitive"
    label: Optional[str] = None
    basetype: str = "CausalPrimitive"
    description: Optional[str] = None
    basetype: str = "CausalPrimitive"
    lastUpdated: Optional[str] = None


@dataclass
class Entity(CausalPrimitive):
    """API definition of an entity.  """
    basetype: str = "Entity"
    confidence: Optional[float] = None


@dataclass
class CausalVariable(CausalPrimitive):
    """API definition of a causal variable.  """
    basetype: str = "CausalVariable"
    range: Range = None
    basetype: str = "CausalVariable"
    units: Optional[str] = None
    basetype: str = "CausalVariable"
    backingEntities: Optional[List[str]] = None
    basetype: str = "CausalVariable"
    lastKnownValue: Optional[TimeSeriesValue] = None
    basetype: str = "CausalVariable"
    confidence: Optional[float] = None


@dataclass
class ConfigurationVariable(CausalPrimitive):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """
    basetype: str = "ConfigurationVariable"
    units: Optional[str] = None
    basetype: str = "ConfigurationVariable"
    lastKnownValue: Optional[TimeSeriesValue] = None
    basetype: str = "ConfigurationVariable"
    range: Optional[Range] = None


@dataclass
class CausalRelationship(CausalPrimitive):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """
    basetype: str = "CausalRelationship"
    source: object = None
    basetype: str = "CausalRelationship"
    target: object = None
    basetype: str = "CausalRelationship"
    confidence: Optional[float] = None
    basetype: str = "CausalRelationship"
    strength: Optional[float] = None
    basetype: str = "CausalRelationship"
    reinforcement: Optional[bool] = None


@dataclass
class Relationship(CausalPrimitive):
    """API definition of a generic relationship between two primitives """
    basetype: str = "Relationship"
    source: object = None
    basetype: str = "Relationship"
    target: object = None
    basetype: str = "Relationship"
    confidence: Optional[float] = None


@dataclass
class Evidence:
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """
    basetype: str = "Evidence"
    id: Optional[str] = None
    basetype: str = "Evidence"
    link: Optional[str] = None
    basetype: str = "Evidence"
    description: Optional[str] = None
    basetype: str = "Evidence"
    category: Optional[str] = None
    basetype: str = "Evidence"
    rank: Optional[int] = None


@dataclass
class Projection:
    """Placeholder docstring for class Projection. """
    basetype: str = "Projection"
    numSteps: int = None
    basetype: str = "Projection"
    stepSize: str = None
    basetype: str = "Projection"
    startTime: Optional[str] = None


@dataclass
class Experiment:
    """structure used for experimentation """
    basetype: str = "Experiment"
    id: Optional[str] = None
    basetype: str = "Experiment"
    label: Optional[str] = None
    basetype: str = "Experiment"
    options: Optional[object] = None


@dataclass
class ForwardProjection(Experiment):
    """a foward projection experiment """
    basetype: str = "ForwardProjection"
    interventions: Optional[List[object]] = None
    basetype: str = "ForwardProjection"
    projection: Optional[Projection] = None


@dataclass
class SensitivityAnalysis(Experiment):
    """a sensitivity analysis experiment """
    basetype: str = "SensitivityAnalysis"
    variables: Optional[List[str]] = None


@dataclass
class ExperimentResult:
    """Notional model of experiment results """
    basetype: str = "ExperimentResult"
    id: Optional[str] = None


@dataclass
class ForwardProjectionResult(ExperimentResult):
    """The result of a forward projection experiment """
    basetype: str = "ForwardProjectionResult"
    projection: Optional[Projection] = None
    basetype: str = "ForwardProjectionResult"
    results: Optional[List[object]] = None


@dataclass
class SensitivityAnalysisResult(ExperimentResult):
    """The result of a sensitivity analysis experiment """
    basetype: str = "SensitivityAnalysisResult"
    results: Optional[List[object]] = None


@dataclass
class Traversal:
    """Placeholder docstring for class Traversal. """
    basetype: str = "Traversal"
    maxDepth: Optional[int] = None


@dataclass
class Version:
    """Placeholder docstring for class Version. """
    basetype: str = "Version"
    icmVersion: Optional[str] = None
    basetype: str = "Version"
    icmProviderVersion: Optional[str] = None