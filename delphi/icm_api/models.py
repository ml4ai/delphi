from enum import Enum, unique
from typing import Optional, List
from dataclasses import dataclass, field, asdict


@unique
class ICMProvider(Enum):
    BAE = 'BAE'
    BBN = 'BBN'
    STR = 'STR'
    DUMMY = 'DUMMY'


@unique
class LifecycleState(Enum):
    PROPOSED = 'PROPOSED'
    APPROVED = 'APPROVED'
    EXPERIMENTAL = 'EXPERIMENTAL'
    OPERATIONAL = 'OPERATIONAL'
    SUSPENDED = 'SUSPENDED'
    ARCHIVED = 'ARCHIVED'
    CREATED = 'CREATED'


@dataclass
class User:
    """Placeholder docstring for class User. """
    id: int = None
    username: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    phone: Optional[str] = None
    userStatus: Optional[int] = None


@dataclass
class ICMMetadata:
    """Placeholder docstring for class ICMMetadata. """
    id: str = None
    icmProvider: Optional[ICMProvider] = None
    title: Optional[str] = None
    version: Optional[int] = None
    created: Optional[str] = None
    createdByUser: Optional[User] = None
    lastAccessed: Optional[str] = None
    lastAccessedByUser: Optional[User] = None
    lastUpdated: Optional[str] = None
    lastUpdatedByUser: Optional[User] = None
    estimatedNumberOfPrimitives: Optional[int] = None
    lifecycleState: Optional[LifecycleState] = None
    derivation: Optional[List[str]] = None


@dataclass
class ServerResponse:
    """Placeholder docstring for class ServerResponse. """
    id: Optional[str] = None
    message: Optional[str] = None


@dataclass
class Range:
    """Top level range object used in a CausalVariable """
    baseType: str = None


@dataclass
class IntegerRange(Range):
    """The range for an integer value """
    range: Optional[object] = None


@dataclass
class FloatRange(Range):
    """The range for a floating point value """
    range: Optional[object] = None


@dataclass
class BooleanRange(Range):
    """Denotes a boolean range """
    range: Optional[object] = None


@dataclass
class EnumRange(Range):
    """The range of values an enumeration value may equal """
    range: Optional[List[str]] = None


@dataclass
class Value:
    """Top level value object used in a TimeSeriesValue """
    baseType: str = None


@dataclass
class IntegerValue(Value):
    """An integer value """
    value: Optional[int] = None


@dataclass
class FloatValue(Value):
    """A floating point value """
    value: Optional[float] = None


@dataclass
class BooleanValue(Value):
    """A boolean value """
    value: Optional[bool] = None


@dataclass
class EnumValue(Value):
    """An enumeration value """
    value: Optional[str] = None


@dataclass
class DistributionValue(Value):
    """A distribution of numeric values ordered from smallest to largest """
    value: Optional[List[float]] = None


@unique
class TimeSeriesState(Enum):
    ACTIVE = 'ACTIVE'
    INACTIVE = 'INACTIVE'
    UNKNOWN = 'UNKNOWN'


@dataclass
class TimeSeriesValue:
    """Time series value at a particular time """
    timestep: int = None
    value: Value = None
    active: Optional[TimeSeriesState] = None
    time: Optional[str] = None


@dataclass
class CausalPrimitive:
    """Top level object that contains common properties that would apply to any causal primitive (variable, relationship, etc.) """
    baseType: str = None
    namespaces: Optional[List[object]] = None
    types: Optional[List[str]] = None
    editable: Optional[bool] = None
    disableable: Optional[bool] = None
    disabled: Optional[bool] = None
    id: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    lastUpdated: Optional[str] = None


@dataclass
class Entity(CausalPrimitive):
    """API definition of an entity.  """
    confidence: Optional[float] = None


@dataclass
class CausalVariable(CausalPrimitive):
    """API definition of a causal variable.  """
    range: Range = None
    units: Optional[str] = None
    backingEntity: Optional[str] = None
    lastKnownValue: Optional[TimeSeriesValue] = None
    confidence: Optional[float] = None


@dataclass
class ConfigurationVariable(CausalPrimitive):
    """Account for pieces of the causal graph that may help interpretation or expose "knobs" that might be editable by the user. """
    lastKnownValue: Optional[TimeSeriesValue] = None


@dataclass
class CausalRelationship(CausalPrimitive):
    """API defintion of a causal relationship. Indicates causality between two causal variables. """
    source: str = None
    target: str = None
    confidence: Optional[float] = None
    strength: Optional[float] = None
    reinforcement: Optional[bool] = None


@dataclass
class Relationship(CausalPrimitive):
    """API definition of a generic relationship between two primitives """
    source: str = None
    target: str = None
    confidence: Optional[float] = None


@dataclass
class Evidence:
    """Object that holds a reference to evidence (either KO from TA1 or human provided). """
    id: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    rank: Optional[int] = None


@dataclass
class Projection:
    """Placeholder docstring for class Projection. """
    numSteps: Optional[int] = None
    stepSize: Optional[str] = None


@dataclass
class Intervention:
    """structure used for experimentation """
    id: Optional[str] = None
    label: Optional[str] = None
    interventions: Optional[List[object]] = None
    projection: Optional[Projection] = None
    options: Optional[object] = None


@dataclass
class InterventionResult:
    """Notional model of experiment results """
    id: Optional[str] = None
    projection: Optional[Projection] = None
    results: Optional[List[object]] = None


@dataclass
class Version:
    """Placeholder docstring for class Version. """
    icmVersion: Optional[str] = None
    icmProviderVersion: Optional[str] = None